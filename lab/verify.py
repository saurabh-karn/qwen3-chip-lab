#!/usr/bin/env python3
"""Bit-exact Python vs production RTL forward for Qwen3-0.6B."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from lab.compare import compare
from lab.rtl_trace import GEOM, VOCAB, write_rtl_trace
from qwen_ref.fp import ARITHMETIC_PROFILE
from qwen_ref.manifest import LAYOUT_SHA256, ROM_BYTES
from qwen_ref.model import forward
from qwen_ref.rom import FlatBF16ROM
from qwen_ref.tokenizer import Tokenizer
from qwen_ref.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "evidence" / "runs"
DEFAULT_ROM = ROOT / "artifacts" / "qwen3.bf16rom"
DEFAULT_MANIFEST = ROOT / "artifacts" / "qwen3.bf16rom.json"
PROFILE_PATH = ROOT / "spec" / "arithmetic" / "qwen-ref-v2" / "profile.json"
SIM_DIR = ROOT / "qwen_chip" / "sim"
MAX_POS = 128
T3_LAYERS = {0, 13, 27}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_rom(rom: Path, manifest: Path) -> dict[str, Any]:
    if not rom.is_file():
        raise FileNotFoundError(
            f"packed ROM not found: {rom}. Stage artifacts/qwen3.bf16rom "
            "locally; this loop will not shrink geometry or skip the image."
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"ROM manifest not found: {manifest}")
    document = _load_json(manifest)
    expected = {
        "format": "qwen_ref.flat-bf16-rom.v1",
        "dtype": "BF16",
        "endianness": "little",
        "arithmetic_profile": ARITHMETIC_PROFILE,
        "layout_sha256": LAYOUT_SHA256,
        "rom_bytes": ROM_BYTES,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"ROM manifest has invalid {key}")
    size = rom.stat().st_size
    if size != ROM_BYTES:
        raise ValueError(f"ROM is {size} bytes; expected {ROM_BYTES}")
    with FlatBF16ROM(rom, manifest):
        pass
    return document


def _oracle_fingerprint(token_ids: list[int], rom_sha256: str | None) -> str:
    """Identity of a Python-oracle run: tokens + ROM + oracle source.

    The oracle is a pure function of token IDs, weights, and the arithmetic
    profile. Hashing the oracle source files (fp, model, trace, tokenizer)
    into the key means any code change invalidates cached traces.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(json.dumps(token_ids).encode())
    h.update((rom_sha256 or "unknown").encode())
    for name in ("fp.py", "model.py", "trace.py", "tokenizer.py", "config.py"):
        path = ROOT / "qwen_ref" / name
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _oracle_cache_valid(work: Path, fingerprint: str) -> bool:
    """A cached oracle run exists and matches this fingerprint.

    The oracle output depends only on (tokens, ROM, oracle source) — not on
    the RTL schedule — so any run dir holding a matching fingerprint is a
    valid source. If another run dir has the trace, copy it here.
    """
    marker = work / "oracle_fingerprint.txt"

    def _complete(directory: Path) -> bool:
        trace = directory / "python.ndjson"
        if not trace.is_file() or trace.stat().st_size == 0:
            return False
        try:
            last_line = trace.read_text(encoding="utf-8").strip().splitlines()[-1]
            return json.loads(last_line).get("event") == "logits"
        except (OSError, IndexError, json.JSONDecodeError):
            return False

    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == fingerprint:
        if _complete(work):
            return True
    # Search sibling run dirs for a matching, complete oracle trace.
    if RUNS_ROOT.is_dir():
        for candidate in RUNS_ROOT.iterdir():
            if candidate == work or not candidate.is_dir():
                continue
            cmark = candidate / "oracle_fingerprint.txt"
            if not cmark.is_file():
                continue
            if cmark.read_text(encoding="utf-8").strip() != fingerprint:
                continue
            if not _complete(candidate):
                continue
            # Seed this work dir from the cached oracle.
            import shutil
            work.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate / "python.ndjson", work / "python.ndjson")
            src_ckpts = candidate / "python_checkpoints"
            dst_ckpts = work / "python_checkpoints"
            dst_ckpts.mkdir(parents=True, exist_ok=True)
            if src_ckpts.is_dir():
                for item in src_ckpts.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst_ckpts / item.name)
            marker.write_text(fingerprint + "\n", encoding="utf-8")
            return True
    return False


def _tokenize(text: str, tokenizer_path: Path) -> list[int]:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"tokenizer.json not found: {tokenizer_path}. "
            "Pass --tokenizer or set TOKENIZER."
        )
    token_ids = Tokenizer(tokenizer_path).encode(text)
    if not token_ids:
        raise ValueError("tokenizer produced an empty statement")
    if len(token_ids) > MAX_POS:
        raise ValueError(
            f"statement has {len(token_ids)} tokens; max is {MAX_POS}"
        )
    return token_ids


def _filter_events(
    records: Iterable[dict[str, Any]],
    *,
    tier: str,
    token_count: int,
) -> list[dict[str, Any]]:
    del token_count
    kept = []
    for record in records:
        event = str(record.get("event", ""))
        layer = record.get("layer", -1)
        if tier == "T2":
            if event == "embedding" or (
                isinstance(layer, int) and layer == 0 and event.startswith("layer.0.")
            ):
                kept.append(record)
        elif tier == "T3":
            if event == "embedding":
                kept.append(record)
            elif isinstance(layer, int) and layer in T3_LAYERS:
                kept.append(record)
        else:
            kept.append(record)
    return kept


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def _job_pid_path(work: Path) -> Path:
    return work / "job.pid"


def _write_job_pid(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    _job_pid_path(work).write_text(str(os.getpid()) + "\n", encoding="utf-8")


def _clear_job_pid(work: Path) -> None:
    try:
        _job_pid_path(work).unlink()
    except OSError:
        pass


def _job_pid_alive(work: Path) -> bool | None:
    """True / False if a pid file exists; None if this is a legacy dir."""
    path = _job_pid_path(work)
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _mark_work_idle(work: Path) -> None:
    """Clear leftover busy flags so a dead job cannot look in-flight."""
    _clear_job_pid(work)
    py_path = work / "python_status.json"
    if not py_path.is_file():
        return
    try:
        py = _load_json(py_path)
    except (OSError, json.JSONDecodeError):
        return
    if py.get("busy"):
        py["busy"] = False
        py["message"] = py.get("message") or "idle"
        _write_json_atomic(py_path, py)


def work_is_busy(work: Path) -> bool:
    """True only while Python or Verilator is actually running.

    A live job writes job.pid. A leftover python_status.busy after the
    process died (or after RTL finished with no pid file) is idle so the
    UI cannot attach to a ghost.
    """
    if not work.is_dir():
        return False
    alive = _job_pid_alive(work)
    if alive is True:
        return True
    if alive is False:
        return False
    rtl_path = work / "rtl" / "rtl_status.json"
    rtl = _load_json(rtl_path) if rtl_path.is_file() else {}
    if rtl.get("busy"):
        return True
    py_path = work / "python_status.json"
    py = _load_json(py_path) if py_path.is_file() else {}
    if not py.get("busy"):
        return False
    if not rtl:
        return True
    tokens = int(rtl.get("token_count") or 0)
    done = int(rtl.get("token_index") or 0)
    if tokens and done >= tokens:
        return False
    if int(rtl.get("cycle") or 0) > 0 and rtl.get("stage") in {"IDLE", "DONE"}:
        return False
    return True


def _rtl_meta_from_status(work: Path) -> dict[str, int]:
    status = _load_json(work / "rtl" / "rtl_status.json") or {}
    return {
        "cycle_count": int(status.get("cycle") or 0),
        "rom_read_count": int(status.get("rom_reads") or 0),
        "sram_read_count": int(status.get("sram_reads") or 0),
        "sram_write_count": int(status.get("sram_writes") or 0),
        "mac_count": int(status.get("macs") or 0),
        "stall_count": int(status.get("stalls") or 0),
        "argmax": int(status.get("argmax") or 0),
    }


def _write_filtered(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _ensure_sim(fused: bool = False) -> Path:
    # Each FUSED schedule builds into its own directory, so both binaries
    # coexist and parallel verify jobs never rebuild each other's binary.
    obj_dir = SIM_DIR / f"obj_dir_fused{1 if fused else 0}"
    binary = obj_dir / "Vtb_qwen3_full_forward"
    makefile = SIM_DIR / "Makefile"
    subprocess.run(
        ["make", "-C", str(SIM_DIR), "all", f"FUSED={1 if fused else 0}"],
        check=True,
        cwd=ROOT,
    )
    if not binary.is_file() and not obj_dir.exists():
        raise FileNotFoundError(f"Verilator build did not produce {binary}")
    built = binary if binary.is_file() else None
    if built is None:
        candidates = list(obj_dir.glob("Vtb_qwen3_full_forward*"))
        executable = [
            path for path in candidates
            if path.is_file() and os.access(path, os.X_OK) and path.suffix == ""
        ]
        if not executable:
            raise FileNotFoundError(
                f"Verilator executable missing after make -C {SIM_DIR}"
            )
        built = executable[0]
    del makefile
    return built


def _run_rtl(
    binary: Path,
    token_ids: list[int],
    rom: Path,
    work: Path,
    *,
    dump_trace: bool = True,
) -> dict[str, int]:
    token_file = work / "tokens.txt"
    token_file.write_text("\n".join(str(token) for token in token_ids) + "\n")
    trace_dir = work / "rtl"
    trace_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        f"+ROM_FILE={rom}",
        f"+TOKEN_FILE={token_file}",
        f"+TRACE_DIR={trace_dir}",
    ]
    subprocess.run(command, check=True, cwd=trace_dir)
    if not dump_trace:
        return _rtl_meta_from_status(work)
    log = trace_dir / "rtl_commits.bin"
    if not log.is_file():
        raise FileNotFoundError(f"RTL commit log missing: {log}")
    return write_rtl_trace(
        log,
        work / "rtl.ndjson",
        work / "rtl_checkpoints",
    )


def verify(
    *,
    text: str | None,
    token_ids: list[int] | None,
    rom: Path,
    manifest: Path,
    tokenizer: Path,
    work: Path,
    tier: str,
    fused: bool = False,
) -> dict[str, Any]:
    tier = tier.upper()
    if tier not in {"T2", "T3", "T4", "T5"}:
        raise ValueError("tier must be T2, T3, T4, or T5")
    manifest_doc = _validate_rom(rom, manifest)
    if token_ids is None:
        if text is None:
            raise ValueError("provide --text or --token-ids")
        token_ids = _tokenize(text, tokenizer)
    elif text is None:
        text = ""
    if not token_ids:
        raise ValueError("at least one token is required")
    if len(token_ids) > MAX_POS:
        raise ValueError(f"statement has {len(token_ids)} tokens; max is {MAX_POS}")
    if tier == "T5" and len(token_ids) != MAX_POS:
        raise ValueError(f"T5 requires exactly {MAX_POS} tokens, got {len(token_ids)}")
    if GEOM["layers"] != 28 or GEOM["hidden"] != 1024 or GEOM["vocab"] != VOCAB:
        raise AssertionError("production geometry drifted")

    work.mkdir(parents=True, exist_ok=True)
    (work / "rtl").mkdir(parents=True, exist_ok=True)
    python_trace = work / "python.ndjson"
    python_ckpts = work / "python_checkpoints"
    python_ckpts.mkdir(parents=True, exist_ok=True)
    _write_job_pid(work)
    try:
        return _verify_body(
            text=text,
            token_ids=token_ids,
            rom=rom,
            manifest=manifest,
            manifest_doc=manifest_doc,
            work=work,
            tier=tier,
            fused=fused,
            python_trace=python_trace,
            python_ckpts=python_ckpts,
        )
    finally:
        _mark_work_idle(work)


def _verify_body(
    *,
    text: str,
    token_ids: list[int],
    rom: Path,
    manifest: Path,
    manifest_doc: dict[str, Any],
    work: Path,
    tier: str,
    fused: bool,
    python_trace: Path,
    python_ckpts: Path,
) -> dict[str, Any]:
    def _python_progress(done: int, total: int, message: str) -> None:
        _write_json_atomic(work / "python_status.json", {
            "phase": "python",
            "busy": True,
            "layer": max(done - 1, 0),
            "layers_done": done,
            "layers_total": total,
            "message": message,
        })

    # The oracle is deterministic in (tokens, ROM, oracle source). Reuse a
    # cached trace when the fingerprint matches: this skips a ~25-minute
    # re-run with zero accuracy cost, since the comparison still checks
    # every checkpoint bit-for-bit against this exact trace.
    fingerprint = _oracle_fingerprint(token_ids, manifest_doc.get("rom_sha256"))
    logits = None
    if _oracle_cache_valid(work, fingerprint):
        (work / "oracle_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
        _write_json_atomic(work / "python_status.json", {
            "phase": "rtl",
            "busy": True,
            "layer": GEOM["layers"] - 1,
            "layers_done": GEOM["layers"],
            "layers_total": GEOM["layers"],
            "message": "Python oracle cached (fingerprint match); starting Verilator",
        })
        # Recover the final logits row from the cached checkpoint for argmax.
        logits_path = python_ckpts / "logits.f32le"
        if logits_path.is_file():
            import struct as _struct
            blob = logits_path.read_bytes()
            row = _struct.unpack(f"<{VOCAB}f", blob[-VOCAB * 4:])
            logits = [list(row)]
    else:
        _write_json_atomic(work / "python_status.json", {
            "phase": "python",
            "busy": True,
            "layer": 0,
            "layers_done": 0,
            "layers_total": GEOM["layers"],
            "message": "Python oracle starting",
        })
        trace = TraceWriter(python_trace, python_ckpts)
        try:
            with FlatBF16ROM(rom, manifest) as weights:
                logits = forward(token_ids, weights, trace=trace, progress=_python_progress)
        finally:
            trace.close()
        (work / "oracle_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")

    _write_oracle_result(work, token_ids, logits)
    _write_json_atomic(work / "python_status.json", {
        "phase": "rtl",
        "busy": True,
        "layer": GEOM["layers"] - 1,
        "layers_done": GEOM["layers"],
        "layers_total": GEOM["layers"],
        "message": "Python oracle complete; starting Verilator",
    })
    binary = _ensure_sim(fused=fused)
    rtl_meta = _run_rtl(binary, token_ids, rom, work)
    rtl_trace = work / "rtl.ndjson"
    rtl_ckpts = work / "rtl_checkpoints"

    py_records = _filter_events(_read_records(python_trace), tier=tier, token_count=len(token_ids))
    rtl_records = _filter_events(_read_records(rtl_trace), tier=tier, token_count=len(token_ids))
    py_filtered = work / "python.filtered.ndjson"
    rtl_filtered = work / "rtl.filtered.ndjson"
    _write_filtered(py_records, py_filtered)
    _write_filtered(rtl_records, rtl_filtered)
    comparison = compare(py_filtered, rtl_filtered, python_ckpts, rtl_ckpts)
    if tier in {"T4", "T5"}:
        comparison["full_logits_compared"] = any(
            record.get("event") == "logits" for record in py_records
        ) and comparison["full_logits_compared"]

    evidence = {
        "schema_version": 1,
        "tier": tier,
        "text": text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "arithmetic_profile": ARITHMETIC_PROFILE,
        "layout_sha256": LAYOUT_SHA256,
        "rom_sha256": manifest_doc.get("rom_sha256"),
        "rom_bytes": ROM_BYTES,
        "git_commit": _git_commit(),
        "geometry": GEOM,
        "mac_lanes": 16,
        "rom_data_w": 256,
        "sram_data_w": 256,
        "debug_short_mode": 0,
        "fused_schedule": 1 if fused else 0,
        "rtl_meta": rtl_meta,
        "python_argmax": _argmax(logits[-1]) if logits else None,
        "rtl_argmax": rtl_meta.get("argmax"),
        "comparison": comparison,
        "passed": bool(comparison.get("passed")),
        "work_dir": str(work),
    }
    (work / "verify.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def infer(
    *,
    text: str | None,
    token_ids: list[int] | None,
    rom: Path,
    manifest: Path,
    tokenizer: Path,
    work: Path,
    fused: bool = True,
) -> dict[str, Any]:
    """Host-token RTL forward after the ROM has been T4-signed.

    Python oracle and checkpoint compare are skipped. Weights stay in ROM;
    the host only supplies token IDs. Trace dump is skipped — live status
    still streams from rtl_status.json.
    """
    manifest_doc = _validate_rom(rom, manifest)
    if token_ids is None:
        if text is None:
            raise ValueError("provide --text or --token-ids")
        token_ids = _tokenize(text, tokenizer)
    elif text is None:
        text = ""
    if not token_ids:
        raise ValueError("at least one token is required")
    if len(token_ids) > MAX_POS:
        raise ValueError(f"statement has {len(token_ids)} tokens; max is {MAX_POS}")

    work.mkdir(parents=True, exist_ok=True)
    (work / "rtl").mkdir(parents=True, exist_ok=True)
    _write_job_pid(work)
    try:
        _write_json_atomic(work / "python_status.json", {
            "phase": "rtl",
            "busy": True,
            "layer": 0,
            "layers_done": GEOM["layers"],
            "layers_total": GEOM["layers"],
            "message": "ROM signed off · RTL only (no Python compare)",
        })
        binary = _ensure_sim(fused=fused)
        rtl_meta = _run_rtl(binary, token_ids, rom, work, dump_trace=False)
        evidence = {
            "schema_version": 1,
            "mode": "rtl_infer",
            "tier": "RTL",
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "arithmetic_profile": ARITHMETIC_PROFILE,
            "layout_sha256": LAYOUT_SHA256,
            "rom_sha256": manifest_doc.get("rom_sha256"),
            "rom_bytes": ROM_BYTES,
            "git_commit": _git_commit(),
            "geometry": GEOM,
            "mac_lanes": 16,
            "fused_schedule": 1 if fused else 0,
            "rtl_meta": rtl_meta,
            "python_argmax": None,
            "rtl_argmax": rtl_meta.get("argmax"),
            "passed": True,
            "compared": False,
            "work_dir": str(work),
        }
        (work / "infer.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        return evidence
    finally:
        _mark_work_idle(work)


def _write_oracle_result(work: Path, token_ids: list[int], logits: list | None) -> None:
    if not logits:
        return
    _write_json_atomic(work / "oracle_result.json", {
        "python_argmax": _argmax(logits[-1]),
        "token_count": len(token_ids),
    })


def _argmax(row: list[float]) -> int:
    best = 0
    for index, value in enumerate(row):
        if value > row[best]:
            best = index
    return best


def cache_key(token_ids: list[int], tier: str, fused: bool = False) -> str:
    import hashlib

    payload = json.dumps(
        {"tier": tier, "token_ids": token_ids, "fused": bool(fused)},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m lab.verify")
    parser.add_argument("--text")
    parser.add_argument("--token-ids", help="comma-separated token ids")
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(os.environ.get("TOKENIZER", ROOT / "artifacts" / "tokenizer.json")),
    )
    parser.add_argument(
        "--tier",
        default="T4",
        help="gate id: T4 = full-statement compare (default); "
        "T2 = one layer; T3 = sampled layers; T5 = 128-token bound",
    )
    parser.add_argument("--fused", action="store_true",
                        help="run the fused streaming-GEMM RTL schedule")
    parser.add_argument("--infer", action="store_true",
                        help="RTL-only forward; skip Python oracle and T4 compare")
    parser.add_argument("--work", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "verify.json",
    )
    args = parser.parse_args(argv)
    token_ids = None
    if args.token_ids:
        token_ids = [int(part) for part in args.token_ids.split(",") if part]
    try:
        ids_for_cache = token_ids
        if ids_for_cache is None and args.text:
            ids_for_cache = _tokenize(args.text, args.tokenizer)
        key = cache_key(ids_for_cache or [], "RTL" if args.infer else args.tier.upper(), args.fused)
        work = args.work or (ROOT / "evidence" / "runs" / key)
        if args.infer:
            evidence = infer(
                text=args.text,
                token_ids=token_ids,
                rom=args.rom,
                manifest=args.manifest,
                tokenizer=args.tokenizer,
                work=work,
                fused=args.fused,
            )
        else:
            evidence = verify(
                text=args.text,
                token_ids=token_ids,
                rom=args.rom,
                manifest=args.manifest,
                tokenizer=args.tokenizer,
                work=work,
                tier=args.tier,
                fused=args.fused,
            )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.infer:
        print(json.dumps({
            "passed": True,
            "mode": "rtl_infer",
            "compared": False,
            "token_count": evidence["token_count"],
            "rtl_argmax": evidence.get("rtl_argmax"),
            "output": str(work / "infer.json"),
        }, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text_out = json.dumps(evidence, indent=2) + "\n"
    args.output.write_text(text_out, encoding="utf-8")
    (work / "verify.json").write_text(text_out, encoding="utf-8")
    print(json.dumps({
        "passed": evidence["passed"],
        "tier": evidence["tier"],
        "token_count": evidence["token_count"],
        "mismatch_count": evidence["comparison"]["mismatch_count"],
        "first_mismatch": evidence["comparison"]["first_mismatch"],
        "output": str(args.output),
    }, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
