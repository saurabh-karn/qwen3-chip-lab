#!/usr/bin/env python3
"""Multi-hardware bit-exact verify: run all backends in parallel.

Each backend runs the full forward on the same tokens and ROM, producing a
trace + checkpoints that lab.compare checks bit-for-bit against the Python
oracle. Exits non-zero if any backend fails.

Usage:
  python3 -m lab.hw_verify --token-ids 785,6722,315 --tier T4 [--backends gpu,tpu]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lab.compare import compare
from lab.hw import BACKENDS, ROOT, RUNS_ROOT, SIM_DIR, Backend, run_backend
from lab.verify import (
    DEFAULT_MANIFEST,
    DEFAULT_ROM,
    _filter_events,
    _oracle_cache_valid,
    _oracle_fingerprint,
    _read_records,
    _tokenize,
    _validate_rom,
    _write_filtered,
)
from qwen_ref.manifest import LAYOUT_SHA256, ROM_BYTES
from qwen_ref.model import forward
from qwen_ref.rom import FlatBF16ROM
from qwen_ref.trace import TraceWriter


def run_oracle(
    token_ids: list[int],
    rom: Path,
    manifest: Path,
    work: Path,
) -> list[float]:
    """Run (or reuse) the Python oracle; return the final logits row.

    The oracle is deterministic in (tokens, ROM, oracle source), so a
    matching fingerprint means the cached trace is bit-identical to a fresh
    run — reuse skips a ~25-minute recompute with zero accuracy cost.
    """
    manifest_doc = _validate_rom(rom, manifest)
    fingerprint = _oracle_fingerprint(token_ids, manifest_doc.get("rom_sha256"))
    logits_path = work / "python_checkpoints" / "logits.f32le"
    if _oracle_cache_valid(work, fingerprint) and logits_path.is_file():
        import struct

        blob = logits_path.read_bytes()
        row = struct.unpack(f"<{len(blob) // 4 // 151936 * 151936}f", blob)
        return list(row[-151936:])
    work.mkdir(parents=True, exist_ok=True)
    (work / "python_checkpoints").mkdir(exist_ok=True)
    trace = TraceWriter(work / "python.ndjson", work / "python_checkpoints")
    try:
        with FlatBF16ROM(rom, manifest) as weights:
            logits = forward(token_ids, weights, trace=trace)
    finally:
        trace.close()
    (work / "oracle_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
    return logits[-1]


def run_one_backend(
    backend: Backend,
    token_ids: list[int],
    rom: Path,
    work: Path,
    tier: str,
) -> dict[str, Any]:
    """Build + run one backend and compare against the oracle artifacts."""
    meta = run_backend(backend, token_ids, rom, work)

    py_filtered = work / "python.filtered.ndjson"
    rtl_filtered = work / "rtl.filtered.ndjson"
    # work dir is "<run_key>_<backend>"; the oracle dir is "<run_key>_oracle".
    oracle_work = work.parent / (work.name[: -len(backend.key) - 1] + "_oracle")
    if not py_filtered.is_file():
        py_records = _filter_events(
            _read_records(oracle_work / "python.ndjson"), tier=tier, token_count=len(token_ids)
        )
        _write_filtered(py_records, py_filtered)
    rtl_records = _filter_events(
        _read_records(work / "rtl.ndjson"), tier=tier, token_count=len(token_ids)
    )
    _write_filtered(rtl_records, rtl_filtered)

    comparison = compare(
        py_filtered,
        rtl_filtered,
        oracle_work / "python_checkpoints",
        work / "rtl_checkpoints",
    )
    return {
        "backend": backend.key,
        "label": backend.label,
        "passed": bool(comparison.get("passed")),
        "mismatch_count": comparison.get("mismatch_count"),
        "first_mismatch": comparison.get("first_mismatch"),
        "rtl_meta": meta.get("rtl_meta"),
        "work_dir": str(work),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m lab.hw_verify")
    parser.add_argument("--text")
    parser.add_argument("--token-ids", help="comma-separated token ids")
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tier", default="T4")
    parser.add_argument(
        "--backends",
        default="mask_rom,mask_rom_fused,gpu,tpu,lpu,gpu_hbm,tpu_hbm,lpu_hbm",
        help="comma-separated subset of: mask_rom,mask_rom_fused,gpu,tpu,"
             "lpu,gpu_hbm,tpu_hbm,lpu_hbm",
    )
    parser.add_argument("--run-key", help="run dir key (default: hw_<ntokens>tok)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    tokenizer = ROOT / "artifacts" / "tokenizer.json"
    token_ids: list[int] | None = None
    if args.token_ids:
        token_ids = [int(p) for p in args.token_ids.split(",") if p]
    if token_ids is None:
        if not args.text:
            parser.error("provide --text or --token-ids")
        token_ids = _tokenize(args.text, tokenizer)

    backend_keys = [k.strip() for k in args.backends.split(",") if k.strip()]
    unknown = [k for k in backend_keys if k not in BACKENDS]
    if unknown:
        parser.error(f"unknown backends: {unknown}")

    run_key = args.run_key or f"hw_{len(token_ids)}tok"
    oracle_work = RUNS_ROOT / f"{run_key}_oracle"

    print(f"oracle: {len(token_ids)} tokens -> {oracle_work}", flush=True)
    logits = run_oracle(token_ids, args.rom, args.manifest, oracle_work)
    oracle_argmax = max(range(len(logits)), key=lambda i: logits[i])
    print(f"oracle argmax: {oracle_argmax}", flush=True)

    # Build every distinct binary BEFORE fanning the runs out (user
    # directive: build first, then verify in parallel). Backends share
    # binaries across memory scenarios, so dedupe by FUSED_SCHEDULE.
    for fused in sorted({BACKENDS[k].fused for k in backend_keys}):
        print(f"build: FUSED={fused} -> obj_dir_fused{fused}", flush=True)
        subprocess.run(
            ["make", "-C", str(SIM_DIR), "all", f"FUSED={fused}"],
            check=True,
            cwd=ROOT,
        )
        binary = SIM_DIR / f"obj_dir_fused{fused}" / "Vtb_qwen3_full_forward"
        if not binary.is_file():
            raise FileNotFoundError(f"build did not produce {binary}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {}
        for key in backend_keys:
            backend = BACKENDS[key]
            work = backend.work_dir(run_key)
            futures[pool.submit(
                run_one_backend, backend, token_ids, args.rom, work, args.tier
            )] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - report, don't crash the fan
                result = {
                    "backend": key,
                    "label": BACKENDS[key].label,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result.setdefault("memory", BACKENDS[key].memory)
            result.setdefault("ext_mem_lat",
                              BACKENDS[key].ext_mem_lat
                              if BACKENDS[key].memory == "hbm" else 0)
            results.append(result)
            status = "PASS" if result.get("passed") else "FAIL"
            print(f"[{status}] {key}: {result.get('label')}", flush=True)

    results.sort(key=lambda r: r["backend"])
    all_passed = all(r.get("passed") for r in results) and bool(results)
    summary = {
        "schema_version": 1,
        "tier": args.tier,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "oracle_argmax": oracle_argmax,
        "backends": results,
        "passed": all_passed,
    }
    out = RUNS_ROOT / f"{run_key}_hw_verify.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
