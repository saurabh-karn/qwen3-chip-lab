"""FastAPI live tile: tokenize, run lab.verify T4, stream RTL status."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi.responses import Response
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from lab.verify import DEFAULT_ROM, _tokenize, cache_key, work_is_busy
from . import auth as lab_auth

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
EVIDENCE = ROOT / "evidence"
VERIFY_JSON = EVIDENCE / "verify.json"
RUNS = EVIDENCE / "runs"
SIGNED_OFF = EVIDENCE / "rtl_signed_off.json"
TOKENIZER = Path(os.environ.get("TOKENIZER", ROOT / "artifacts" / "tokenizer.json"))
ROM = Path(os.environ.get("ROM_FILE", DEFAULT_ROM))
MANIFEST = Path(str(ROM) + ".json")

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None and TOKENIZER.is_file():
        from qwen_ref.tokenizer import Tokenizer
        _tokenizer = Tokenizer(TOKENIZER)
    return _tokenizer


def _decode_ids(token_ids: list[int]) -> str:
    tok = _get_tokenizer()
    if tok is None or not token_ids:
        return ""
    try:
        return tok.decode(token_ids)
    except ValueError:
        return ""


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1)
    tier: str = "T4"
    force_verify: bool = False


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1)


class LoginRequest(BaseModel):
    pin: str = Field(min_length=8, max_length=8, pattern=r"^\d{8}$")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rom_sha256() -> str | None:
    doc = _read_json(MANIFEST) or {}
    sha = doc.get("rom_sha256")
    return str(sha) if sha else None


def _signed_off() -> dict[str, Any] | None:
    """ROM is trusted after one passing fused T4 for this image."""
    doc = _read_json(SIGNED_OFF)
    if not doc:
        return None
    current = _rom_sha256()
    if not current or doc.get("rom_sha256") != current:
        return None
    if doc.get("passed") is False:
        return None
    return doc


def _record_signoff(evidence: dict[str, Any]) -> None:
    if not evidence.get("passed") or evidence.get("mode") == "rtl_infer":
        return
    if not evidence.get("fused_schedule"):
        return
    SIGNED_OFF.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "passed": True,
        "rom_sha256": evidence.get("rom_sha256"),
        "layout_sha256": evidence.get("layout_sha256"),
        "fused": True,
        "tier": evidence.get("tier"),
        "work_dir": evidence.get("work_dir"),
        "text": evidence.get("text"),
        "git_commit": evidence.get("git_commit"),
    }
    tmp = SIGNED_OFF.with_name(SIGNED_OFF.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(SIGNED_OFF)


def _ready() -> dict[str, Any]:
    rom_ok = ROM.is_file()
    manifest_ok = MANIFEST.is_file()
    tokenizer_ok = TOKENIZER.is_file()
    return {
        "rom": rom_ok,
        "manifest": manifest_ok,
        "tokenizer": tokenizer_ok,
        "ready": rom_ok and manifest_ok and tokenizer_ok,
        "rom_path": str(ROM),
        "manifest_path": str(MANIFEST),
        "tokenizer_path": str(TOKENIZER),
        "rom_bytes": ROM.stat().st_size if rom_ok else 0,
        "rtl_signed_off": _signed_off() is not None,
        "error": None if (rom_ok and manifest_ok and tokenizer_ok) else (
            "Missing packed ROM or tokenizer. Stage artifacts/qwen3.bf16rom "
            "and artifacts/tokenizer.json locally."
        ),
    }


def _live_status(work: Path | None) -> dict[str, Any]:
    if work is None:
        return {"python_status": None, "rtl_status": None}
    return {
        "python_status": _read_json(work / "python_status.json"),
        "rtl_status": _read_json(work / "rtl" / "rtl_status.json"),
    }


def _compact_timeline(work: Path | None) -> list[dict[str, Any]]:
    if work is None:
        return []
    path = work / "python.ndjson"
    if not path.is_file():
        path = work / "python.filtered.ndjson"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if not event:
            continue
        events.append({
            "event": event,
            "layer": record.get("layer", -1),
            "token": record.get("token", record.get("position", -1)),
            "stage": record.get("stage", str(event).rsplit(".", 1)[-1]),
            "shape": record.get("shape") or [],
            "elements": record.get("elements"),
            "values": list(record.get("values") or [])[:16],
        })
    return events


def _mismatch_events(evidence: dict[str, Any] | None) -> list[str]:
    if not evidence:
        return []
    names: list[str] = []
    mismatches = (evidence.get("comparison") or {}).get("mismatches") or []
    for item in mismatches:
        key = item.get("key")
        if isinstance(key, (list, tuple)) and key:
            names.append(str(key[0]))
        elif isinstance(key, str):
            names.append(key)
    return names


def _work_payload(work: Path | None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _live_status(work)
    payload["timeline"] = _compact_timeline(work)
    payload["mismatch_events"] = _mismatch_events(evidence)
    if work is not None:
        payload["work_dir"] = str(work)
    return _with_decode(payload, evidence)


def _oracle_sidecar(work: Path | None) -> dict[str, Any]:
    if work is None:
        return {}
    return _read_json(work / "oracle_result.json") or {}


def _with_decode(payload: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    ev = evidence if isinstance(evidence, dict) else {}
    ids = payload.get("token_ids") or ev.get("token_ids") or []
    payload["token_texts"] = [_decode_ids([int(i)]) for i in ids]
    side = _oracle_sidecar(Path(payload["work_dir"]) if payload.get("work_dir") else None)
    py_arg = ev.get("python_argmax") if ev.get("python_argmax") is not None else side.get("python_argmax")
    rtl_arg = ev.get("rtl_argmax") if ev.get("rtl_argmax") is not None else (ev.get("rtl_meta") or {}).get("argmax")
    if py_arg is not None:
        payload["python_argmax"] = py_arg
        payload["python_argmax_text"] = _decode_ids([int(py_arg)])
    if rtl_arg is not None:
        payload["rtl_argmax"] = rtl_arg
        payload["rtl_argmax_text"] = _decode_ids([int(rtl_arg)])
    prompt = str(payload.get("text") or ev.get("text") or "")
    nxt = payload.get("python_argmax_text") or payload.get("rtl_argmax_text") or ""
    payload["prompt_text"] = prompt
    payload["predicted_text"] = prompt + nxt if nxt else prompt
    return payload


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    view = dict(job)
    work = Path(job["work_dir"]) if job.get("work_dir") else None
    evidence = view.get("evidence")
    view.update(_work_payload(work, evidence if isinstance(evidence, dict) else None))
    if view.get("status") == "running":
        rtl = view.get("rtl_status") or {}
        python = view.get("python_status") or {}
        if rtl.get("busy"):
            view["phase"] = "rtl"
            view["message"] = (
                f"RTL {rtl.get('stage', '?')} layer {rtl.get('layer', '?')} "
                f"cycle {rtl.get('cycle', '?')}"
            )
        elif python.get("phase") == "python" and python.get("busy"):
            view["phase"] = "python"
            view["message"] = python.get("message", "Python oracle running")
        else:
            view["phase"] = python.get("phase", "starting")
    view["rtl_signed_off"] = _signed_off() is not None
    return _with_decode(view, evidence if isinstance(evidence, dict) else None)


def _verify_matches(candidate: dict[str, Any] | None, text: str,
                    token_ids: list[int], tier: str) -> bool:
    if not candidate:
        return False
    if candidate.get("tier") != tier:
        return False
    if candidate.get("passed") is not True:
        return False
    ids_match = candidate.get("token_ids") == token_ids
    text_match = bool(text) and candidate.get("text") == text
    return bool(ids_match or text_match)


def _cached_evidence(text: str, token_ids: list[int], tier: str) -> dict[str, Any] | None:
    """Load a finished, PASSING T4 for this token sequence.

    A failed comparison is never replayed as "cached": the UI would show a
    stale mismatch forever and never re-run. Failed runs are re-verified.
    Scan every run dir — cache_key used to omit `fused`, so the hash can drift.
    """
    work = RUNS / cache_key(token_ids, tier, fused=True)
    docs: list[tuple[float, dict[str, Any]]] = []
    paths = [work / "verify.json", RUNS / cache_key(token_ids, tier, fused=False) / "verify.json", VERIFY_JSON]
    if RUNS.is_dir():
        paths.extend(d / "verify.json" for d in RUNS.iterdir() if d.is_dir())
    seen: set[str] = set()
    for path in paths:
        candidate = _read_json(path)
        if not _verify_matches(candidate, text, token_ids, tier):
            continue
        # Prefer the directory that actually holds this verify.json. Deployed
        # copies still have a laptop work_dir baked into the file.
        stored = path.parent if path.parent != EVIDENCE else (
            Path(candidate["work_dir"]) if candidate.get("work_dir") else work
        )
        marker = str(stored)
        if marker in seen:
            continue
        if not (stored.is_dir() and (stored / "python.ndjson").is_file()):
            continue
        seen.add(marker)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        patched = dict(candidate)
        patched["work_dir"] = str(stored)
        docs.append((mtime, patched))
    if not docs:
        return None
    docs.sort(key=lambda item: item[0], reverse=True)
    return docs[0][1]


def _work_is_busy(work: Path) -> bool:
    return work_is_busy(work)


def _sync_job_from_disk(job: dict[str, Any]) -> None:
    """Refresh an attached or in-flight job from work-dir status files."""
    work = Path(job["work_dir"]) if job.get("work_dir") else None
    if work is None or not work.is_dir():
        return
    ev = _read_json(work / "verify.json")
    py = _read_json(work / "python_status.json") or {}
    rtl = _read_json(work / "rtl" / "rtl_status.json") or {}
    if ev and ev.get("passed") is True:
        job["status"] = "completed"
        job["evidence"] = ev
        job["phase"] = "done"
        job["message"] = "Python vs RTL compared."
        return
    if ev and ev.get("passed") is False:
        job["status"] = "failed"
        job["evidence"] = ev
        job["phase"] = "done"
        return
    infer = _read_json(work / "infer.json")
    if infer and infer.get("mode") == "rtl_infer":
        job["status"] = "completed"
        job["evidence"] = infer
        job["phase"] = "done"
        job["mode"] = "rtl_infer"
        job["message"] = "RTL forward (ROM signed off; not re-compared)."
        return
    if not _work_is_busy(work):
        if job.get("status") == "running":
            job["status"] = "failed"
            job["phase"] = "error"
            job["message"] = "Job process is gone; work dir is idle."
        return
    if rtl.get("busy"):
        job["status"] = "running"
        job["phase"] = "rtl"
        return
    if py.get("busy"):
        job["status"] = "running"
        job["phase"] = "python"
        job["message"] = py.get("message", "Python oracle running")
        return


def _find_run_dir(token_ids: list[int], tier: str) -> Path | None:
    fused = RUNS / cache_key(token_ids, tier, fused=True)
    if fused.is_dir() and (
        _work_is_busy(fused)
        or (fused / "verify.json").is_file()
        or (fused / "python.ndjson").is_file()
        or (fused / "infer.json").is_file()
    ):
        return fused
    unfused = RUNS / cache_key(token_ids, tier, fused=False)
    if unfused.is_dir() and (
        _work_is_busy(unfused)
        or (unfused / "verify.json").is_file()
        or (unfused / "python.ndjson").is_file()
    ):
        return unfused
    if not RUNS.is_dir():
        return None
    newest: tuple[float, Path] | None = None
    for d in RUNS.iterdir():
        if not d.is_dir():
            continue
        doc = _read_json(d / "verify.json")
        if doc and doc.get("token_ids") == token_ids and doc.get("tier") == tier:
            try:
                mtime = (d / "verify.json").stat().st_mtime
            except OSError:
                mtime = 0.0
            if newest is None or mtime > newest[0]:
                newest = (mtime, d)
    return newest[1] if newest else None


def _replay_payload(text: str, token_ids: list[int], tier: str) -> dict[str, Any]:
    """Look up a prompt's T4 or signed-off RTL infer without starting a job."""
    signed = _signed_off() is not None
    if signed:
        cached_i = _cached_infer(token_ids)
        if cached_i:
            work = Path(cached_i["work_dir"])
            return {
                "status": "cached",
                "mode": "rtl_infer",
                "job_id": None,
                "tier": "RTL",
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "work_dir": str(work),
                "message": "ROM signed off. Replaying this prompt’s RTL forward.",
                "evidence": cached_i,
                "rtl_signed_off": True,
                **_work_payload(work, cached_i),
            }
        infer_work = RUNS / cache_key(token_ids, "RTL", fused=True)
        if _work_is_busy(infer_work):
            return {
                "status": "running",
                "mode": "rtl_infer",
                "job_id": None,
                "tier": "RTL",
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "work_dir": str(infer_work),
                "phase": "rtl",
                "message": "ROM signed off. RTL forward in flight.",
                "rtl_signed_off": True,
                **_work_payload(infer_work, None),
            }

    cached = _cached_evidence(text, token_ids, tier)
    if cached:
        work = Path(cached["work_dir"])
        return {
            "status": "cached",
            "job_id": None,
            "tier": tier,
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "work_dir": str(work),
            "message": "Loaded completed verify for this token sequence.",
            "evidence": cached,
            "rtl_signed_off": signed,
            **_work_payload(work, cached),
        }
    work = _find_run_dir(token_ids, tier)
    if work is None:
        return {"status": "miss", "text": text, "token_ids": token_ids, "tier": tier,
                "rtl_signed_off": signed}
    py = _read_json(work / "python_status.json") or {}
    rtl = _read_json(work / "rtl" / "rtl_status.json") or {}
    evidence = _read_json(work / "verify.json")
    mode = "t4"
    if _work_is_busy(work):
        status = "running"
        phase = "rtl" if rtl.get("busy") else "python"
        message = py.get("message") or "In-flight T4 — replaying checkpoints already written."
    elif not (work / "python.ndjson").is_file() and not evidence:
        return {"status": "miss", "text": text, "token_ids": token_ids, "tier": tier,
                "rtl_signed_off": signed}
    elif evidence:
        status = "completed" if evidence.get("passed") else "failed"
        phase = "done"
        message = "Loaded verify for this token sequence."
    else:
        # Python trace is enough to play the walk. Do not launch Verilator
        # again — POST /api/verify used to start a full T4 here.
        status = "cached"
        phase = "done"
        mode = "oracle_replay"
        message = (
            "Python forward is cached. Playing it back — "
            "Python and Verilator are not running again."
        )
    payload = {
        "status": status,
        "job_id": None,
        "tier": tier,
        "text": text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "work_dir": str(work),
        "phase": phase,
        "mode": mode,
        "message": message,
        "evidence": evidence,
        "rtl_signed_off": signed,
        **_work_payload(work, evidence),
    }
    payload["token_ids"] = token_ids
    payload["text"] = text
    payload["status"] = status
    payload["phase"] = phase
    payload["mode"] = mode
    return _with_decode(payload, evidence)


app = FastAPI(title="Qwen3 chip tile", version="1")

_PUBLIC_PATHS = frozenset({"/login", "/api/login", "/styles.css"})
_NO_STORE = {"Cache-Control": "no-store"}


@app.middleware("http")
async def pin_gate(request: Request, call_next):
    if not lab_auth.pin_configured() or lab_auth.request_authed(request):
        return await call_next(request)
    path = request.url.path
    if path in _PUBLIC_PATHS:
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "login required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html", headers=_NO_STORE)


@app.get("/login")
def login_page(request: Request):
    if not lab_auth.pin_configured() or lab_auth.request_authed(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(WEB / "login.html", headers=_NO_STORE)


@app.post("/api/login")
def api_login(body: LoginRequest, request: Request) -> JSONResponse:
    if not lab_auth.pin_configured():
        raise HTTPException(400, "PIN gate is not enabled")
    if not lab_auth.pin_matches(body.pin):
        raise HTTPException(401, "Wrong PIN.")
    response = JSONResponse({"ok": True})
    lab_auth.attach_session(response, request)
    return response


@app.post("/api/logout")
def api_logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    lab_auth.clear_session(response)
    return response


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(WEB / "app.js", headers=_NO_STORE)


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(WEB / "styles.css", headers=_NO_STORE)


@app.get("/gpu-experience.js")
def gpu_experience_js() -> FileResponse:
    return FileResponse(WEB / "gpu-experience.js", headers=_NO_STORE)


@app.get("/gpu-experience.css")
def gpu_experience_css() -> FileResponse:
    return FileResponse(WEB / "gpu-experience.css", headers=_NO_STORE)


@app.get("/hw-packages.js")
def hw_packages_js() -> FileResponse:
    return FileResponse(WEB / "hw-packages.js", headers=_NO_STORE)


@app.get("/hw-packages.css")
def hw_packages_css() -> FileResponse:
    return FileResponse(WEB / "hw-packages.css", headers=_NO_STORE)


@app.get("/api/ready")
def api_ready() -> dict[str, Any]:
    return _ready()


@app.post("/api/tokenize")
def api_tokenize(body: TokenizeRequest) -> dict[str, Any]:
    if not TOKENIZER.is_file():
        raise HTTPException(400, f"tokenizer.json not found: {TOKENIZER}")
    try:
        token_ids = _tokenize(body.text, TOKENIZER)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"text": body.text, "token_ids": token_ids, "token_count": len(token_ids),
            "token_texts": [_decode_ids([int(i)]) for i in token_ids]}


@app.get("/api/replay")
def api_replay(text: str = "", tier: str = "T4") -> dict[str, Any]:
    """Load cached or in-flight T4 for this prompt. Never starts a new job."""
    tier = tier.upper()
    if tier not in {"T2", "T3", "T4", "T5"}:
        raise HTTPException(400, "tier must be T2, T3, T4, or T5")
    text = (text or "").strip()
    if not text:
        return {"status": "miss"}
    if not TOKENIZER.is_file():
        raise HTTPException(400, f"tokenizer.json not found: {TOKENIZER}")
    try:
        token_ids = _tokenize(text, TOKENIZER)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _replay_payload(text, token_ids, tier)


def _latest_evidence() -> dict[str, Any] | None:
    """Newest completed verify: the top-level file if current, else the best run.

    The top-level evidence/verify.json can lag behind (CLI runs with --output
    elsewhere, interrupted runs). Prefer the run whose work-dir verify.json is
    newer, so the UI never shows a stale verdict for a finished run.
    """
    top = _read_json(VERIFY_JSON)
    candidates: list[tuple[float, dict[str, Any]]] = []
    if top and top.get("work_dir"):
        # Stamp the top-level doc with its OWN mtime, not its work-dir's:
        # work_dir/verify.json is a different, newer document.
        try:
            candidates.append((VERIFY_JSON.stat().st_mtime, top))
        except OSError:
            pass
    if RUNS.is_dir():
        for run_dir in RUNS.iterdir():
            doc = _read_json(run_dir / "verify.json")
            if doc and doc.get("work_dir"):
                try:
                    mtime = (run_dir / "verify.json").stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, doc))
    if not candidates:
        return top
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


@app.get("/api/evidence")
def api_evidence() -> dict[str, Any]:
    evidence = _latest_evidence()
    work = Path(evidence["work_dir"]) if evidence and evidence.get("work_dir") else None
    payload = {"evidence": evidence, **_work_payload(work, evidence)}
    if evidence:
        payload["token_ids"] = evidence.get("token_ids")
        payload["text"] = evidence.get("text")
        payload["tier"] = evidence.get("tier")
        payload["status"] = "completed" if evidence.get("passed") else "failed"
        _with_decode(payload, evidence)
    return payload


@app.get("/api/compare")
def api_compare(seq: int = 1) -> dict[str, Any]:
    """Analytical H100/B200/ASIC comparison for one forward pass."""
    from lab.compute_metrics import compare as platform_compare

    seq = max(1, min(int(seq), 40960))
    model = json.loads((ROOT / "configs" / "qwen3_0_6b.json").read_text(encoding="utf-8"))
    return platform_compare(model, seq)


@app.get("/api/deck/{name}")
def api_deck_evidence(name: str) -> dict[str, Any]:
    """Golden evidence JSONs the deck renders from (exec_models, array_scaling,
    precision_scaling, multi_die_scaling, gpu_tile_sim). Served no-store so
    slides always match the regenerated files."""
    allowed = {
        "exec_models": "exec_models.json",
        "array_scaling": "array_scaling.json",
        "precision_scaling": "precision_scaling.json",
        "multi_die_scaling": "multi_die_scaling.json",
        "gpu_tile_sim": "gpu_tile_sim.json",
        "machine_timelines": "machine_timelines.json",
    }
    if name not in allowed:
        raise HTTPException(404, "unknown deck evidence file")
    path = EVIDENCE / allowed[name]
    if not path.is_file():
        raise HTTPException(404, f"evidence file missing: {allowed[name]}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/stage_breakdown")
def api_stage_breakdown() -> dict[str, Any]:
    """Measured per-stage cycles/MACs/ROM reads for both schedules, with
    N4-class time and energy derived per stage (energy labeled estimated)."""

    def load(name: str) -> dict[str, Any]:
        path = ROOT / "evidence" / "calibration" / f"{name}_stage_breakdown.json"
        return json.loads(path.read_text(encoding="utf-8"))

    clock = 500e6  # measured chip clock
    mac_pj = 1.2   # 28nm-class measured-chip constants (lab.compute_metrics)
    rom_pj = 0.8
    out: dict[str, Any] = {"schema_version": 1, "clock_hz": clock, "schedules": {}}
    for name in ("fused", "unfused"):
        doc = load(name)
        rows = []
        for stage in doc["stages"]:
            cycles = stage["cycles"]
            macs = stage.get("macs", 0)
            rom = stage.get("rom_reads", 0) * 256  # bits
            rows.append({
                "stage": stage["stage"],
                "entries": stage["entries"],
                "cycles": cycles,
                "macs": macs,
                "time_us": cycles / clock * 1e6,
                "mac_energy_j": macs * mac_pj / 1e12,
                "rom_energy_j": rom * rom_pj / 1e12,
            })
        total = sum(r["cycles"] for r in rows)
        out["schedules"][name] = {
            "total_cycles": doc["total_cycles"],
            "total_time_us": doc["total_cycles"] / clock * 1e6,
            "stages": sorted(rows, key=lambda r: -r["cycles"]),
        }
    return out


@app.get("/api/walk")
def api_walk(schedule: str = "fused") -> dict[str, Any]:
    """Measured per-layer walk + per-stage profile for one token pass.

    Serves the precomputed walk JSON derived from the instrumented RTL
    commit logs (stage-entry cycle stamps). Layers 0-27 with per-stage
    cycle spans, plus global per-stage totals — the raw material for the
    slide-1 walkthrough and profiling panels.

    "streaming" is a different, much smaller real RTL run: gpu_model/rtl/
    gpu_stream_core.sv, an own-authored streaming-SIMD reference design
    (see docs there), Verilator/iverilog-measured, not the ASIC. It has no
    "layers" key (one-shot walk, not 28 time-multiplexed layers).
    """
    if schedule not in ("fused", "unfused", "streaming"):
        raise HTTPException(400, "schedule must be fused, unfused, or streaming")
    path = ROOT / "evidence" / "calibration" / f"{schedule}_walk.json"
    if not path.is_file():
        raise HTTPException(404, f"walk data missing for {schedule}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/arch")
def api_arch() -> dict[str, Any]:
    """Precomputed architecture comparison: HBM vs chiplet vs CIM vs wafer.

    Serves evidence/arch_comparison.json (12 scenarios: 4 architectures x
    3 weight precisions, all calibrated on the measured 0.6B constants).
    Backs the slide-2 architecture section and diagram viewer.
    """
    path = ROOT / "evidence" / "arch_comparison.json"
    if not path.is_file():
        raise HTTPException(404, "arch comparison data missing")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/arch/diagram/{name}")
def api_arch_diagram(name: str) -> Any:
    """Serve one of the five architecture SVG diagrams."""
    if name not in {"hbm_streaming", "rom_chiplet", "analog_cim",
                    "wafer_scale", "comparison"}:
        raise HTTPException(404, "unknown diagram")
    path = ROOT / "docs" / "diagrams" / f"{name}.svg"
    if not path.is_file():
        raise HTTPException(404, "diagram missing")
    return Response(content=path.read_text(encoding="utf-8"),
                    media_type="image/svg+xml")


@app.get("/api/scale1t")
def api_scale_1t(
    preset: str = "moe_1t",
    lanes: int = 4096,
    ctx: int = 1,
    schedule: str = "fused",
    weight_bits: int = 16,
    tier: str = "hbm",
    dies: int | None = None,
    kv_bits: int = 16,
) -> dict[str, Any]:
    """1T-scaling scenario: memory hierarchy, calibrated cycles, rivals."""
    from lab.scale_1t import MODEL_PRESETS, scenario

    if preset not in MODEL_PRESETS:
        raise HTTPException(404, f"unknown preset {preset}")
    lanes = max(16, min(int(lanes), 65536))
    ctx = max(1, min(int(ctx), 131072))
    if schedule not in ("fused", "unfused"):
        raise HTTPException(400, "schedule must be fused or unfused")
    if weight_bits not in (4, 8, 16):
        raise HTTPException(400, "weight_bits must be 4, 8, or 16")
    if kv_bits not in (8, 16):
        raise HTTPException(400, "kv_bits must be 8 or 16")
    if tier not in ("sram", "lpddr", "hbm", "hbm_wide", "ddr"):
        raise HTTPException(400, "unknown tier")
    if dies is not None:
        dies = max(1, min(int(dies), 1024))
    return scenario(preset, lanes, ctx, schedule, weight_bits, tier, dies,
                    kv_bits)


@app.get("/api/hw_comparison")
def api_hw_comparison() -> dict[str, Any]:
    """Measured per-hardware comparison: mask-ROM vs GPU vs TPU vs LPU.

    Serves evidence/hw_comparison.json (built by lab/hw_comparison.py from
    the hw_verify run summaries + stage-entry traces). Same compute (MACs),
    different memory behavior (HBM traffic, cycles) per backend.
    """
    path = ROOT / "evidence" / "hw_comparison.json"
    if not path.is_file():
        raise HTTPException(404, "hw comparison data missing; run lab.hw_comparison")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/verify/{job_id}")
def api_verify_status(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown verify job")
        _sync_job_from_disk(job)
        snapshot = dict(job)
    return _job_view(snapshot)


def _event_shape(work: Path, event: str) -> list[int]:
    path = work / "python.ndjson"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == event:
            return list(record.get("shape") or [])
    return []


def _resolve_work_dir(job_id: str | None, work_dir: str | None) -> Path | None:
    """Prefer the explicit job/work dir; fall back to the latest evidence."""
    if work_dir:
        candidate = Path(work_dir)
        if candidate.is_dir():
            return candidate
    if job_id:
        with _lock:
            job = _jobs.get(job_id)
        if job and job.get("work_dir"):
            candidate = Path(job["work_dir"])
            if candidate.is_dir():
                return candidate
    evidence = _read_json(VERIFY_JSON)
    if evidence and evidence.get("work_dir"):
        candidate = Path(evidence["work_dir"])
        if candidate.is_dir():
            return candidate
    return None


@app.get("/api/tensor")
def api_tensor(
    event: str,
    source: str = "python",
    offset: int = 0,
    limit: int = 4096,
    job_id: str | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    if source not in {"python", "rtl"}:
        raise HTTPException(400, "source must be python or rtl")
    work = _resolve_work_dir(job_id, work_dir)
    if work is None:
        raise HTTPException(404, "no run work dir found for this request")
    safe = event.replace("/", "_").replace(".", "_") + ".f32le"
    path = work / f"{source}_checkpoints" / safe
    if not path.is_file():
        raise HTTPException(404, f"no {source} checkpoint for {event}")
    blob = path.read_bytes()
    total = len(blob) // 4
    offset = max(0, int(offset))
    limit = min(max(1, int(limit)), 8192)
    end = min(total, offset + limit)
    count = end - offset
    values = list(struct.unpack("<" + "f" * count, blob[offset * 4:end * 4])) if count else []
    return {
        "event": event,
        "source": source,
        "shape": _event_shape(work, event),
        "elements": total,
        "offset": offset,
        "values": values,
    }


def _run_verify(job_id: str, text: str, token_ids: list[int], tier: str, work: Path) -> None:
    output = EVIDENCE / "verify.json"
    command = [
        sys.executable,
        "-m",
        "lab.verify",
        "--text",
        text,
        "--token-ids",
        ",".join(str(token) for token in token_ids),
        "--tier",
        tier,
        "--work",
        str(work),
        "--output",
        str(output),
        "--rom",
        str(ROM),
        "--manifest",
        str(MANIFEST),
        "--tokenizer",
        str(TOKENIZER),
        "--fused",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        evidence = _read_json(output) or _read_json(work / "verify.json")
        if evidence and evidence.get("passed"):
            _record_signoff(evidence)
        with _lock:
            _jobs[job_id].update({
                "status": "completed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-8000:],
                "stderr": (proc.stderr or "")[-8000:],
                "evidence": evidence,
                "phase": "done",
                "message": (
                    "Python vs RTL compared."
                    if proc.returncode == 0
                    else (
                        "T4 compared; mismatches recorded in evidence."
                        if evidence
                        else (proc.stderr or proc.stdout or "verify failed")[-500:]
                    )
                ),
            })
    except Exception as exc:  # noqa: BLE001 — surface job failure to the UI
        with _lock:
            _jobs[job_id].update({
                "status": "failed",
                "error": str(exc),
                "phase": "error",
                "message": str(exc),
            })


def _run_infer(job_id: str, text: str, token_ids: list[int], work: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "lab.verify",
        "--infer",
        "--fused",
        "--text",
        text,
        "--token-ids",
        ",".join(str(token) for token in token_ids),
        "--work",
        str(work),
        "--rom",
        str(ROM),
        "--manifest",
        str(MANIFEST),
        "--tokenizer",
        str(TOKENIZER),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        evidence = _read_json(work / "infer.json")
        with _lock:
            _jobs[job_id].update({
                "status": "completed" if proc.returncode == 0 and evidence else "failed",
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-8000:],
                "stderr": (proc.stderr or "")[-8000:],
                "evidence": evidence,
                "phase": "done",
                "mode": "rtl_infer",
                "message": (
                    "RTL forward (ROM signed off; Python not re-run)."
                    if proc.returncode == 0
                    else (proc.stderr or proc.stdout or "RTL infer failed")[-500:]
                ),
            })
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _jobs[job_id].update({
                "status": "failed",
                "error": str(exc),
                "phase": "error",
                "message": str(exc),
            })


def _cached_infer(token_ids: list[int]) -> dict[str, Any] | None:
    work = RUNS / cache_key(token_ids, "RTL", fused=True)
    doc = _read_json(work / "infer.json")
    if not doc or doc.get("mode") != "rtl_infer":
        return None
    if doc.get("rom_sha256") != _rom_sha256():
        return None
    # An RTL-only infer whose argmax was never compared against the canonical
    # oracle is NOT a verified answer. The UI must not present it as the
    # model's prediction (a multi-token RTL bug can make it silently wrong).
    if doc.get("python_argmax") is None and not doc.get("compared"):
        return None
    doc["work_dir"] = str(work)
    return doc


@app.post("/api/verify")
def api_verify(body: VerifyRequest) -> dict[str, Any]:
    text = body.text
    tier = body.tier.upper()
    if tier not in {"T2", "T3", "T4", "T5"}:
        raise HTTPException(400, "tier must be T2, T3, T4, or T5")
    ready = _ready()
    if not ready["ready"]:
        raise HTTPException(400, ready["error"])
    try:
        token_ids = _tokenize(text, TOKENIZER)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    signed = _signed_off()
    if signed and not body.force_verify:
        cached_i = _cached_infer(token_ids)
        if cached_i:
            work = Path(cached_i["work_dir"])
            return {
                "status": "cached",
                "mode": "rtl_infer",
                "job_id": None,
                "tier": "RTL",
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "work_dir": str(work),
                "message": "ROM signed off. Replaying this prompt’s RTL forward.",
                "evidence": cached_i,
                "rtl_signed_off": True,
                **_work_payload(work, cached_i),
            }
        replay = _replay_payload(text, token_ids, tier)
        if replay.get("status") != "miss":
            return replay
        job_id = hashlib.sha256(f"RTL:fused:{token_ids}".encode()).hexdigest()[:12]
        work = RUNS / cache_key(token_ids, "RTL", fused=True)
        return _start_job(
            job_id, text, token_ids, "RTL", work,
            phase="rtl",
            message="ROM signed off. Host sends token IDs; RTL only — no Python compare.",
            mode="rtl_infer",
            runner=_run_infer,
        )

    cached = _cached_evidence(text, token_ids, tier)
    if cached:
        work = Path(cached["work_dir"])
        return {
            "status": "cached",
            "job_id": None,
            "tier": tier,
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "work_dir": str(work),
            "message": "Loaded completed verify for this token sequence.",
            "evidence": cached,
            "rtl_signed_off": _signed_off() is not None,
            **_work_payload(work, cached),
        }

    # A saved Python trace is a replay, not a reason to launch Verilator.
    # force_verify still runs a full fused T4 (oracle cache still skips Python).
    if not body.force_verify:
        replay = _replay_payload(text, token_ids, tier)
        if replay.get("status") != "miss":
            return replay

    job_id = hashlib.sha256(f"{tier}:fused:{token_ids}".encode()).hexdigest()[:12]
    work = RUNS / cache_key(token_ids, tier, fused=True)
    return _start_job(
        job_id, text, token_ids, tier, work,
        phase="python",
        message="Tokenized. Python oracle then fused Verilator T4.",
        mode="t4",
        runner=_run_verify,
    )


def _start_job(
    job_id: str,
    text: str,
    token_ids: list[int],
    tier: str,
    work: Path,
    *,
    phase: str,
    message: str,
    mode: str,
    runner,
) -> dict[str, Any]:
    with _lock:
        existing = _jobs.get(job_id)
        if existing and existing.get("status") == "running":
            _sync_job_from_disk(existing)
            if existing.get("status") == "running":
                return _job_view(existing)
        if _work_is_busy(work):
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "phase": phase,
                "mode": mode,
                "tier": tier,
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "work_dir": str(work),
                "message": "Attached to in-flight run for this prompt — not starting a second one.",
                "attached": True,
            }
            _sync_job_from_disk(_jobs[job_id])
            snapshot = dict(_jobs[job_id])
            return _job_view(snapshot)
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "phase": phase,
            "mode": mode,
            "tier": tier,
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "work_dir": str(work),
            "message": message,
        }
        snapshot = dict(_jobs[job_id])
    if runner is _run_infer:
        thread = threading.Thread(
            target=runner,
            args=(job_id, text, token_ids, work),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=runner,
            args=(job_id, text, token_ids, tier, work),
            daemon=True,
        )
    thread.start()
    return _job_view(snapshot)


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI/uvicorn missing. Install with: "
            "python3 -m pip install -r services/chip_lab/requirements.txt"
        ) from exc

    uvicorn.run(
        "services.chip_lab.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
