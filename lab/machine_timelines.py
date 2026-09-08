"""Per-machine trace timelines for the deck's replay player.

Reads the hw_verify manifest (evidence/runs/hw_1tok_hw_verify.json), parses each
backend's instrumented RTL commit log, and emits evidence/machine_timelines.json:
one chronological segment list per machine — (stage, layer, cycle span, ROM-read
delta, MAC delta) — the raw material for trace-driven playback on slide 7.

Attribution convention: the work counted between stage-entry stamps i and i+1
happened during segment i, and the final segment runs to the meta totals, so
every machine's segments sum exactly to its rtl_meta counters. The generator
fails loudly if they do not.

Energy is never stored here — the player prices segments with the same
measured-chip constants the stage-breakdown API uses (1.2 pJ/MAC,
0.8 pJ/bit ROM, 256-bit ROM words; lab/compute_metrics.py 28 nm class).
"""

from __future__ import annotations

from lab.energy_constants import (
    MEASURED_TECH_NODE,
    ROM_WORD_BITS,
    node as energy_node,
)

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.rtl_trace import STAGE_NAMES, read_commit_log  # noqa: E402

_MT_NODE = energy_node(MEASURED_TECH_NODE)

MANIFEST = ROOT / "evidence" / "runs" / "hw_1tok_hw_verify.json"
OUTPUT = ROOT / "evidence" / "machine_timelines.json"

# Same measured-chip constants as services/chip_lab/app.py /api/stage_breakdown,
# now read from the one canonical table instead of being restated here.
ENERGY_CONSTANTS = {
    "mac_pj": _MT_NODE["mac_pj_per_op"],
    "rom_pj_per_bit": _MT_NODE["rom_pj_per_bit"],
    "rom_read_bits": ROM_WORD_BITS,
    "source": f"lab/energy_constants.py {_MT_NODE['name']} (measured counters)",
}


def build_segments(payload: dict) -> tuple[list[dict], dict]:
    """Chronological segments from stage-entry stamps, reconciled to meta."""
    entries = payload.get("stage_entries") or []
    meta = payload.get("meta") or {}
    if not entries or not meta:
        raise ValueError("trace has no stage-entry instrumentation or meta record")
    total_cycles = int(meta["cycle_count"])
    segments: list[dict] = []
    prev_rom = 0
    prev_macs = 0
    for i, (stage, layer, cycle, rom, macs) in enumerate(entries):
        end = entries[i + 1][2] if i + 1 < len(entries) else total_cycles
        if end < cycle:
            raise ValueError(f"non-monotonic cycle stamps at entry {i}: {cycle} -> {end}")
        segments.append({
            "stage": STAGE_NAMES.get(stage, f"S{stage}"),
            "layer": int(layer),
            "c0": int(cycle),
            "c1": int(end),
            "rom": int(rom) - prev_rom,
            "macs": int(macs) - prev_macs,
        })
        prev_rom = int(rom)
        prev_macs = int(macs)
    # The stamps only count work up to the final entry; the tail runs to meta.
    tail_rom = int(meta["rom_read_count"]) - prev_rom
    tail_macs = int(meta["mac_count"]) - prev_macs
    if tail_rom or tail_macs:
        segments[-1]["rom"] += tail_rom
        segments[-1]["macs"] += tail_macs
    return segments, meta


def reconcile(segments: list[dict], meta: dict, label: str) -> None:
    sums = {
        "cycles": sum(s["c1"] - s["c0"] for s in segments),
        "rom_reads": sum(s["rom"] for s in segments),
        "macs": sum(s["macs"] for s in segments),
    }
    expected = {
        "cycles": int(meta["cycle_count"]),
        "rom_reads": int(meta["rom_read_count"]),
        "macs": int(meta["mac_count"]),
    }
    for key, want in expected.items():
        got = sums[key]
        if got != want:
            raise ValueError(
                f"{label}: segment {key} sum {got} != meta {want}"
            )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    machines: dict[str, dict] = {}
    for backend in manifest.get("backends", []):
        key = backend.get("backend")
        work_dir = backend.get("work_dir")
        if not key or not work_dir:
            continue
        trace = Path(work_dir) / "rtl" / "rtl_commits.bin"
        if not trace.is_file():
            print(f"skip {key}: no commit log at {trace}", file=sys.stderr)
            continue
        payload = read_commit_log(trace)
        try:
            segments, meta = build_segments(payload)
        except ValueError as exc:
            print(f"skip {key}: {exc}", file=sys.stderr)
            continue
        reconcile(segments, meta, key)
        machines[key] = {
            "label": backend.get("label", key),
            "memory": backend.get("memory"),
            "ext_mem_lat": backend.get("ext_mem_lat", 0),
            "passed": bool(backend.get("passed")),
            "meta": {
                "cycle_count": int(meta["cycle_count"]),
                "rom_read_count": int(meta["rom_read_count"]),
                "sram_read_count": int(meta.get("sram_read_count", 0)),
                "sram_write_count": int(meta.get("sram_write_count", 0)),
                "mac_count": int(meta["mac_count"]),
                "stall_count": int(meta.get("stall_count", 0)),
                "argmax": int(meta.get("argmax", 0)),
            },
            "segments": segments,
        }
        print(
            f"{key}: {len(segments)} segments, "
            f"{meta['cycle_count']:,} cycles, {meta['mac_count']:,} MACs"
        )
    if not machines:
        raise SystemExit("no machine timelines could be built")

    doc = {
        "schema_version": 1,
        "token_count": manifest.get("token_count"),
        "source": "evidence/runs/hw_1tok_hw_verify.json",
        "clock_hz": 500e6,
        "energy_constants": ENERGY_CONSTANTS,
        "machines": machines,
    }
    OUTPUT.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(machines)} machines)")


if __name__ == "__main__":
    main()
