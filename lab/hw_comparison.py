"""Per-hardware measured comparison: mask-ROM vs GPU vs TPU vs LPU.

Reads the hw_verify run summaries (evidence/runs/hw_*_hw_verify.json) plus
each backend's stage-entry trace, and produces one JSON document:

  evidence/hw_comparison.json

Each backend row carries:
  - passed / mismatch_count (bit-exactness verdict vs the Python oracle)
  - measured cycles, MAC count, ROM reads, SRAM reads/writes (from rtl_meta)
  - per-stage cycle breakdown (from the kind-5 stage-entry records)
  - HBM-equivalent weight traffic: ROM reads x 256b for mask-ROM/GPU/TPU;
    LPU's stage pass + spill traffic; GPU/TPU/LPU compute-phase weight
    reads that hit on-die memory instead of HBM.

The point: same compute (identical MACs, bit-exact outputs), different
memory behavior (HBM traffic, cycles, utilization) per hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evidence" / "runs"
OUT = ROOT / "evidence" / "hw_comparison.json"

# Weight bytes moved from the ROM (HBM-equivalent) per backend, derived from
# the measured counters:
#   mask_rom: every weight operand is a ROM read (16 lanes x 2B = 32B/word).
#   gpu:      same streaming count — weights stream through the FIFO.
#   tpu:      same row-major fetch count — staged per row into the tile.
#   lpu:      stage pass reads each word once; compute hits on-die SRAM.
#   (LM_HEAD on LPU streams, so its count matches the others.)
ROM_WORD_BYTES = 32  # 16 lanes x BF16


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _stage_breakdown(work: Path) -> dict[str, Any]:
    """Per-stage cycle spans via lab.rtl_trace's own breakdown."""
    from lab.rtl_trace import stage_breakdown

    log = work / "rtl" / "rtl_commits.bin"
    if not log.is_file():
        return {}
    try:
        return stage_breakdown(log)
    except (OSError, ValueError, KeyError):
        return {}


def build(run_keys: list[str] | None = None) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    if run_keys is None:
        # Newest gate summaries first (hw_<n>tok keys); smoke/scratch keys
        # (anything else) follow so ad-hoc runs never displace the gates.
        gate = sorted(
            (p for p in RUNS.glob("hw_*tok_hw_verify.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        other = sorted(
            (p for p in RUNS.glob("hw_*_hw_verify.json")
             if p not in gate),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_keys = [p.name.replace("_hw_verify.json", "")
                    for p in gate + other]
    for run_key in run_keys:
        summary = _load(RUNS / f"{run_key}_hw_verify.json")
        if not summary:
            continue
        rows = []
        for result in summary.get("backends", []):
            key = result.get("backend")
            work = Path(result["work_dir"]) if result.get("work_dir") else None
            meta = result.get("rtl_meta") or {}
            rom_reads = int(meta.get("rom_read_count") or 0)
            sram_reads = int(meta.get("sram_read_count") or 0)
            row = {
                "backend": key,
                "label": result.get("label"),
                "passed": result.get("passed"),
                "mismatch_count": result.get("mismatch_count"),
                "error": result.get("error"),
                "cycles": int(meta.get("cycle_count") or 0),
                "macs": int(meta.get("mac_count") or 0),
                "rom_reads": rom_reads,
                "sram_reads": sram_reads,
                "sram_writes": int(meta.get("sram_write_count") or 0),
                "stalls": int(meta.get("stall_count") or 0),
                "argmax": int(meta.get("argmax") or 0),
                # Memory scenario: which store the weight stream pays for.
                # "rom" = direct on-die ROM port (mask-ROM chip reality);
                # "hbm" = same datapath paying +EXT_MEM_LAT response
                # latency (HBM-machine scenario). Data identical either way.
                "memory": result.get("memory", "rom"),
                "ext_mem_lat": int(result.get("ext_mem_lat") or 0),
                # HBM-equivalent weight traffic (bytes): what this design
                # pulls from the weight store during one forward pass.
                "hbm_weight_bytes": rom_reads * ROM_WORD_BYTES,
                # On-die weight reads (bytes): compute-phase reads that hit
                # local memory rather than the weight store.
                "ondie_weight_reads": sram_reads,
                "work_dir": str(work) if work else None,
            }
            # Achieved store bandwidth vs HBM3e capacity (3.35 TB/s =
            # 6,700 B/cycle at the 500 MHz lab clock, docs/references.md
            # SS6). Reported for every backend; for the HBM scenarios this
            # is the utilization the latency model leaves on the table.
            cycles = row["cycles"]
            if cycles:
                row["store_bytes_per_cycle"] = (
                    rom_reads * ROM_WORD_BYTES / cycles
                )
                row["hbm_utilization"] = (
                    row["store_bytes_per_cycle"] / 6700.0
                )
            # Energy from measured counters through cited N4 constants
            # (lab/hw_energy.py; no new constants).
            from lab.hw_energy import energy_row

            row["energy"] = energy_row(result)
            if work:
                row["stage_breakdown"] = _stage_breakdown(work)
            rows.append(row)
        # Same-compute invariant: the GEMM-engine backends (gpu/tpu/lpu and
        # their HBM scenarios, plus the fused mask-ROM chip — same fused
        # dataflow) must report identical MAC counts — the workload is
        # fixed, only the memory path differs. mask_rom (unfused) counts
        # extra MAC-lane steps for its unfused attention/V_PROJ path
        # (padded lanes), so it is excluded from the equality set but its
        # delta is reported.
        engine_macs = {r["macs"] for r in rows
                       if r.get("passed") and r["backend"] != "mask_rom"}
        mask_rom_macs = next((r["macs"] for r in rows
                              if r["backend"] == "mask_rom"), None)
        same_compute = len(engine_macs) == 1
        engine_macs_val = next(iter(engine_macs), None)
        rows.sort(key=lambda r: r["backend"])
        summaries.append({
            "run_key": run_key,
            "token_count": summary.get("token_count"),
            "tier": summary.get("tier"),
            "oracle_argmax": summary.get("oracle_argmax"),
            "same_compute": same_compute,
            "engine_macs": engine_macs_val,
            "mask_rom_extra_macs": (mask_rom_macs - engine_macs_val
                                    if mask_rom_macs is not None and engine_macs_val
                                    else None),
            "backends": rows,
        })
    return {"schema_version": 1, "runs": summaries}


def main() -> int:
    document = build()
    OUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(document['runs'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
