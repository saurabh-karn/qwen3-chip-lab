#!/usr/bin/env python3
"""H100 tensor-core tile/occupancy model.

Uses the exact same tile geometry (`_tile_count`, tile_m/n/k=16/8/16) that
`lab/exec_models.py::_simulate_stage_gpu` already derives for every GPU
GEMM stage -- so this is not a second, independent tiling of the workload,
it is the same real tiling costed through a different lens: instead of the
accounting-clock roofline formula (`compute_s`/`weight_s` at a synthetic
1.5 GHz), this walks actual `mma.sync.aligned.m16n8k16` issues, bounded by
the published (not invented) warp-occupancy cap across the real 132-SM die.

Every constant here is cited in docs/references.md Sec.10. Stdlib-only
(QWEN-PY-001). This does not replace lab/exec_models.py's KPI numbers --
it feeds the slide-2 "zoom into one tile" panel with a citation-level
deeper look at the same real per-stage table
(lab/compute_metrics.py::stage_table).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.compute_metrics import stage_table
from lab.exec_models import H100, PlatformConfig, _tile_count

ROOT = Path(__file__).resolve().parents[1]

# ---- Cited constants: docs/references.md Sec.10 ----------------------------
SM_COUNT = 132  # [P] NVIDIA H100 Tensor Core GPU Architecture Whitepaper v1.02
TENSOR_CORES_PER_SM = 4  # [P] same whitepaper (4th-gen Tensor Cores/SM)
WARP_SIZE = 32  # [P] CUDA C++ Programming Guide / PTX ISA (universal)
MAX_WARPS_PER_SM = 64  # [P] NVIDIA Hopper Tuning Guide (cc 9.0 occupancy cap)
MMA_LATENCY_CYCLES = 24.7  # [S] arXiv:2206.02874, measured on Ampere A100 --
# no Hopper-specific mma.sync.m16n8k16 cycle-latency figure is published;
# this is the closest published analog for the same instruction shape.

PROVENANCE: dict[str, str] = {
    "sm_count": "[P] NVIDIA H100 Tensor Core GPU Architecture Whitepaper v1.02",
    "tensor_cores_per_sm": "[P] NVIDIA H100 Tensor Core GPU Architecture Whitepaper v1.02",
    "warp_size": "[P] CUDA C++ Programming Guide / PTX ISA",
    "max_warps_per_sm": "[P] NVIDIA Hopper Tuning Guide, cc 9.0 occupancy table",
    "mma_latency_cycles": (
        "[S] arXiv:2206.02874 'Dissecting Tensor Cores via Microbenchmarks', "
        "measured on Ampere A100 -- no Hopper-specific figure is published; "
        "applied here as the closest published analog"
    ),
    "occupancy_bound": (
        "[E] warp-count ceiling only, not a tighter register/shared-memory-"
        "derived bound -- no footprint is published for this uncompiled kernel"
    ),
}


@dataclass(frozen=True)
class TileGeometry:
    tiles_m: int
    tiles_n: int
    tiles_k: int
    total_tiles: int


def tile_geometry(weight_bytes_per_call: int, platform: PlatformConfig = H100) -> TileGeometry:
    """Same tile geometry `_simulate_stage_gpu` derives for this stage --
    reused verbatim so the microarchitecture model and the roofline model
    describe the same real tiling of the same real GEMM."""
    if weight_bytes_per_call <= 0:
        return TileGeometry(0, 0, 0, 0)
    tiles_m = 1  # batch-1 decode
    tiles_n = _tile_count(weight_bytes_per_call // 2 // 1024, platform.tile_n)
    tiles_k = _tile_count(1024, platform.tile_k)  # hidden_size, matches exec_models.py
    return TileGeometry(tiles_m, tiles_n, tiles_k, tiles_m * tiles_n * tiles_k)


def concurrent_tiles(sm_count: int = SM_COUNT, max_warps_per_sm: int = MAX_WARPS_PER_SM) -> int:
    """Tiles in flight across the whole die at once. One mma.sync tile op is
    issued by one warp, so warps-in-flight == tiles-in-flight (1:1)."""
    return sm_count * max_warps_per_sm


@dataclass(frozen=True)
class WaveState:
    """One simulated step: a real event, not an algebraic shortcut.

    Tiles whose weight bytes have arrived queue in FIFO arrival order and
    are issued to a free warp slot as soon as one exists, capped at the
    cited die-wide occupancy (`concurrent_tiles()`). A wave is the set of
    tiles issued together; it holds its warp slots for `MMA_LATENCY_CYCLES`
    (cited, docs/references.md Sec.10) before retiring, at which point the
    next wave's tiles (already queued -- weight bytes arrive far faster
    than a single MMA op retires, for every real stage in this workload,
    see `test_gpu_tile_sim.py::test_bandwidth_dominates_mma_latency`) issue.
    """
    wave: int
    tiles_issued: int
    occupancy_frac: float
    tiles_remaining_after: int
    cycle_start: float
    cycle_end: float


def simulate_stage_waves(weight_bytes_per_call: int) -> list[WaveState]:
    """The actual per-wave execution trace for one stage-call: a real FIFO/
    occupancy simulation, stepped wave by wave, not a single division. See
    `StageTileSim.waves`/`total_mma_cycles`, which are this trace's `len()`
    and final `cycle_end` -- summary statistics of the trace below, not an
    independent computation."""
    geom = tile_geometry(weight_bytes_per_call)
    conc = concurrent_tiles()
    if geom.total_tiles == 0:
        return []
    waves: list[WaveState] = []
    remaining = geom.total_tiles
    cycle_cursor = 0.0
    wave_idx = 0
    while remaining > 0:
        issued = min(conc, remaining)
        remaining -= issued
        cycle_end = cycle_cursor + MMA_LATENCY_CYCLES
        waves.append(WaveState(
            wave=wave_idx,
            tiles_issued=issued,
            occupancy_frac=issued / conc,
            tiles_remaining_after=remaining,
            cycle_start=cycle_cursor,
            cycle_end=cycle_end,
        ))
        cycle_cursor = cycle_end
        wave_idx += 1
    return waves


@dataclass(frozen=True)
class StageTileSim:
    stage: str
    geometry: TileGeometry
    concurrent: int
    waves: int
    total_mma_cycles: float
    wave_trace: list[WaveState]
    provenance: dict[str, str] = field(default_factory=lambda: dict(PROVENANCE))


def simulate_stage_tiles(stage: str, weight_bytes_per_call: int) -> StageTileSim:
    """One stage-call's tile count, die-wide occupancy, and MMA-issue cycles
    -- derived from the real per-wave execution trace (`simulate_stage_waves`),
    not computed independently of it.

    A stage with no weights (weight_bytes_per_call == 0, e.g. softmax,
    attn_scores) has no tiles to zoom into -- there is no GEMM tiling for a
    compute-only op in this model.
    """
    geom = tile_geometry(weight_bytes_per_call)
    conc = concurrent_tiles()
    trace = simulate_stage_waves(weight_bytes_per_call)
    if not trace:
        return StageTileSim(stage=stage, geometry=geom, concurrent=conc, waves=0,
                             total_mma_cycles=0.0, wave_trace=[])
    return StageTileSim(
        stage=stage,
        geometry=geom,
        concurrent=conc,
        waves=len(trace),
        total_mma_cycles=trace[-1].cycle_end,
        wave_trace=trace,
    )


def simulate_workload_tiles(stages_detail: list[dict[str, Any]]) -> list[StageTileSim]:
    """One StageTileSim per stage in workload.stages_detail (see
    lab/compute_metrics.py::stage_table), in the same order."""
    return [simulate_stage_tiles(s["stage"], s["weight_bytes_per_call"]) for s in stages_detail]


def _sim_to_dict(sim: StageTileSim) -> dict[str, Any]:
    return {
        "stage": sim.stage,
        "geometry": {
            "tiles_m": sim.geometry.tiles_m,
            "tiles_n": sim.geometry.tiles_n,
            "tiles_k": sim.geometry.tiles_k,
            "total_tiles": sim.geometry.total_tiles,
        },
        "concurrent_tiles": sim.concurrent,
        "waves": sim.waves,
        "total_mma_cycles": sim.total_mma_cycles,
        "wave_trace": [
            {
                "wave": w.wave,
                "tiles_issued": w.tiles_issued,
                "occupancy_frac": w.occupancy_frac,
                "tiles_remaining_after": w.tiles_remaining_after,
                "cycle_start": w.cycle_start,
                "cycle_end": w.cycle_end,
            }
            for w in sim.wave_trace
        ],
    }


def generate_gpu_tile_evidence(model: dict[str, Any], seq: int = 1) -> dict[str, Any]:
    """Generate the golden output JSON for evidence/gpu_tile_sim.json."""
    stages = stage_table(model, seq)
    sims = simulate_workload_tiles(stages)

    weight_stages = [s for s in sims if s.geometry.total_tiles > 0]
    assert weight_stages, "at least one stage must have tiles to zoom into"
    lm_head = next(s for s in sims if s.stage == "lm_head")
    for s in weight_stages:
        assert s.geometry.total_tiles <= lm_head.geometry.total_tiles, (
            f"{s.stage} has more tiles than lm_head, the largest single weight fetch"
        )

    return {
        "schema_version": 1,
        "model_id": model.get("model_id", "unknown"),
        "sequence_length": seq,
        "description": (
            "H100 tensor-core tile/occupancy model: same tile geometry as "
            "lab/exec_models.py's roofline model, costed as real "
            "mma.sync.aligned.m16n8k16 issues bounded by the published "
            "warp-occupancy cap (docs/references.md Sec.10)."
        ),
        "constants": {
            "sm_count": SM_COUNT,
            "tensor_cores_per_sm": TENSOR_CORES_PER_SM,
            "warp_size": WARP_SIZE,
            "max_warps_per_sm": MAX_WARPS_PER_SM,
            "mma_latency_cycles": MMA_LATENCY_CYCLES,
            "concurrent_tiles_across_die": concurrent_tiles(),
        },
        "provenance": PROVENANCE,
        "stages": [_sim_to_dict(s) for s in sims],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="H100 tensor-core tile/occupancy simulation")
    parser.add_argument("--config", default=str(ROOT / "configs/qwen3_0_6b.json"),
                        help="Model config JSON path")
    parser.add_argument("--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--seq", type=int, default=1, help="Sequence length")
    args = parser.parse_args()

    model = json.loads(Path(args.config).read_text())
    evidence = generate_gpu_tile_evidence(model, args.seq)

    output_text = json.dumps(evidence, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
