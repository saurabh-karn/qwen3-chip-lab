#!/usr/bin/env python3
"""Execution-model engine: per-stage simulation for GPU, TPU, LPU, CIM, PIM, ROM.

Stdlib-only (QWEN-PY-001). One engine, six hardware configurations — not six
independent programs. GPU, TPU, LPU, CIM, and PIM differ only in memory
hierarchy and compute model; they share tiling, stage iteration, and accounting.

The ROM platform is always the MEASURED RTL result — never replaced by an
analytical model. The engine consumes measured counters from evidence/runs/
as its ROM row.

Architecture from docs/simulation_build_plan_01092026_1900.md:
- stage_table() from lab.compute_metrics (one workload definition)
- PlatformConfig dataclass per hardware
- simulate(platform, model, seq) -> per-stage records with:
    tiles, cycles, hbm_bytes, sram_bytes, utilization, energy split
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.compute_metrics import stage_table, totals, measured_fused, ASIC

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Technology node energy constants
# =============================================================================
#
# PRIMARY REFERENCE (baseline):
#   Horowitz, "Computing's energy problem (and what we can do about it)",
#   ISSCC 2014, pp. 10-14. doi:10.1109/ISSCC.2014.6757323
#   - 45nm baseline: 32-bit FP MAC ~1.5 pJ, 8KB SRAM access ~10 pJ
#   - DRAM access: 1-2 nJ, HBM ~200 pJ per 32-bit (6.25 pJ/bit)
#
# SCALING REFERENCES:
#   [ISSCC22] Tu et al., "A 28nm 29.2TFLOPS/W BF16 ... CIM Processor", ISSCC 2022
#     → 29.2 TFLOPS/W BF16 = 34.2 fJ/FLOP = ~0.07 pJ/MAC (CIM-optimized)
#   [ISSCC24] "A 3nm 23.2TOPS/W ... Neural Engine", ISSCC 2024
#     → 23.2 TOPS/W INT8 = 43 fJ/OP (includes memory overhead)
#   [ISSCC22-5nm] "A 5nm 254-TOPS/W ... CIM Macro", ISSCC 2022
#     → 254 TOPS/W INT8 = 3.9 fJ/MAC (ideal CIM, excludes I/O)
#
# METHODOLOGY:
#   - mac_pj_per_op: BF16 MAC for standard digital logic (not CIM).
#     Derived from Horowitz 45nm baseline (1.5 pJ) scaled by ~0.7x per node.
#   - rom_pj_per_bit: Scaled from SRAM with ~0.6x factor (simpler read path).
#   - sram_pj_per_bit: Horowitz 8KB SRAM = 10 pJ / 64Kbit ≈ 0.15 pJ/bit at 45nm,
#     scaled by ~0.65x per node (empirical ISSCC trend).
#   - hbm_pj_per_bit: Horowitz ~6 pJ/bit at 45nm; HBM3e improves to ~3.5-5 pJ/bit
#     (package-level, mostly I/O dominated, scales slowly).
#   - logic_density: TSMC published relative transistor densities.
#
# Note: These are representative estimates for SYSTEM-LEVEL modeling, not
# transistor-level precision. CIM papers report 10-100x better efficiency
# but those are macro-level figures excluding control/I/O overhead.
# =============================================================================

# Canonical per-node constants now live in lab/energy_constants.py so that
# every module pricing a measured counter uses the same numbers. Re-exported
# here to keep `from lab.exec_models import TECH_NODES` working.
from lab.energy_constants import (  # noqa: E402
    TECH_NODES,
    DEFAULT_TECH_NODE,
    MEASURED_TECH_NODE,
    energy_label,
)


@dataclass
class PlatformConfig:
    """Hardware configuration for a simulation platform."""
    name: str
    kind: str  # "gpu", "tpu", "lpu", "rom"

    # Compute
    flops_per_s: float = 0.0
    tile_m: int = 16  # GEMM tile M dimension
    tile_n: int = 8   # GEMM tile N dimension
    tile_k: int = 16  # GEMM tile K dimension

    # Memory hierarchy
    hbm_bytes_per_s: float = 0.0
    hbm_capacity_bytes: int = 0
    l2_capacity_bytes: int = 0
    sram_capacity_bytes: int = 0

    # Energy (pJ/bit for memory, pJ/op for compute)
    hbm_pj_per_bit: float = 5.0
    sram_pj_per_bit: float = 0.5
    compute_pj_per_flop: float = 0.0  # Derived from TDP if not set
    tdp_watts: float = 0.0

    # Physical
    transistors: float = 0.0
    area_mm2: float = 0.0

    # Provenance
    provenance_kind: str = "analytical_estimate"
    provenance_sources: str = ""


# Platform configurations from build plan
H100 = PlatformConfig(
    name="NVIDIA H100 SXM",
    kind="gpu",
    flops_per_s=989e12,  # BF16 dense
    tile_m=16, tile_n=8, tile_k=16,  # tensor-core tile m16n8k16
    hbm_bytes_per_s=3.35e12,
    hbm_capacity_bytes=80 * 1024**3,  # 80 GB
    l2_capacity_bytes=50 * 1024**2,   # 50 MB
    hbm_pj_per_bit=5.0,
    sram_pj_per_bit=0.5,
    tdp_watts=700.0,
    transistors=80e9,
    area_mm2=814.0,
    provenance_sources=(
        "datasheet: BF16 dense 989 TFLOPS, HBM3e 3.35 TB/s, 700W TDP; "
        "tile_m/n/k=16/8/16 is the mma.sync.aligned.m16n8k16 PTX ISA shape; "
        "SM count, tensor-cores/SM, warp/occupancy constants cited in "
        "docs/references.md Sec.10 (used by lab/gpu_tile_sim.py, not this module)"
    ),
)

B200 = PlatformConfig(
    name="NVIDIA B200",
    kind="gpu",
    flops_per_s=2.25e15,  # BF16 dense
    tile_m=16, tile_n=8, tile_k=16,
    hbm_bytes_per_s=8.0e12,
    hbm_capacity_bytes=192 * 1024**3,  # 192 GB
    l2_capacity_bytes=128 * 1024**2,   # est. 128 MB
    hbm_pj_per_bit=4.5,
    sram_pj_per_bit=0.5,
    tdp_watts=1000.0,
    transistors=208e9,
    area_mm2=1600.0,
    provenance_sources="datasheet: BF16 dense 2.25 PFLOPS, HBM3e 8 TB/s, 1000W TDP",
)

TPU_V5E = PlatformConfig(
    name="Google TPU v5e",
    kind="tpu",
    flops_per_s=197e12,
    tile_m=128, tile_n=128, tile_k=128,  # MXU 128x128
    hbm_bytes_per_s=1.6e12,
    hbm_capacity_bytes=16 * 1024**3,  # 16 GB
    hbm_pj_per_bit=5.0,
    sram_pj_per_bit=0.5,
    tdp_watts=170.0,
    transistors=32e9,  # est.
    area_mm2=325.0,    # est.
    provenance_sources="datasheet: 197 TFLOPS BF16, HBM 1.6 TB/s, 170W TDP",
)

GROQ_LPU = PlatformConfig(
    name="Groq LPU",
    kind="lpu",
    flops_per_s=188e12,
    tile_m=256, tile_n=256, tile_k=256,  # large deterministic tiles
    hbm_bytes_per_s=0.0,  # No HBM — all on-chip SRAM
    sram_capacity_bytes=230 * 1024**2,  # 230 MB on-chip
    hbm_pj_per_bit=0.0,
    sram_pj_per_bit=0.5,
    # CORRECTED: this was 750 W, which is Groq's SYSTEM/node figure (host,
    # networking, cooling). Every other platform here is charged its
    # ACCELERATOR TDP (H100 700 W SXM module, B200 1000 W, TPU v5e 170 W), so
    # charging Groq a system number made it the worst machine in the table on
    # an axis it actually does well on: its weights are SRAM-resident and
    # never move, so its weight-movement term is only ~4.8 mJ/token.
    # 375 W is the GroqCard 1 accelerator TDP, the like-for-like comparator.
    tdp_watts=375.0,
    transistors=26.8e9,
    area_mm2=725.0,
    provenance_sources=(
        "published: GroqChip TSP, 188 TFLOPS BF16, 230 MB SRAM; "
        "375 W GroqCard accelerator TDP (card-level, comparable to the H100 "
        "SXM module TDP used above - NOT the ~750 W system/node figure)"
    ),
)

# Analog compute-in-memory (RRAM/ReRAM crossbar, ISAAC-class).
# Weights live IN the cell array; a GEMM becomes per-array MxV passes:
#   - 4-bit nibble slicing: a BF16 weight needs 4 passes per operand bit-group
#     (ISAAC bit-splitting, shift-and-add between passes)
#   - Each pass: row drivers fire, cells conduct, ADC converts columns
#   - The MACs happen IN the cells — the digital side only shift-adds the
#     per-pass partial sums (one add per ADC conversion), it does not re-MAC.
#   - Throughput IS the array-pass model (tiles x 4 passes at the ADC clock);
#     there is no datasheet FLOPS number to cite.
# Constants cited in docs/references.md §5:
#   - ISAAC (Shafiee et al., ISCA 2016): ADC ~5 pJ/conv, 256x256 arrays,
#     4-bit nibble passes, shift-and-add
#   - Cell read ~0.01 pJ/bit, row drive ~0.1 pJ, shift-add ~0.1 pJ [E]
ANALOG_CIM = PlatformConfig(
    name="Analog CIM (crossbar)",
    kind="cim",
    flops_per_s=0.0,  # throughput comes from the array-pass model, not a datasheet
    tile_m=256, tile_n=256, tile_k=256,  # crossbar array size
    hbm_bytes_per_s=0.0,  # weights never move; activations stream in
    sram_capacity_bytes=64 * 1024**2,  # activation/tile buffers
    hbm_pj_per_bit=0.0,
    sram_pj_per_bit=0.5,
    tdp_watts=1.0,  # 1.0 W power class [E] — NOT used for energy (bottom-up)
    transistors=26e9,  # est. incl. 1T1R cells as 1T each + ADC/DAC + logic
    area_mm2=600.0,
    provenance_sources=(
        "ISAAC (Shafiee et al., ISCA 2016) ADC model: 8-bit SAR ~5 pJ/conv, "
        "256x256 arrays, 4-bit nibble passes; cell/row/shift-add energies are "
        "estimates (docs/references.md §5); throughput = array-pass model. "
        "NOTE these constants are ~32nm-class and do NOT scale with the "
        "selected tech_node, so a CIM row priced beside an N4 ROM row is not "
        "at a common node. Attention has no stationary weights at any context "
        "length, so a crossbar must run it on digital lanes."
    ),
)

# Processing-in-memory (HBM-PIM / DRAM-PIM class, Samsung FIM / UPMEM style).
# Weights live in DRAM banks; MAC units sit beside the sense amps.
#   - In-bank data movement ~2 pJ/bit vs ~5 pJ/bit off-stack (HBM3e)
#   - Published throughput: 1.2 TFLOPS BF16, 1.2 TB/s in-stack bandwidth
#     (Samsung HBM-PIM, ISSCC 2021 / Kwon et al.)
#   - DRAM 1T1C cells counted as 1 transistor per bit in transistor totals
HBM_PIM = PlatformConfig(
    name="HBM-PIM / DRAM-PIM",
    kind="pim",
    flops_per_s=1.2e12,  # published: Samsung HBM-PIM 1.2 TFLOPS BF16
    tile_m=1, tile_n=8, tile_k=64,  # bank-level MAC groups
    hbm_bytes_per_s=1.2e12,  # published: 1.2 TB/s in-stack
    hbm_capacity_bytes=16 * 1024**3,  # 16 GB stacks
    l2_capacity_bytes=0,
    hbm_pj_per_bit=2.0,  # in-bank movement vs ~5 pJ/bit off-stack
    sram_pj_per_bit=0.5,
    tdp_watts=60.0,  # est. PIM stack class (label [E])
    transistors=20e9,  # est. incl. DRAM 1T1C cells (label [E])
    area_mm2=1200.0,  # full stack, est.
    provenance_sources=(
        "published: Samsung HBM-PIM 1.2 TFLOPS / 1.2 TB/s in-stack (ISSCC 2021); "
        "in-bank ~2 pJ/bit vs ~5 pJ/bit off-stack (docs/references.md); "
        "TDP and transistor count are estimates"
    ),
)

PLATFORMS = {
    "h100": H100,
    "b200": B200,
    "tpu_v5e": TPU_V5E,
    "lpu": GROQ_LPU,
    "cim": ANALOG_CIM,
    "pim": HBM_PIM,
}


@dataclass
class StageResult:
    """Per-stage simulation result."""
    stage: str
    layer_idx: int  # -1 for non-layer stages (embedding, lm_head)

    # Workload
    macs: int = 0
    flops: int = 0
    weight_bytes: int = 0
    act_bytes: int = 0

    # Tiling
    tiles_m: int = 1
    tiles_n: int = 1
    tiles_k: int = 1
    total_tiles: int = 1

    # Execution
    compute_cycles: int = 0
    memory_cycles: int = 0
    cycles: int = 0
    utilization: float = 0.0

    # Energy breakdown (joules)
    compute_energy_j: float = 0.0
    weight_energy_j: float = 0.0
    act_energy_j: float = 0.0
    total_energy_j: float = 0.0

    # Memory traffic
    hbm_bytes_read: int = 0
    hbm_bytes_written: int = 0
    sram_bytes_accessed: int = 0


@dataclass
class SimulationResult:
    """Full simulation result for a platform."""
    platform: str
    kind: str
    model_id: str
    sequence_length: int

    # Per-stage results
    stages: list[StageResult] = field(default_factory=list)

    # Totals
    total_macs: int = 0
    total_flops: int = 0
    total_weight_bytes: int = 0
    total_act_bytes: int = 0
    total_cycles: int = 0
    latency_s: float = 0.0
    throughput_toks: float = 0.0

    # Energy totals
    compute_energy_j: float = 0.0
    weight_energy_j: float = 0.0
    act_energy_j: float = 0.0
    total_energy_j: float = 0.0

    # Utilization
    avg_utilization: float = 0.0
    #: Datasheet peak MAC rate (MAC/s). 0.0 when the platform publishes no
    #: FLOPS figure - analog CIM's throughput comes from its array-pass model,
    #: so it has no comparable peak and utilisation is undefined for it.
    peak_macs_per_s: float = 0.0

    # Physical
    transistors: float = 0.0
    area_mm2: float = 0.0

    # Provenance
    provenance: dict = field(default_factory=dict)


def _tile_count(dim: int, tile_size: int) -> int:
    """Number of tiles needed to cover a dimension."""
    return max(1, (dim + tile_size - 1) // tile_size)


def _simulate_stage_gpu(
    stage: dict[str, Any],
    layer_idx: int,
    platform: PlatformConfig,
    clock_hz: float,
) -> StageResult:
    """Simulate a single stage on a GPU platform."""
    macs = stage["macs_per_call"]
    flops = 2 * macs  # MACs -> FLOPs
    weight_bytes = stage["weight_bytes_per_call"]
    act_bytes = stage["act_bytes_per_call"]

    # Tiling: for GEMMs, tile by tensor-core dimensions
    # Simplified: assume weight matrix is [K, N], activation is [M, K]
    # For batch-1 decode, M=1 typically
    tiles_m = 1
    tiles_n = _tile_count(weight_bytes // 2 // 1024, platform.tile_n) if weight_bytes > 0 else 1
    tiles_k = _tile_count(1024, platform.tile_k)  # hidden_size
    total_tiles = tiles_m * tiles_n * tiles_k

    # Compute time
    compute_s = flops / platform.flops_per_s if flops > 0 else 0.0

    # Memory time: weights from HBM, activations from L2 (batch-1)
    weight_s = weight_bytes / platform.hbm_bytes_per_s if weight_bytes > 0 else 0.0
    # Activations mostly fit in L2 for batch-1, minimal HBM traffic
    act_s = 0.0  # Activations in L2

    # Latency is max of compute and memory (roofline)
    latency_s = max(compute_s, weight_s + act_s)

    # Cycles
    compute_cycles = int(compute_s * clock_hz) if compute_s > 0 else 0
    memory_cycles = int((weight_s + act_s) * clock_hz)
    cycles = int(latency_s * clock_hz)

    # Utilization: how much of compute capacity is used
    utilization = compute_s / latency_s if latency_s > 0 else 0.0

    # Energy
    # Compute: TDP fraction while busy. NOTE this is a power-budget heuristic,
    # not a bottom-up count like the ROM/CIM/PIM paths - it dominates the GPU
    # and TPU totals, so total-vs-total comparisons against this chip mix two
    # methodologies. Compare weight_energy_j for a like-for-like mechanism
    # number.
    compute_j = platform.tdp_watts * latency_s * 0.6 if latency_s > 0 else 0.0
    # Weight movement: HBM reads
    weight_j = weight_bytes * 8 * platform.hbm_pj_per_bit / 1e12
    # Activation: L2 SRAM accesses
    act_j = act_bytes * 8 * platform.sram_pj_per_bit / 1e12

    return StageResult(
        stage=stage["stage"],
        layer_idx=layer_idx,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        total_tiles=total_tiles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        cycles=cycles,
        utilization=utilization,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=compute_j + weight_j + act_j,
        hbm_bytes_read=weight_bytes,
        hbm_bytes_written=0,
        sram_bytes_accessed=act_bytes,
    )


def _simulate_stage_tpu(
    stage: dict[str, Any],
    layer_idx: int,
    platform: PlatformConfig,
    clock_hz: float,
) -> StageResult:
    """Simulate a single stage on TPU with systolic MXU."""
    macs = stage["macs_per_call"]
    flops = 2 * macs
    weight_bytes = stage["weight_bytes_per_call"]
    act_bytes = stage["act_bytes_per_call"]

    # TPU MXU: 128x128 systolic array
    # Weights stream through the array (FIFO from HBM)
    tiles_m = _tile_count(1, platform.tile_m)  # batch-1
    tiles_n = _tile_count(weight_bytes // 2 // 128, platform.tile_n) if weight_bytes > 0 else 1
    tiles_k = _tile_count(128, platform.tile_k)
    total_tiles = tiles_m * tiles_n * tiles_k

    compute_s = flops / platform.flops_per_s if flops > 0 else 0.0
    weight_s = weight_bytes / platform.hbm_bytes_per_s if weight_bytes > 0 else 0.0
    latency_s = max(compute_s, weight_s)

    compute_cycles = int(compute_s * clock_hz)
    memory_cycles = int(weight_s * clock_hz)
    cycles = int(latency_s * clock_hz)
    utilization = compute_s / latency_s if latency_s > 0 else 0.0

    compute_j = platform.tdp_watts * latency_s * 0.6 if latency_s > 0 else 0.0
    weight_j = weight_bytes * 8 * platform.hbm_pj_per_bit / 1e12
    act_j = act_bytes * 8 * platform.sram_pj_per_bit / 1e12

    return StageResult(
        stage=stage["stage"],
        layer_idx=layer_idx,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        total_tiles=total_tiles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        cycles=cycles,
        utilization=utilization,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=compute_j + weight_j + act_j,
        hbm_bytes_read=weight_bytes,
        hbm_bytes_written=0,
        sram_bytes_accessed=act_bytes,
    )


def _simulate_stage_lpu(
    stage: dict[str, Any],
    layer_idx: int,
    platform: PlatformConfig,
    clock_hz: float,
) -> StageResult:
    """Simulate a single stage on LPU with on-chip SRAM weights."""
    macs = stage["macs_per_call"]
    flops = 2 * macs
    weight_bytes = stage["weight_bytes_per_call"]
    act_bytes = stage["act_bytes_per_call"]

    # LPU: weights in on-chip SRAM, deterministic dataflow
    # No HBM traffic — everything is on-chip
    tiles_m = 1
    tiles_n = _tile_count(weight_bytes // 2 // 256, platform.tile_n) if weight_bytes > 0 else 1
    tiles_k = _tile_count(256, platform.tile_k)
    total_tiles = tiles_m * tiles_n * tiles_k

    compute_s = flops / platform.flops_per_s if flops > 0 else 0.0
    # Weights traverse SRAM fabric; bandwidth ~64 bytes per FLOP/s
    weight_s = weight_bytes / (platform.flops_per_s / 64) if weight_bytes > 0 else 0.0
    latency_s = max(compute_s, weight_s)

    compute_cycles = int(compute_s * clock_hz)
    memory_cycles = int(weight_s * clock_hz)
    cycles = int(latency_s * clock_hz)
    utilization = compute_s / latency_s if latency_s > 0 else 0.0

    # LPU: TDP-dominated energy (no HBM)
    compute_j = platform.tdp_watts * latency_s * 0.6 if latency_s > 0 else 0.0
    # All memory is SRAM
    weight_j = weight_bytes * 8 * platform.sram_pj_per_bit / 1e12
    act_j = act_bytes * 8 * platform.sram_pj_per_bit / 1e12

    return StageResult(
        stage=stage["stage"],
        layer_idx=layer_idx,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        total_tiles=total_tiles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        cycles=cycles,
        utilization=utilization,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=compute_j + weight_j + act_j,
        hbm_bytes_read=0,  # No HBM
        hbm_bytes_written=0,
        sram_bytes_accessed=weight_bytes + act_bytes,
    )


def _simulate_stage_cim(
    stage: dict[str, Any],
    layer_idx: int,
    platform: PlatformConfig,
    clock_hz: float,
    tech: dict[str, Any] | None = None,
) -> StageResult:
    """Simulate a single stage on an analog CIM crossbar (ISAAC-class).

    Weights are burned into the cell array (like our ROM, but analog multi-bit).
    A GEMM runs as MxV passes over 256x256 arrays:
      - weights sliced into 4-bit nibbles -> 4 passes per BF16 weight operand
      - each pass: row drivers fire, cells conduct, one ADC per output column
      - the MACs happen IN the cells; the digital side shift-adds the per-pass
        partial sums (one add per ADC conversion)
    Energy is priced bottom-up — ADC conversions dominate, and the whole-chip
    TDP term would bury the mechanism the slide teaches.
    """
    macs = stage["macs_per_call"]
    flops = 2 * macs
    weight_bytes = stage["weight_bytes_per_call"]
    act_bytes = stage["act_bytes_per_call"]

    # Crossbar geometry
    arr_rows = platform.tile_k  # 256 inputs per array
    arr_cols = platform.tile_n  # 256 outputs per array
    nibble_passes = 4  # BF16 weight = 4 x 4-bit nibbles (ISAAC bit-splitting)

    # Tiling: how many 256x256 arrays are needed per stage
    tiles_m = 1  # batch-1: one activation vector
    tiles_n = _tile_count(weight_bytes // 2 // arr_rows, arr_cols) if weight_bytes > 0 else 1
    tiles_k = _tile_count(1024, arr_rows)
    total_tiles = tiles_m * tiles_n * tiles_k

    # Time: each pass over an array takes one array-cycle; the ADC conversion
    # period sets the array clock (pipelined rows convert back-to-back)
    array_cycles = total_tiles * nibble_passes
    latency_s = array_cycles / clock_hz
    cycles = array_cycles

    # Utilization: fraction of time arrays are actively converting
    utilization = 1.0 if cycles > 0 else 0.0

    # Energy (bottom-up, per stage call):
    # - ADC: one conversion per output column per pass — DOMINANT
    #   5 pJ per 8-bit SAR conversion (ISAAC energy model, docs/references.md §5)
    adc_convs = tiles_n * arr_cols * nibble_passes * tiles_k
    adc_j = adc_convs * 5.0 / 1e12
    # - Cell reads: weight bits sensed once per pass (weights stationary)
    cell_j = weight_bytes * 8 * 0.01 / 1e12   # ~0.01 pJ/bit charge sensing [E]
    # - Row drives: each array drives its 256 rows once per pass
    row_drives = total_tiles * arr_rows * nibble_passes
    row_j = row_drives * 0.1 / 1e12           # ~0.1 pJ per row drive [E]
    # - Activations: streamed through SRAM buffers
    act_j = act_bytes * 8 * platform.sram_pj_per_bit / 1e12
    # - Digital shift-add: one add per ADC conversion at ~0.1 pJ [E]
    shift_add_j = adc_convs * 0.1 / 1e12

    weight_energy = adc_j + cell_j + row_j

    return StageResult(
        stage=stage["stage"],
        layer_idx=layer_idx,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        total_tiles=total_tiles,
        compute_cycles=0,  # analog: array cycles, not digital compute cycles
        memory_cycles=0,
        cycles=cycles,
        utilization=utilization,
        compute_energy_j=shift_add_j,
        weight_energy_j=weight_energy,
        act_energy_j=act_j,
        total_energy_j=shift_add_j + weight_energy + act_j,
        hbm_bytes_read=0,  # weights never move
        hbm_bytes_written=0,
        sram_bytes_accessed=act_bytes,
    )


def _simulate_stage_pim(
    stage: dict[str, Any],
    layer_idx: int,
    platform: PlatformConfig,
    clock_hz: float,
    tech: dict[str, Any] | None = None,
) -> StageResult:
    """Simulate a single stage on HBM-PIM / DRAM-PIM.

    Weights live in DRAM banks; MAC units sit beside the sense amplifiers.
    Latency comes from the published envelope (1.2 TFLOPS BF16, 1.2 TB/s
    in-stack bandwidth). Energy is priced bottom-up — the stack TDP is not
    published, so a TDP term would be an invented constant driving the result:
      - MACs at the tech node's digital pJ/MAC (bank MACs are digital logic)
      - in-bank weight reads at ~2 pJ/bit (vs ~5 pJ/bit moving off-stack)
      - activations through the stack interface at SRAM-class pJ/bit
    """
    macs = stage["macs_per_call"]
    flops = 2 * macs
    weight_bytes = stage["weight_bytes_per_call"]
    act_bytes = stage["act_bytes_per_call"]

    # Bank-level MAC groups: weights move within banks (in-bank pJ/bit)
    tiles_m = 1
    tiles_n = _tile_count(weight_bytes // 2 // 64, platform.tile_n) if weight_bytes > 0 else 1
    tiles_k = _tile_count(1024, platform.tile_k)
    total_tiles = tiles_m * tiles_n * tiles_k

    # Latency from the published envelope
    compute_s = flops / platform.flops_per_s if flops > 0 else 0.0
    weight_s = weight_bytes / platform.hbm_bytes_per_s if weight_bytes > 0 else 0.0
    latency_s = max(compute_s, weight_s)

    compute_cycles = int(compute_s * clock_hz)
    memory_cycles = int(weight_s * clock_hz)
    cycles = int(latency_s * clock_hz)
    utilization = compute_s / latency_s if latency_s > 0 else 0.0

    # Energy, bottom-up (no TDP term — stack TDP is unpublished)
    mac_pj = (tech or {}).get("mac_pj_per_op", 0.45)
    compute_j = macs * mac_pj / 1e12
    weight_j = weight_bytes * 8 * platform.hbm_pj_per_bit / 1e12  # in-bank 2 pJ/bit
    act_j = act_bytes * 8 * platform.sram_pj_per_bit / 1e12

    return StageResult(
        stage=stage["stage"],
        layer_idx=layer_idx,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        tiles_k=tiles_k,
        total_tiles=total_tiles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        cycles=cycles,
        utilization=utilization,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=compute_j + weight_j + act_j,
        hbm_bytes_read=0,  # in-bank reads, not off-stack traffic
        hbm_bytes_written=0,
        sram_bytes_accessed=weight_bytes + act_bytes,
    )


def simulate(
    platform_key: str,
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
) -> SimulationResult:
    """Run simulation for a platform on a model.

    Args:
        platform_key: One of "h100", "b200", "tpu_v5e", "lpu", "rom"
        model: Model configuration dict
        seq: Sequence length (number of tokens)
        tech_node: Technology node for energy estimates (e.g., "28nm", "N4", "N3")

    Returns:
        SimulationResult with per-stage records and totals
    """
    if platform_key == "rom":
        return _simulate_rom(model, seq, tech_node)

    platform = PLATFORMS[platform_key]
    stages_def = stage_table(model, seq)
    layers = int(model["num_hidden_layers"])

    # Choose simulation function based on platform kind
    sim_fn = {
        "gpu": _simulate_stage_gpu,
        "tpu": _simulate_stage_tpu,
        "lpu": _simulate_stage_lpu,
        "cim": _simulate_stage_cim,
        "pim": _simulate_stage_pim,
    }[platform.kind]

    # Estimate clock from FLOPS and typical operations per cycle
    # GPU: ~2000 ops/cycle at 2 GHz -> ~1e9 Hz effective
    # TPU: ~500 MHz class
    # LPU: ~1 GHz deterministic
    # CIM: ADC conversion period sets the array clock (~500 MHz class)
    # PIM: DRAM bank timing (~500 MHz class)
    clock_hz = {
        "gpu": 1.5e9,
        "tpu": 500e6,
        "lpu": 1.0e9,
        "cim": 500e6,
        "pim": 500e6,
    }[platform.kind]

    stage_results: list[StageResult] = []
    tech = TECH_NODES.get(tech_node, TECH_NODES[DEFAULT_TECH_NODE])

    for stage_def in stages_def:
        count = stage_def["count"]
        for i in range(count):
            layer_idx = i if count > 1 else -1
            if platform.kind in ("cim", "pim"):
                result = sim_fn(stage_def, layer_idx, platform, clock_hz, tech)
            else:
                result = sim_fn(stage_def, layer_idx, platform, clock_hz)
            stage_results.append(result)

    # Aggregate totals
    total_macs = sum(r.macs for r in stage_results)
    total_flops = sum(r.flops for r in stage_results)
    total_weight_bytes = sum(r.weight_bytes for r in stage_results)
    total_act_bytes = sum(r.act_bytes for r in stage_results)
    total_cycles = sum(r.cycles for r in stage_results)

    latency_s = total_cycles / clock_hz
    throughput_toks = 1.0 / latency_s if latency_s > 0 else 0.0

    compute_j = sum(r.compute_energy_j for r in stage_results)
    weight_j = sum(r.weight_energy_j for r in stage_results)
    act_j = sum(r.act_energy_j for r in stage_results)
    total_j = compute_j + weight_j + act_j

    # Average utilization weighted by cycles
    total_util_cycles = sum(r.utilization * r.cycles for r in stage_results)
    avg_util = total_util_cycles / total_cycles if total_cycles > 0 else 0.0

    return SimulationResult(
        platform=platform.name,
        kind=platform.kind,
        # 1 MAC = 2 FLOP. Zero for analog CIM, which publishes no FLOPS.
        peak_macs_per_s=platform.flops_per_s / 2.0,
        model_id=model.get("model_id", "unknown"),
        sequence_length=seq,
        stages=stage_results,
        total_macs=total_macs,
        total_flops=total_flops,
        total_weight_bytes=total_weight_bytes,
        total_act_bytes=total_act_bytes,
        total_cycles=total_cycles,
        latency_s=latency_s,
        throughput_toks=throughput_toks,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=total_j,
        avg_utilization=avg_util,
        transistors=platform.transistors,
        area_mm2=platform.area_mm2,
        provenance={
            "kind": platform.provenance_kind,
            "measured": False,
            "sources": platform.provenance_sources,
        },
    )


def _simulate_rom(
    model: dict[str, Any],
    seq: int,
    tech_node: str = DEFAULT_TECH_NODE,
) -> SimulationResult:
    """Return the MEASURED RTL result for the ROM platform.

    This never uses analytical simulation — it reads the measured counters
    from the evidence directory. Energy is calculated using the specified
    technology node's pJ constants.

    Args:
        model: Model configuration dict
        seq: Sequence length
        tech_node: Technology node for energy estimates (e.g., "28nm", "N4", "N3")
    """
    fused = measured_fused()
    t = totals(model, seq)
    tech = TECH_NODES.get(tech_node, TECH_NODES[DEFAULT_TECH_NODE])

    if fused is None:
        # No measured evidence available — return placeholder
        return SimulationResult(
            platform="Qwen3-0.6B chip (this repo)",
            kind="rom",
            model_id=model.get("model_id", "unknown"),
            sequence_length=seq,
            stages=[],
            total_macs=t["total_macs"],
            total_flops=t["total_flops"],
            total_weight_bytes=t["weight_bytes"],
            total_act_bytes=t["act_bytes"],
            total_cycles=0,
            latency_s=0.0,
            throughput_toks=0.0,
            provenance={
                "kind": "no_measured_evidence",
                "measured": False,
                "sources": "No measured RTL evidence found in evidence/runs/",
                "tech_node": tech_node,
            },
        )

    # Use measured counters
    cycles = int(fused["cycles"])
    macs = int(fused["macs"])
    rom_reads = int(fused.get("rom_reads") or 0)
    clock_hz = float(ASIC["clock_hz"])

    latency_s = cycles / clock_hz
    throughput_toks = 1.0 / latency_s if latency_s > 0 else 0.0

    # Energy from measured counters + tech node pJ constants
    mac_pj = tech["mac_pj_per_op"]
    rom_pj_per_bit = tech["rom_pj_per_bit"]
    sram_pj_per_bit = tech["sram_pj_per_bit"]

    compute_j = macs * mac_pj / 1e12
    weight_j = rom_reads * 256 * rom_pj_per_bit / 1e12  # 256-bit ROM bus
    act_j = t["act_bytes"] * 8 * sram_pj_per_bit / 1e12
    total_j = compute_j + weight_j + act_j

    # Transistor count from compute_metrics ASIC constants
    rom_t = t["weight_bytes"] * 8 * ASIC["rom_transistors_per_bit"]
    sram_t = 4 * 40960 * 1024 * ASIC["sram_transistors_per_bit"]
    compute_t = (ASIC["mac_lanes"] * ASIC["gates_per_lane"] + ASIC["control_gates"]) * 2

    return SimulationResult(
        platform="Qwen3-0.6B chip (this repo)",
        kind="rom",
        # Our ceiling is the array itself: one MAC per lane per clock.
        peak_macs_per_s=float(ASIC["mac_lanes"]) * float(ASIC["clock_hz"]),
        model_id=model.get("model_id", "unknown"),
        sequence_length=seq,
        stages=[],  # Per-stage not available from measured counters
        total_macs=macs,
        total_flops=2 * macs,
        total_weight_bytes=t["weight_bytes"],
        total_act_bytes=t["act_bytes"],
        total_cycles=cycles,
        latency_s=latency_s,
        throughput_toks=throughput_toks,
        compute_energy_j=compute_j,
        weight_energy_j=weight_j,
        act_energy_j=act_j,
        total_energy_j=total_j,
        avg_utilization=macs / (cycles * ASIC["mac_lanes"]) if cycles > 0 else 0.0,
        transistors=rom_t + sram_t + compute_t,
        area_mm2=0.0,  # Requires PDK characterization
        provenance={
            "kind": "measured",
            "measured": True,
            "fused": fused.get("fused", True),
            "source": fused.get("source", "evidence/runs/"),
            "cycles": cycles,
            "macs": macs,
            "rom_reads": rom_reads,
            "tech_node": tech_node,
            "tech_node_name": tech["name"],
            "energy_constants": f"{tech['name']} ({mac_pj} pJ/MAC, {rom_pj_per_bit} pJ/bit ROM)",
        },
    )


def simulate_all(
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
) -> dict[str, SimulationResult]:
    """Run simulation for all platforms.

    Args:
        model: Model configuration dict
        seq: Sequence length
        tech_node: Technology node for ROM energy estimates
    """
    return {
        key: simulate(key, model, seq, tech_node)
        for key in list(PLATFORMS.keys()) + ["rom"]
    }


def result_to_dict(result: SimulationResult) -> dict[str, Any]:
    """Convert SimulationResult to JSON-serializable dict."""
    return {
        "platform": result.platform,
        "kind": result.kind,
        "model_id": result.model_id,
        "sequence_length": result.sequence_length,
        "metrics": {
            "total_macs": result.total_macs,
            "total_flops": result.total_flops,
            "total_weight_bytes": result.total_weight_bytes,
            "total_act_bytes": result.total_act_bytes,
            "total_cycles": result.total_cycles,
            "latency_s": result.latency_s,
            "throughput_toks": result.throughput_toks,
            "compute_energy_j": result.compute_energy_j,
            "weight_energy_j": result.weight_energy_j,
            "act_energy_j": result.act_energy_j,
            "total_energy_j": result.total_energy_j,
            "avg_utilization": result.avg_utilization,
            # Peak vs achieved MAC rate. This is the comparison the table was
            # missing: a memory-bound machine runs far below its own ceiling,
            # so "N times faster" hides how much of its silicon sits idle.
            "peak_macs_per_s": result.peak_macs_per_s,
            "achieved_macs_per_s": (
                result.total_macs / result.latency_s if result.latency_s > 0 else 0.0
            ),
            "peak_utilization": (
                (result.total_macs / result.latency_s) / result.peak_macs_per_s
                if result.latency_s > 0 and result.peak_macs_per_s > 0 else None
            ),
            "transistors": result.transistors,
            "area_mm2": result.area_mm2,
        },
        "provenance": result.provenance,
        "stage_count": len(result.stages),
    }


def generate_golden_output(
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
) -> dict[str, Any]:
    """Generate the golden output JSON for evidence/exec_models.json."""
    results = simulate_all(model, seq, tech_node)
    t = totals(model, seq)
    tech = TECH_NODES.get(tech_node, TECH_NODES[DEFAULT_TECH_NODE])

    return {
        "schema_version": 1,
        "model_id": model.get("model_id", "unknown"),
        "sequence_length": seq,
        "tech_node": tech_node,
        "tech_node_name": tech["name"],
        "energy_references": {
            "baseline": "Horowitz, 'Computing's energy problem', ISSCC 2014, doi:10.1109/ISSCC.2014.6757323",
            "scaling_validation": [
                "Tu et al., '28nm 29.2TFLOPS/W BF16 CIM Processor', ISSCC 2022",
                "'5nm 254-TOPS/W CIM Macro', ISSCC 2022",
                "'3nm 23.2TOPS/W Neural Engine', ISSCC 2024",
            ],
            "tech_node_ref": tech.get("ref", "scaled from Horowitz 2014 baseline"),
        },
        "workload": {
            "total_macs": t["total_macs"],
            "total_flops": t["total_flops"],
            "weight_bytes": t["weight_bytes"],
            "act_bytes": t["act_bytes"],
            "stages": len(t["rows"]),
            # Per-stage rows: the computational model as data (the deck's
            # workload table renders from these; each row is Q = yW work)
            "stages_detail": [
                {
                    "stage": row["stage"],
                    "note": row.get("note", ""),
                    "count": row["count"],
                    "macs_per_call": row["macs_per_call"],
                    "weight_bytes_per_call": row["weight_bytes_per_call"],
                    "act_bytes_per_call": row["act_bytes_per_call"],
                    "total_macs": row["count"] * row["macs_per_call"],
                }
                for row in t["rows"]
            ],
        },
        "platforms": {key: result_to_dict(result) for key, result in results.items()},
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Execution-model engine")
    parser.add_argument("--model", default=str(ROOT / "configs/qwen3_0_6b.json"))
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/exec_models.json")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()) + ["rom", "all"], default="all")
    parser.add_argument("--tech-node", choices=list(TECH_NODES.keys()), default=DEFAULT_TECH_NODE,
                        help="Technology node for energy estimates (default: N4)")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))

    if args.platform == "all":
        output = generate_golden_output(model, args.seq, args.tech_node)
    else:
        result = simulate(args.platform, model, args.seq, args.tech_node)
        output = result_to_dict(result)

    text = json.dumps(output, indent=2) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} (tech node: {args.tech_node})")

    # Print summary
    if args.platform == "all":
        print(f"\nPlatform comparison ({args.tech_node}):")
        for key, data in output["platforms"].items():
            m = data["metrics"]
            print(f"  {data['platform']:30} {m['throughput_toks']:8.1f} tok/s  "
                  f"{m['total_energy_j']*1000:6.1f} mJ  ({data['provenance']['kind']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
