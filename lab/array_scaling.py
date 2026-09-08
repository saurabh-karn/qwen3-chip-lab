#!/usr/bin/env python3
"""Array scaling simulation: parameterized 2D systolic array from 16×1 to 128×128.

Stdlib-only (QWEN-PY-001). Models how throughput and energy scale as the
ROM-stationary MXU grows from the lab prototype (16 lanes) to production
configurations (up to 128×128 = 16,384 PEs).

Key insight: energy per token stays ~constant because:
- Compute energy scales with MACs (fixed per token)
- ROM read energy scales with weight bits (fixed per token)
- Larger arrays just finish faster, not more efficiently

References:
- Measured anchor: 16×1 @ 500 MHz → 8.8 tok/s (evidence/runs/)
- Clock scaling: 500 MHz (lab) to 2 GHz (production, TSMC N4 target)
- Energy constants: Horowitz ISSCC 2014, scaled per tech node (see exec_models.py)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.compute_metrics import totals, measured_fused, ASIC
from lab.exec_models import TECH_NODES, DEFAULT_TECH_NODE

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Array configurations
# =============================================================================
#
# The baseline is the measured 16-lane (16×1) array at 500 MHz.
# Production targets scale up to 128×128 at 2 GHz.
#
# Array geometry: rows × cols where each PE does 1 MAC/cycle.
# Total PEs = rows × cols. MACs/cycle = PEs (fully utilized).
# =============================================================================

@dataclass
class ArrayConfig:
    """Configuration for a 2D systolic array."""
    name: str
    rows: int
    cols: int
    clock_hz: float
    tech_node: str = DEFAULT_TECH_NODE

    @property
    def pes(self) -> int:
        """Total processing elements."""
        return self.rows * self.cols

    @property
    def macs_per_cycle(self) -> int:
        """MACs executed per clock cycle (assuming full utilization)."""
        return self.pes


# Predefined array configurations
ARRAY_CONFIGS = {
    # Lab prototype (measured anchor)
    "16x1_500mhz": ArrayConfig("16×1 Lab (measured)", 16, 1, 500e6, "28nm"),

    # MEASURED width sweep: widening the reduction dimension only (rows x 1).
    # Every one of these is a real Verilator run, bit-exact against 16x1.
    "32x1_500mhz": ArrayConfig("32×1 (measured)", 32, 1, 500e6, "28nm"),
    "64x1_500mhz": ArrayConfig("64×1 (measured)", 64, 1, 500e6, "28nm"),
    "128x1_500mhz": ArrayConfig("128×1 (measured)", 128, 1, 500e6, "28nm"),

    # Scaling series at 500 MHz (lab clock). cols > 1 widens the OUTPUT
    # dimension, which is still modelled - no RTL exists for it yet.
    "16x16_500mhz": ArrayConfig("16×16 @ 500 MHz", 16, 16, 500e6, "28nm"),
    "32x32_500mhz": ArrayConfig("32×32 @ 500 MHz", 32, 32, 500e6, "28nm"),
    "64x64_500mhz": ArrayConfig("64×64 @ 500 MHz", 64, 64, 500e6, "28nm"),
    "128x128_500mhz": ArrayConfig("128×128 @ 500 MHz", 128, 128, 500e6, "28nm"),

    # Production series at 2 GHz (N4 target)
    "16x16_2ghz": ArrayConfig("16×16 @ 2 GHz", 16, 16, 2e9, "N4"),
    "32x32_2ghz": ArrayConfig("32×32 @ 2 GHz", 32, 32, 2e9, "N4"),
    "64x64_2ghz": ArrayConfig("64×64 @ 2 GHz", 64, 64, 2e9, "N4"),
    "128x128_2ghz": ArrayConfig("128×128 @ 2 GHz", 128, 128, 2e9, "N4"),
}


@dataclass
class ArrayScalingResult:
    """Result of array scaling simulation."""
    config: ArrayConfig
    model_id: str
    sequence_length: int

    # Workload (constant across array sizes)
    total_macs: int
    total_flops: int
    weight_bytes: int
    act_bytes: int

    # Performance (scales with array size)
    total_cycles: int
    latency_s: float
    throughput_toks: float

    # Energy (roughly constant per token)
    compute_energy_j: float
    weight_energy_j: float
    act_energy_j: float
    total_energy_j: float

    # Hardware
    utilization: float
    transistors: int

    # Provenance
    provenance: dict[str, Any]


_WIDTH_SWEEP_PATH = ROOT / "evidence" / "array_width_sweep.json"


def _measured_width_sweep() -> dict[int, dict]:
    """Measured cycles per MAC-array width, keyed by lane count.

    Produced by qwen_chip/sim at MAC_LANES=16/32/64/128 (FUSED=1), each
    verified bit-exact against the 16-lane reference. These replace the
    util_factor guesses for the widths that were actually run.
    """
    try:
        doc = json.loads(_WIDTH_SWEEP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for point in doc.get("points", []):
        if point.get("bit_exact_vs_16_lane"):
            out[int(point["mac_lanes"])] = point
    return out


def simulate_array(
    config: ArrayConfig,
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str | None = None,
) -> ArrayScalingResult:
    """Simulate a single array configuration.

    Args:
        config: Array configuration (rows, cols, clock)
        model: Model config dict
        seq: Sequence length
        tech_node: Override tech node (default: use config's tech_node)

    Returns:
        ArrayScalingResult with performance and energy metrics
    """
    tech_node = tech_node or config.tech_node
    tech = TECH_NODES.get(tech_node, TECH_NODES[DEFAULT_TECH_NODE])

    # Get workload from stage table
    t = totals(model, seq)
    total_macs = t["total_macs"]
    total_flops = t["total_flops"]
    weight_bytes = t["weight_bytes"]
    act_bytes = t["act_bytes"]

    # Check for measured anchor (16×1 @ 500 MHz)
    is_measured_anchor = (
        config.rows == 16 and config.cols == 1 and
        config.clock_hz == 500e6 and seq == 1
    )
    fused = measured_fused() if is_measured_anchor else None

    # A measured width sweep exists for 1-D widenings (rows x 1). Prefer it
    # over the analytical model: these are real Verilator cycle counts, each
    # bit-exact against the 16-lane reference.
    sweep = _measured_width_sweep()
    swept = sweep.get(config.rows) if config.cols == 1 and config.clock_hz == 500e6 else None
    if swept is not None and not is_measured_anchor:
        total_cycles = int(swept["cycles"])
        energy_macs = int(swept["macs"])
        # Each ROM read returns one word, and the word IS the array width in
        # BF16 lanes - so a 32-lane build reads 512 bits per access, not 256.
        # Pricing every width at 256 b would have made weight energy halve on
        # each doubling, which is exactly the "energy stays flat" claim the
        # scaling series is supposed to demonstrate.
        energy_rom_bits = int(swept["rom_reads"]) * int(swept["rom_data_w"])
        provenance_kind = "measured"
        provenance_source = (
            f"evidence/array_width_sweep.json MAC_LANES={config.rows} "
            "(Verilator, bit-exact vs 16-lane)"
        )
    elif fused is not None:
        # Use measured cycles for the anchor point
        total_cycles = int(fused["cycles"])
        energy_macs = int(fused["macs"])
        energy_rom_bits = int(fused.get("rom_reads") or 0) * 256
        provenance_kind = "measured"
        provenance_source = fused.get("source", "evidence/runs/")
    else:
        # Analytical model: cycles = MACs / (PEs per cycle)
        # This assumes perfect utilization; real utilization is lower due to
        # pipeline bubbles, memory stalls, etc.
        ideal_cycles = math.ceil(total_macs / config.macs_per_cycle)

        # Apply utilization factor based on array size.
        #
        # WARNING: these four numbers are ASSUMPTIONS, not measurements, and
        # they are what the deck's ~40,000 tok/s claim rests on. The only
        # measured point is 16x1, which achieves 66.7% (608,945,152 MACs over
        # 57,025,007 cycles = 10.68 of 16 lanes). Everything below assumes
        # utilization *improves* with size, up to 90%. At batch 1 that is
        # likely backwards: one activation vector (~1024 elements) has to keep
        # the whole array fed, so a bigger array is harder to saturate, not
        # easier. Measuring could move the throughput claim DOWN.
        #
        # Measuring it for real is blocked on one located RTL limitation, not
        # on the model: the datapath width is now parameterised end to end
        # (qwen_mmap_rom DPI, the ROM/SRAM address maps, qwen3_stream_gemm and
        # the testbench all take MAC_LANES), and a width-16 build is
        # bit-identical to this anchor, and the GEMM path is width-clean
        # (V_PROJ commits match exactly at MAC_LANES=32). The blocker is SRAM
        # address alignment: only some controller address paths run through
        # aligned_sram_addr(), so at a 64-byte word (SRAM_DATA_W=512) the
        # others straddle words and read a neighbour buffer. MAC_LANES=32 ran
        # 30,433,423 cycles but returned argmax 198 instead of 21806, first
        # diverging at Q_NORM by a constant 1.3914x scale (its sum of squares
        # comes out ~0.517x correct). The controller now refuses MAC_LANES
        # != 16 rather than emit that number. Making every SRAM address path
        # word-aligned would let this whole table become measured cycles
        # instead of util_factor guesses.
        if config.pes <= 16:
            util_factor = 0.67  # Measured from RTL
        elif config.pes <= 256:
            util_factor = 0.75
        elif config.pes <= 1024:
            util_factor = 0.82
        elif config.pes <= 4096:
            util_factor = 0.87
        else:
            util_factor = 0.90

        total_cycles = int(ideal_cycles / util_factor)
        energy_macs = total_macs
        energy_rom_bits = weight_bytes * 8
        provenance_kind = "analytical_estimate"
        provenance_source = (
            f"array_scaling model; utilization {util_factor:.2f} assumed "
            "(only the 16x1 point is measured, at 0.67)"
        )

    # Latency and throughput
    latency_s = total_cycles / config.clock_hz
    throughput_toks = 1.0 / latency_s if latency_s > 0 else 0.0

    # Energy calculation using tech node constants
    # Key insight: energy per token is ~constant regardless of array size
    # because the work (MACs, ROM reads) is fixed per token
    compute_energy_j = energy_macs * tech["mac_pj_per_op"] / 1e12
    weight_energy_j = energy_rom_bits * tech["rom_pj_per_bit"] / 1e12
    act_energy_j = act_bytes * 8 * tech["sram_pj_per_bit"] / 1e12
    total_energy_j = compute_energy_j + weight_energy_j + act_energy_j

    # Utilization: actual MACs/cycle vs theoretical max
    actual_macs_per_cycle = total_macs / total_cycles if total_cycles > 0 else 0
    utilization = actual_macs_per_cycle / config.macs_per_cycle

    # Transistor count: ROM (6T per bit) + compute (gates × 2)
    # Reference: ASIC config in compute_metrics.py
    rom_transistors = weight_bytes * 8 * ASIC["rom_transistors_per_bit"]
    sram_transistors = 4 * 40960 * 1024 * ASIC["sram_transistors_per_bit"]  # Fixed SRAM
    compute_transistors = (
        config.pes * ASIC["gates_per_lane"] + ASIC["control_gates"]
    ) * 2
    transistors = int(rom_transistors + sram_transistors + compute_transistors)

    return ArrayScalingResult(
        config=config,
        model_id=model.get("model_id", "unknown"),
        sequence_length=seq,
        total_macs=total_macs,
        total_flops=total_flops,
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        total_cycles=total_cycles,
        latency_s=latency_s,
        throughput_toks=throughput_toks,
        compute_energy_j=compute_energy_j,
        weight_energy_j=weight_energy_j,
        act_energy_j=act_energy_j,
        total_energy_j=total_energy_j,
        utilization=utilization,
        transistors=transistors,
        provenance={
            "kind": provenance_kind,
            "measured": provenance_kind == "measured",
            "source": provenance_source,
            "tech_node": tech_node,
            "tech_node_ref": tech.get("ref", ""),
        },
    )


def simulate_scaling_series(
    model: dict[str, Any],
    seq: int = 1,
    clock_hz: float = 500e6,
    tech_node: str = "28nm",
    array_sizes: list[tuple[int, int]] | None = None,
) -> list[ArrayScalingResult]:
    """Simulate a series of array sizes at a fixed clock.

    Args:
        model: Model config dict
        seq: Sequence length
        clock_hz: Clock frequency in Hz
        tech_node: Technology node for energy constants
        array_sizes: List of (rows, cols) tuples, or None for default series

    Returns:
        List of ArrayScalingResult, one per array size
    """
    if array_sizes is None:
        array_sizes = [
            (16, 1),    # Lab prototype (measured)
            # Measured width sweep: reduction dimension only, each bit-exact
            # against 16x1 (evidence/array_width_sweep.json).
            (32, 1),
            (64, 1),
            (128, 1),
            (16, 16),   # 256 PEs
            (32, 32),   # 1,024 PEs
            (64, 64),   # 4,096 PEs
            (128, 128), # 16,384 PEs
        ]

    results = []
    for rows, cols in array_sizes:
        config = ArrayConfig(
            name=f"{rows}×{cols} @ {clock_hz/1e6:.0f} MHz",
            rows=rows,
            cols=cols,
            clock_hz=clock_hz,
            tech_node=tech_node,
        )
        result = simulate_array(config, model, seq, tech_node)
        results.append(result)

    return results


def result_to_dict(result: ArrayScalingResult) -> dict[str, Any]:
    """Convert ArrayScalingResult to JSON-serializable dict."""
    return {
        "config": {
            "name": result.config.name,
            "rows": result.config.rows,
            "cols": result.config.cols,
            "pes": result.config.pes,
            "clock_hz": result.config.clock_hz,
            "tech_node": result.config.tech_node,
        },
        "model_id": result.model_id,
        "sequence_length": result.sequence_length,
        "workload": {
            "total_macs": result.total_macs,
            "total_flops": result.total_flops,
            "weight_bytes": result.weight_bytes,
            "act_bytes": result.act_bytes,
        },
        "metrics": {
            "total_cycles": result.total_cycles,
            "latency_s": result.latency_s,
            "throughput_toks": result.throughput_toks,
            "compute_energy_j": result.compute_energy_j,
            "weight_energy_j": result.weight_energy_j,
            "act_energy_j": result.act_energy_j,
            "total_energy_j": result.total_energy_j,
            "energy_mj": result.total_energy_j * 1000,
            "utilization": result.utilization,
            "transistors": result.transistors,
        },
        "provenance": result.provenance,
    }


def generate_scaling_evidence(
    model: dict[str, Any],
    seq: int = 1,
) -> dict[str, Any]:
    """Generate the golden output JSON for evidence/array_scaling.json."""
    # Lab series: 500 MHz, 28nm
    lab_results = simulate_scaling_series(
        model, seq, clock_hz=500e6, tech_node="28nm"
    )

    # Production series: 2 GHz, N4
    prod_results = simulate_scaling_series(
        model, seq, clock_hz=2e9, tech_node="N4"
    )

    # Verify invariants
    lab_macs = [r.total_macs for r in lab_results]
    prod_macs = [r.total_macs for r in prod_results]
    assert len(set(lab_macs)) == 1, "MAC count must be constant across array sizes"
    assert len(set(prod_macs)) == 1, "MAC count must be constant across array sizes"

    # Verify throughput is monotone increasing
    lab_toks = [r.throughput_toks for r in lab_results]
    prod_toks = [r.throughput_toks for r in prod_results]
    assert lab_toks == sorted(lab_toks), "Lab throughput must be monotone increasing"
    assert prod_toks == sorted(prod_toks), "Prod throughput must be monotone increasing"

    return {
        "schema_version": 1,
        "model_id": model.get("model_id", "unknown"),
        "sequence_length": seq,
        "description": "Array scaling: 16×1 → 128×128, showing throughput scales with PEs while energy stays constant",
        "energy_references": {
            "baseline": "Horowitz, 'Computing's energy problem', ISSCC 2014",
            "measured_anchor": "16×1 @ 500 MHz from evidence/runs/ (8.8 tok/s)",
        },
        "lab_series": {
            "clock_hz": 500e6,
            "tech_node": "28nm",
            "arrays": [result_to_dict(r) for r in lab_results],
        },
        "production_series": {
            "clock_hz": 2e9,
            "tech_node": "N4",
            "arrays": [result_to_dict(r) for r in prod_results],
        },
        "invariants_verified": {
            "mac_count_constant": True,
            "throughput_monotone": True,
            "anchor_matched": lab_results[0].provenance["kind"] == "measured",
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Array scaling simulation")
    parser.add_argument("--config", default=str(ROOT / "configs/qwen3_0_6b.json"),
                        help="Model config JSON path")
    parser.add_argument("--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--seq", type=int, default=1, help="Sequence length")
    args = parser.parse_args()

    model = json.loads(Path(args.config).read_text())
    evidence = generate_scaling_evidence(model, args.seq)

    output_text = json.dumps(evidence, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n")
        print(f"Wrote {args.output}")

        # Summary table
        print("\nArray scaling summary:")
        print("-" * 70)
        print(f"{'Config':<25} {'PEs':>8} {'tok/s':>10} {'Energy (mJ)':>12}")
        print("-" * 70)
        for series_name, series_key in [("Lab (500 MHz)", "lab_series"),
                                         ("Prod (2 GHz)", "production_series")]:
            for arr in evidence[series_key]["arrays"]:
                cfg = arr["config"]
                m = arr["metrics"]
                print(f"{cfg['name']:<25} {cfg['pes']:>8,} {m['throughput_toks']:>10,.1f} {m['energy_mj']:>12.3f}")
            print("-" * 70)
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
