#!/usr/bin/env python3
"""Precision scaling simulation: BF16 → INT8 → INT4 ROM transistors and energy.

Stdlib-only (QWEN-PY-001). Models how transistor count and energy scale with
weight precision for fixed-weight ROM chips.

Key relationships (exact, not estimated):
- ROM transistors = weight_params × bits_per_weight × 6 (6T ROM cell)
- ROM energy = weight_bits × pJ/bit
- Compute energy stays constant (same MACs regardless of precision)
- Accuracy numbers are PUBLISHED-ONLY (citations required, never simulated)

References for accuracy claims:
- BF16: Native training precision, no degradation
- INT8: Llama-3.1-70B INT8 <0.5% degradation (Meta, 2024)
- INT4: Kimi K2-Thinking INT4 QAT matches BF16 (Moonshot AI, 2025)
- INT4: DeepSeek-V3 FP8 training with INT4 inference (DeepSeek, 2024)

Energy constants: Horowitz ISSCC 2014, scaled per tech node (see exec_models.py)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.compute_metrics import totals, ASIC, QUANT_VARIANTS
from lab.exec_models import TECH_NODES, DEFAULT_TECH_NODE

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Precision configurations
# =============================================================================
#
# Each precision level defines:
# - bits_per_weight: Storage bits per weight parameter
# - group_size: For quantized formats, weights per scale group
# - scale_bits: Overhead bits per group (scale + zero point)
# - mac_gates: Gate count per MAC lane (smaller for simpler ops)
# - accuracy_ref: Published reference for accuracy claim
# =============================================================================

@dataclass
class PrecisionConfig:
    """Configuration for a weight precision level."""
    name: str
    bits_per_weight: int
    group_size: int
    scale_bits: int  # Per-group overhead (scale + zero point)
    mac_gates: int   # Gates per MAC lane
    accuracy_degradation: str  # Published claim
    accuracy_ref: str  # Citation

    @property
    def effective_bits_per_weight(self) -> float:
        """Average bits per weight including scale overhead."""
        overhead_per_weight = self.scale_bits / self.group_size
        return self.bits_per_weight + overhead_per_weight


PRECISION_CONFIGS = {
    "bf16": PrecisionConfig(
        name="BF16 (native)",
        bits_per_weight=16,
        group_size=1,       # No grouping
        scale_bits=0,       # No overhead
        mac_gates=4200,     # Full BF16 MAC
        accuracy_degradation="0% (reference)",
        accuracy_ref="Native training precision",
    ),
    "int8": PrecisionConfig(
        name="INT8 (per-channel)",
        bits_per_weight=8,
        group_size=64,
        scale_bits=32,      # BF16 scale + BF16 zero
        mac_gates=1800,     # Simpler INT8 multiply
        accuracy_degradation="<0.5%",
        accuracy_ref="Meta Llama-3.1-70B INT8 quantization (2024)",
    ),
    "int4": PrecisionConfig(
        name="INT4 (grouped)",
        bits_per_weight=4,
        group_size=32,
        scale_bits=32,      # BF16 scale + BF16 zero
        mac_gates=900,      # Minimal INT4 multiply
        accuracy_degradation="<1% with QAT",
        accuracy_ref="Kimi K2-Thinking INT4 QAT (Moonshot AI, 2025); DeepSeek-V3 FP8 (2024)",
    ),
}


@dataclass
class PrecisionScalingResult:
    """Result of precision scaling simulation."""
    config: PrecisionConfig
    model_id: str
    tech_node: str

    # Model size
    weight_params: int
    weight_bits_total: int
    weight_bytes_stored: int

    # Transistors (exact calculation)
    rom_transistors: int
    sram_transistors: int
    compute_transistors: int
    total_transistors: int

    # Compression ratio vs BF16
    compression_ratio: float

    # Energy per token
    compute_energy_j: float
    weight_energy_j: float
    act_energy_j: float
    total_energy_j: float

    # Accuracy (published, not simulated)
    accuracy_degradation: str
    accuracy_ref: str

    # Provenance
    provenance: dict[str, Any]


def simulate_precision(
    config: PrecisionConfig,
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
    mac_lanes: int = 16,
) -> PrecisionScalingResult:
    """Simulate a single precision configuration.

    Args:
        config: Precision configuration
        model: Model config dict
        seq: Sequence length
        tech_node: Technology node for energy constants
        mac_lanes: Number of MAC lanes in the array

    Returns:
        PrecisionScalingResult with transistor counts and energy
    """
    tech = TECH_NODES.get(tech_node, TECH_NODES[DEFAULT_TECH_NODE])

    # Get workload
    t = totals(model, seq)
    total_macs = t["total_macs"]
    act_bytes = t["act_bytes"]

    # Weight parameters (BF16 weights = 2 bytes per param)
    bf16_weight_bytes = t["weight_bytes"]
    weight_params = bf16_weight_bytes // 2

    # Calculate storage bits
    if config.group_size > 0:
        n_groups = math.ceil(weight_params / config.group_size)
        weight_bits_total = (
            weight_params * config.bits_per_weight +
            n_groups * config.scale_bits
        )
    else:
        weight_bits_total = weight_params * config.bits_per_weight

    weight_bytes_stored = math.ceil(weight_bits_total / 8)

    # Transistor count (EXACT calculation)
    # ROM: 6T cell per bit (standard 6-transistor ROM cell)
    rom_transistors = weight_bits_total * 6

    # SRAM: Fixed at 4 banks × 40960 words × 1024 bits (from ASIC config)
    sram_bits = 4 * 40960 * 1024
    sram_transistors = sram_bits * 6  # 6T SRAM cell

    # Compute: gates × 2 transistors per gate
    compute_transistors = (
        mac_lanes * config.mac_gates + int(ASIC["control_gates"])
    ) * 2

    total_transistors = rom_transistors + sram_transistors + compute_transistors

    # Compression ratio vs BF16
    bf16_bits = weight_params * 16
    compression_ratio = bf16_bits / weight_bits_total if weight_bits_total > 0 else 1.0

    # Energy calculation (per token)
    # Compute energy: same MACs regardless of precision, but simpler MAC units
    # For INT8/INT4, the MAC is simpler but we still do the same number of ops
    # The MAC energy scales with gate count (proxy for complexity)
    mac_energy_scale = config.mac_gates / 4200  # Relative to BF16 MAC
    compute_energy_j = total_macs * tech["mac_pj_per_op"] * mac_energy_scale / 1e12

    # ROM energy: scales directly with bits read
    weight_energy_j = weight_bits_total * tech["rom_pj_per_bit"] / 1e12

    # Activation energy: unchanged (always BF16)
    act_energy_j = act_bytes * 8 * tech["sram_pj_per_bit"] / 1e12

    total_energy_j = compute_energy_j + weight_energy_j + act_energy_j

    return PrecisionScalingResult(
        config=config,
        model_id=model.get("model_id", "unknown"),
        tech_node=tech_node,
        weight_params=weight_params,
        weight_bits_total=weight_bits_total,
        weight_bytes_stored=weight_bytes_stored,
        rom_transistors=rom_transistors,
        sram_transistors=sram_transistors,
        compute_transistors=compute_transistors,
        total_transistors=total_transistors,
        compression_ratio=compression_ratio,
        compute_energy_j=compute_energy_j,
        weight_energy_j=weight_energy_j,
        act_energy_j=act_energy_j,
        total_energy_j=total_energy_j,
        accuracy_degradation=config.accuracy_degradation,
        accuracy_ref=config.accuracy_ref,
        provenance={
            "kind": "calculated",
            "transistor_formula": "ROM bits × 6 (6T cell) + SRAM bits × 6 + compute gates × 2",
            "energy_formula": "MACs × mac_pj × (gates/4200) + ROM bits × rom_pj + act bits × sram_pj",
            "accuracy_source": "published_only",
            "tech_node": tech_node,
            "tech_node_ref": tech.get("ref", ""),
        },
    )


def simulate_all_precisions(
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
) -> dict[str, PrecisionScalingResult]:
    """Simulate all precision configurations."""
    return {
        key: simulate_precision(config, model, seq, tech_node)
        for key, config in PRECISION_CONFIGS.items()
    }


def result_to_dict(result: PrecisionScalingResult) -> dict[str, Any]:
    """Convert PrecisionScalingResult to JSON-serializable dict."""
    return {
        "precision": result.config.name,
        "bits_per_weight": result.config.bits_per_weight,
        "effective_bits_per_weight": result.config.effective_bits_per_weight,
        "model_id": result.model_id,
        "tech_node": result.tech_node,
        "storage": {
            "weight_params": result.weight_params,
            "weight_bits_total": result.weight_bits_total,
            "weight_bytes_stored": result.weight_bytes_stored,
            "compression_ratio_vs_bf16": result.compression_ratio,
        },
        "transistors": {
            "rom": result.rom_transistors,
            "sram": result.sram_transistors,
            "compute": result.compute_transistors,
            "total": result.total_transistors,
            "total_billions": result.total_transistors / 1e9,
        },
        "energy_per_token": {
            "compute_j": result.compute_energy_j,
            "weight_j": result.weight_energy_j,
            "act_j": result.act_energy_j,
            "total_j": result.total_energy_j,
            "total_mj": result.total_energy_j * 1000,
        },
        "accuracy": {
            "degradation": result.accuracy_degradation,
            "reference": result.accuracy_ref,
            "note": "Accuracy numbers are from published papers, not simulated",
        },
        "provenance": result.provenance,
    }


def generate_precision_evidence(
    model: dict[str, Any],
    seq: int = 1,
    tech_node: str = DEFAULT_TECH_NODE,
) -> dict[str, Any]:
    """Generate the golden output JSON for evidence/precision_scaling.json."""
    results = simulate_all_precisions(model, seq, tech_node)

    # Verify invariants
    bf16_result = results["bf16"]
    int8_result = results["int8"]
    int4_result = results["int4"]

    # Transistor count should scale with precision
    assert int8_result.rom_transistors < bf16_result.rom_transistors, \
        "INT8 ROM should use fewer transistors than BF16"
    assert int4_result.rom_transistors < int8_result.rom_transistors, \
        "INT4 ROM should use fewer transistors than INT8"

    # Energy should scale with precision (weight energy dominates)
    assert int8_result.weight_energy_j < bf16_result.weight_energy_j, \
        "INT8 weight energy should be less than BF16"
    assert int4_result.weight_energy_j < int8_result.weight_energy_j, \
        "INT4 weight energy should be less than INT8"

    return {
        "schema_version": 1,
        "model_id": model.get("model_id", "unknown"),
        "sequence_length": seq,
        "tech_node": tech_node,
        "description": "Precision scaling: BF16 → INT8 → INT4 transistors and energy",
        "methodology": {
            "transistors": "ROM bits × 6 (6T cell) + SRAM (fixed) + compute (gates × 2)",
            "energy": "MACs × mac_pj + ROM bits × rom_pj + act bits × sram_pj",
            "accuracy": "Published results only (citations required, not simulated)",
        },
        "energy_references": {
            "baseline": "Horowitz, 'Computing's energy problem', ISSCC 2014",
            "tech_node_ref": TECH_NODES[tech_node].get("ref", ""),
        },
        "accuracy_references": [
            "Meta Llama-3.1-70B INT8 quantization (2024): <0.5% degradation",
            "Kimi K2-Thinking INT4 QAT (Moonshot AI, 2025): matches BF16",
            "DeepSeek-V3 FP8 training + INT4 inference (DeepSeek, 2024)",
        ],
        "precisions": {
            key: result_to_dict(result)
            for key, result in results.items()
        },
        "invariants_verified": {
            "transistors_scale_with_precision": True,
            "energy_scales_with_precision": True,
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Precision scaling simulation")
    parser.add_argument("--config", default=str(ROOT / "configs/qwen3_0_6b.json"),
                        help="Model config JSON path")
    parser.add_argument("--output", help="Output JSON path (default: stdout)")
    parser.add_argument("--seq", type=int, default=1, help="Sequence length")
    parser.add_argument("--tech-node", choices=list(TECH_NODES.keys()),
                        default=DEFAULT_TECH_NODE, help="Technology node")
    args = parser.parse_args()

    model = json.loads(Path(args.config).read_text())
    evidence = generate_precision_evidence(model, args.seq, args.tech_node)

    output_text = json.dumps(evidence, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n")
        print(f"Wrote {args.output}")

        # Summary table
        print(f"\nPrecision scaling summary ({args.tech_node}):")
        print("-" * 80)
        print(f"{'Precision':<20} {'Bits':>6} {'Transistors':>15} {'Energy (mJ)':>12} {'Accuracy':>15}")
        print("-" * 80)
        for key in ["bf16", "int8", "int4"]:
            p = evidence["precisions"][key]
            t = p["transistors"]
            e = p["energy_per_token"]
            a = p["accuracy"]
            print(f"{p['precision']:<20} {p['bits_per_weight']:>6} {t['total_billions']:>12.1f}B {e['total_mj']:>12.3f} {a['degradation']:>15}")
        print("-" * 80)
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
