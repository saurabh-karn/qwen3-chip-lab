#!/usr/bin/env python3
"""Per-stage forward-pass cost model: Qwen3-0.6B ASIC vs H100 vs B200.

Dependency-free (standard library only), matching the repo's QWEN-PY-001
rule. Every number is labeled measured or estimated; GPU figures are
datasheet-based analytical estimates, not measurements.

The model decomposes one forward pass (prefill of `seq` tokens) into the
same stages the UI visualizes, computes FLOPs and memory bytes per stage,
then prices them on three platforms:

- asic: this repo's 16-lane BF16 MAC datapath, on-chip BF16 ROM,
  analytical clock/energy from lab/sweep conventions.
- h100: SXM H100, BF16 dense ~989 TFLOPS, HBM3e ~3.35 TB/s, 700 W TDP.
- b200: Blackwell B200, BF16 dense ~2.25 PFLOPS, HBM3e ~8 TB/s, 1000 W TDP.

Energy per token is dominated by weight traffic at batch 1; the model
reports compute, weight-movement, and activation-movement energy
separately so the H100/B200 vs on-chip-ROM tradeoff is visible.
"""

from __future__ import annotations

from lab.energy_constants import (
    MEASURED_TECH_NODE,
    node as energy_node,
)

# The measured RTL counters come from the 28nm lab die; every energy figure
# for THIS chip is priced at that node unless a caller asks for another.
_ASIC_NODE = energy_node(MEASURED_TECH_NODE)

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "qwen3_0_6b.json"

# Datasheet-class analytical constants. Labeled estimated everywhere they
# surface. BF16 dense (no sparsity), boost clocks, published HBM bandwidth.
GPUS = {
    "h100": {
        "name": "NVIDIA H100 SXM",
        "bf16_flops_per_s": 989e12,
        "hbm_bytes_per_s": 3.35e12,
        "tdp_watts": 700.0,
        "hbm_pj_per_bit": 5.0,     # HBM3e ~5 pJ/bit access energy (est.)
        "sram_pj_per_bit": 0.5,    # L2 SRAM ~0.5 pJ/bit (est.)
        "transistors": 80e9,
        "area_mm2": 814.0,
    },
    "b200": {
        "name": "NVIDIA B200",
        "bf16_flops_per_s": 2.25e15,
        "hbm_bytes_per_s": 8.0e12,
        "tdp_watts": 1000.0,
        "hbm_pj_per_bit": 4.5,     # HBM3e improved (est.)
        "sram_pj_per_bit": 0.5,
        "transistors": 208e9,
        "area_mm2": 1600.0,
    },
}

# Quantized weight-ROM variants: same schedule and geometry, weights stored
# at reduced precision in ROM and dequantized on the fly in the MAC lane.
# These are APPROXIMATE architectures — a different model, not bit-exact to
# the BF16 oracle — so they are labeled as variants, never as the lab's
# correctness target.
QUANT_VARIANTS = {
    "int8": {
        "name": "Qwen3-0.6B chip · INT8 weight ROM",
        "weight_bits": 8,
        "mac_pj_per_op": 0.5,          # narrower multiplier is cheaper
        # Per-bit ROM read energy is a property of the node, not of the
        # stored precision: INT8 wins by reading fewer bits, not cheaper
        # ones. Was 0.5 here, inherited from the old 0.8 pJ/bit lineage.
        "rom_pj_per_bit": _ASIC_NODE["rom_pj_per_bit"],
        "gates_per_lane": 1800,
        "group_size": 64,
        "scale_overhead_bits_per_group": 16 + 16,
    },
    "int4": {
        "name": "Qwen3-0.6B chip · INT4 weight ROM",
        "weight_bits": 4,
        "mac_pj_per_op": 0.2,          # narrower multiplier is cheaper
        "rom_pj_per_bit": _ASIC_NODE["rom_pj_per_bit"],
        "gates_per_lane": 900,
        "group_size": 32,
        "scale_overhead_bits_per_group": 16 + 16,
    },
}

# TPU/LPU analytical constants.
TPU = {
    "name": "Google TPU v5e",
    "bf16_flops_per_s": 197e12,
    "hbm_bytes_per_s": 1.6e12,
    "tdp_watts": 170.0,
    "hbm_pj_per_bit": 5.0,
    "sram_pj_per_bit": 0.5,
    # Google unpublished. Est. from Wikipedia die 300–350 mm² × H100-class
    # ~98 MTr/mm² (80B / 814 mm²) → ~32B at 325 mm².
    "transistors": 32e9,
    "area_mm2": 325.0,
}

LPU = {
    "name": "Groq LPU",
    "bf16_flops_per_s": 188e12,
    "hbm_bytes_per_s": 0.0,
    "tdp_watts": 750.0,
    "hbm_pj_per_bit": 0.0,
    "sram_pj_per_bit": 0.5,
    # GroqChip TSP, published-class (~26.8B, ~725 mm², 14nm).
    "transistors": 26.8e9,
    "area_mm2": 725.0,
}

# ASIC analytical constants, consistent with lab/sweep.py conventions.
ASIC = {
    "name": "Qwen3-0.6B chip (this repo)",
    "mac_lanes": 16,
    "clock_hz": 500e6,
    # Analytical gate counts (est.): BF16 mul + FP32 serial add per lane.
    "gates_per_lane": 4200,
    "control_gates": 12000,
    "rom_transistors_per_bit": 6,   # 6T cell
    "sram_transistors_per_bit": 6,  # 6T cell
    # Energy constants come from lab/energy_constants.py at the node this
    # block has always declared (28nm, the lab die). They used to be local
    # literals that disagreed with every other module: rom was 0.8 pJ/bit
    # here versus 0.10 for the same node in the canonical table.
    "rom_pj_per_bit": _ASIC_NODE["rom_pj_per_bit"],
    "sram_pj_per_bit": _ASIC_NODE["sram_pj_per_bit"],
    "mac_pj_per_op": _ASIC_NODE["mac_pj_per_op"],
    "tech_node": _ASIC_NODE["name"] + " (analytical energy)",
}


def stage_table(model: dict[str, Any], seq: int) -> list[dict[str, Any]]:
    """One row per forward-pass stage: MACs, weight bytes, activation bytes."""
    h = int(model["hidden_size"])
    inter = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    nh = int(model["num_attention_heads"])
    nkv = int(model["num_key_value_heads"])
    hd = int(model["head_dim"])
    vocab = int(model["vocab_size"])
    qout, kvout = nh * hd, nkv * hd
    bf16 = 2

    rows: list[dict[str, Any]] = []

    def add(stage: str, count: int, macs: int, weight_bytes: int, act_bytes: int,
            note: str) -> None:
        rows.append({
            "stage": stage,
            "count": count,
            "macs_per_call": macs,
            "weight_bytes_per_call": weight_bytes,
            "act_bytes_per_call": act_bytes,
            "note": note,
        })

    add("embedding", 1, 0, h * bf16, seq * h * bf16,
        "row gather from tied table")
    add("input_rmsnorm", layers, 2 * h, h * bf16, 2 * seq * h * bf16,
        "sum of squares + scale, weight gamma")
    add("qkv_proj", layers, seq * h * (qout + 2 * kvout),
        h * (qout + 2 * kvout) * bf16, seq * (qout + 2 * kvout) * bf16,
        "fused Q/K/V GEMM")
    add("qk_norm", layers, 2 * (qout + kvout), (qout + kvout) * bf16,
        2 * seq * (qout + kvout) * bf16, "per-head RMSNorm")
    add("rope", layers, 0, 0, seq * (qout + kvout) * bf16,
        "rotate-half, table-backed cos/sin")
    add("attn_scores", layers, 2 * nh * hd * seq * (seq + 1) // 2, 0,
        2 * seq * (seq + 1) // 2 * nh * bf16, "Q*K^T over causal context")
    add("softmax", layers, 3 * nh * seq * (seq + 1) // 2, 0,
        2 * nh * seq * (seq + 1) // 2 * bf16, "max/exp/sum/divide")
    add("attn_value", layers, 2 * nh * hd * seq * (seq + 1) // 2, 0,
        seq * qout * bf16, "P*V")
    add("o_proj", layers, seq * h * qout, qout * h * bf16, seq * h * bf16,
        "output projection")
    add("attn_residual", layers, seq * h, 0, 2 * seq * h * bf16, "elementwise add")
    add("post_rmsnorm", layers, 2 * h, h * bf16, 2 * seq * h * bf16, "")
    add("gate_up_proj", layers, seq * h * 2 * inter, h * 2 * inter * bf16,
        seq * 2 * inter * bf16, "fused gate/up GEMM")
    add("silu_swiglu", layers, 3 * seq * inter, 0, 3 * seq * inter * bf16,
        "SiLU + elementwise product")
    add("down_proj", layers, seq * h * inter, inter * h * bf16, seq * h * bf16,
        "")
    add("mlp_residual", layers, seq * h, 0, 2 * seq * h * bf16, "")
    add("final_rmsnorm", 1, 2 * h, h * bf16, 2 * seq * h * bf16, "")
    add("lm_head", 1, seq * h * vocab, h * vocab * bf16, seq * vocab * bf16,
        "tied embedding as head; argmax reduces in flight")
    return rows


def totals(model: dict[str, Any], seq: int) -> dict[str, Any]:
    rows = stage_table(model, seq)
    macs = sum(r["count"] * r["macs_per_call"] for r in rows)
    weight_bytes = sum(r["count"] * r["weight_bytes_per_call"] for r in rows)
    act_bytes = sum(r["count"] * r["act_bytes_per_call"] for r in rows)
    return {
        "rows": rows,
        "total_macs": macs,
        "total_flops": 2 * macs,
        "weight_bytes": weight_bytes,
        "act_bytes": act_bytes,
    }


def gpu_estimate(model: dict[str, Any], seq: int, gpu: dict[str, Any]) -> dict[str, Any]:
    t = totals(model, seq)
    flops = t["total_flops"]
    # Batch-1 prefill: weights stream from HBM once per pass; activations
    # mostly live in L2 (bytes moved HBM-side dominated by weights).
    compute_s = flops / gpu["bf16_flops_per_s"]
    weight_s = t["weight_bytes"] / gpu["hbm_bytes_per_s"]
    act_s = t["act_bytes"] / gpu["hbm_bytes_per_s"]
    latency_s = max(compute_s, weight_s + act_s)
    # Roofline-style energy: TDP share while busy + HBM/SRAM movement energy.
    busy_s = latency_s
    compute_j = gpu["tdp_watts"] * busy_s * 0.6      # est. 60% of TDP on compute
    weight_j = t["weight_bytes"] * 8 * gpu["hbm_pj_per_bit"] / 1e12
    act_j = t["act_bytes"] * 8 * gpu["sram_pj_per_bit"] / 1e12
    return {
        "platform": gpu["name"],
        "kind": "gpu",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "sources": "datasheet-class: BF16 dense TFLOPS, HBM bandwidth, TDP",
        },
        "metrics": {
            "total_flops": flops,
            "weight_bytes": t["weight_bytes"],
            "act_bytes": t["act_bytes"],
            "compute_time_s": compute_s,
            "memory_time_s": weight_s + act_s,
            "latency_s": latency_s,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "act_energy_j": act_j,
            "total_energy_j": compute_j + weight_j + act_j,
            "transistors": gpu["transistors"],
            "area_mm2": gpu["area_mm2"],
        },
    }


def counted_macs(model: dict[str, Any], seq: int = 1, ctx: int = 1) -> int:
    """MACs the unfused controller actually counts (validated vs T4 evidence).

    The schedule re-reduces the full sum-of-squares for every norm output
    element and pads every reduction to a whole 16-lane step. Reconstructing
    that accounting reproduces the measured 668,655,616 MACs from the
    passing T4 run exactly (see tests/test_compute_metrics.py).
    """
    h = int(model["hidden_size"])
    inter = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    nh = int(model["num_attention_heads"])
    nkv = int(model["num_key_value_heads"])
    hd = int(model["head_dim"])
    vocab = int(model["vocab_size"])
    qout, kvout = nh * hd, nkv * hd
    lanes = int(ASIC["mac_lanes"])

    def counted(outputs: int, reduce_len: int) -> int:
        steps = (reduce_len + lanes - 1) // lanes
        return outputs * steps * lanes

    per_layer = (
        counted(h, h)            # input_norm: full re-reduce per element
        + counted(qout, h)       # q_proj
        + counted(kvout, h)      # k_proj
        + counted(kvout, h)      # v_proj
        + counted(qout, hd)      # q_norm
        + counted(kvout, hd)     # k_norm
        + counted(nh * ctx, hd)  # attn scores
        + counted(qout, ctx)     # attn value
        + counted(h, qout)       # o_proj
        + counted(h, 1)          # attn residual (padded)
        + counted(h, h)          # post_norm
        + counted(inter, h)      # gate
        + counted(inter, h)      # up
        + counted(h, inter)      # down
        + counted(h, 1)          # mlp residual (padded)
    )
    return per_layer * layers + counted(vocab, h) + counted(h, h)


def _measured_from(doc: dict[str, Any], path: Path) -> dict[str, Any] | None:
    meta = doc.get("rtl_meta") or {}
    if not meta.get("cycle_count"):
        return None
    return {
        "kind": "measured",
        "measured": True,
        "fused": bool(doc.get("fused_schedule")),
        "source": str(path),
        "cycles": meta["cycle_count"],
        "macs": meta["mac_count"],
        "rom_reads": meta.get("rom_read_count"),
        "sram_reads": meta.get("sram_read_count"),
        "sram_writes": meta.get("sram_write_count"),
        "stalls": meta.get("stall_count"),
        "token_ids": doc.get("token_ids"),
    }


def measured_baseline() -> dict[str, Any] | None:
    """The passing unfused T4 run's measured counters, when present."""
    path = ROOT / "evidence" / "runs" / "c581899d4aa0d74c" / "verify.json"
    if not path.is_file():
        path = ROOT / "evidence" / "verify.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _measured_from(doc, path)


def measured_fused() -> dict[str, Any] | None:
    """The passing fused-schedule T4 run's measured counters, when present."""
    runs = ROOT / "evidence" / "runs"
    if not runs.is_dir():
        return None
    best: dict[str, Any] | None = None
    best_mtime = -1.0
    for run_dir in runs.iterdir():
        path = run_dir / "verify.json"
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not doc.get("fused_schedule") or not doc.get("passed"):
            continue
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = _measured_from(doc, path)
    return best


def asic_estimate(model: dict[str, Any], seq: int) -> dict[str, Any]:
    t = totals(model, seq)
    lanes = int(ASIC["mac_lanes"])
    counted = counted_macs(model, seq)
    # Latency: prefer the MEASURED fused-schedule cycle count (the repo's
    # headline chip). The idealized counted/lanes model assumes a perfect
    # 16 MACs/cycle issue rate, which the fused RTL does not sustain
    # (measured: 608,945,152 MACs over 57,025,007 cycles ~ 10.7 MACs/cycle).
    fused = measured_fused()
    if fused is not None and seq == 1:
        cycles = int(fused["cycles"])
        latency_source = "measured_fused_t4"
        # Energy from the MEASURED counters: the fused run's actual MAC
        # count and ROM reads (not the idealized schedule reconstruction).
        energy_macs = int(fused["macs"])
        energy_rom_bits = int(fused.get("rom_reads") or 0) * 256
        energy_source = "measured_fused_t4_counters"
    else:
        cycles = (counted + lanes - 1) // lanes
        latency_source = "idealized_schedule_model"
        energy_macs = counted
        energy_rom_bits = t["weight_bytes"] * 8
        energy_source = "idealized_schedule_model"
    latency_s = cycles / ASIC["clock_hz"]
    mac_j = energy_macs * ASIC["mac_pj_per_op"] / 1e12
    rom_j = energy_rom_bits * ASIC["rom_pj_per_bit"] / 1e12
    sram_j = t["act_bytes"] * 8 * ASIC["sram_pj_per_bit"] / 1e12
    # Transistor estimate: weights are 6T cells; compute is gate-count based
    # (2 transistors per gate, analytical).
    rom_t = t["weight_bytes"] * 8 * ASIC["rom_transistors_per_bit"]
    # SRAM: 4 banks x 40960 words x 1024 bits = 167,772,160 bits (20.97 MB).
    sram_t = 4 * 40960 * 1024 * ASIC["sram_transistors_per_bit"]
    compute_t = (lanes * ASIC["gates_per_lane"] + ASIC["control_gates"]) * 2
    return {
        "platform": ASIC["name"],
        "kind": "asic",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": fused is not None and seq == 1,
            "latency_source": latency_source,
            "energy_source": energy_source,
            "sources": ("measured fused T4 cycles + MAC/ROM counters "
                        "(evidence/runs); energy = measured counters x "
                        "28nm-class pJ constants (analytical); 6T memory "
                        "cells; gate-count compute"),
        },
        "metrics": {
            "total_flops": t["total_flops"],
            "useful_macs": t["total_macs"],
            "counted_macs": counted,
            "energy_macs": energy_macs,
            "latency_source": latency_source,
            "energy_source": energy_source,
            "weight_bytes": t["weight_bytes"],
            "act_bytes": t["act_bytes"],
            "cycles": cycles,
            "latency_s": latency_s,
            "compute_energy_j": mac_j,
            "weight_energy_j": rom_j,
            "act_energy_j": sram_j,
            "total_energy_j": mac_j + rom_j + sram_j,
            "transistors": rom_t + sram_t + compute_t,
            "rom_transistors": rom_t,
            "sram_transistors": sram_t,
            "compute_transistors": compute_t,
            "area_mm2": None,  # requires PDK characterization
        },
    }


def quantized_estimate(model: dict[str, Any], seq: int, variant: dict[str, Any]) -> dict[str, Any]:
    """Analytical estimate for a fixed-weight quantized-ROM chip variant.

    Weights live in ROM at weight_bits precision with per-group scales.
    The MAC lane dequantizes (INT weight x fp16 scale) and accumulates in
    INT32; activations stay BF16 and are quantized on the fly at the lane.
    Energy and transistor counts scale with the reduced ROM width; latency
    is unchanged (same MAC count, same schedule).
    """
    t = totals(model, seq)
    lanes = int(ASIC["mac_lanes"])
    counted = counted_macs(model, seq)
    cycles = (counted + lanes - 1) // lanes
    latency_s = cycles / ASIC["clock_hz"]

    wb = int(variant["weight_bits"])
    params = t["weight_bytes"] // 2  # BF16 weights -> parameter count
    weight_bits_total = params * wb
    groups = params // int(variant["group_size"])
    weight_bits_total += groups * int(variant["scale_overhead_bits_per_group"])
    weight_bytes_stored = weight_bits_total // 8

    mac_j = counted * variant["mac_pj_per_op"] / 1e12
    rom_j = weight_bits_total * variant["rom_pj_per_bit"] / 1e12
    sram_j = t["act_bytes"] * 8 * ASIC["sram_pj_per_bit"] / 1e12

    rom_t = weight_bits_total * ASIC["rom_transistors_per_bit"]
    # SRAM: 4 banks x 40960 words x 1024 bits = 167,772,160 bits (20.97 MB).
    sram_t = 4 * 40960 * 1024 * ASIC["sram_transistors_per_bit"]
    compute_t = (lanes * variant["gates_per_lane"] + ASIC["control_gates"]) * 2

    bf16_ref = t["weight_bytes"] * 8
    return {
        "platform": variant["name"],
        "kind": "asic-variant",
        "variant": f"int{wb}",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "approximate_model": True,
            "note": "Assumes precision-aware training (QAT): the model is "
                    "trained with fake quantization so the final checkpoint "
                    "is natively low-bit — the ROM mask is that checkpoint, "
                    "not a post-hoc quantization of a BF16 model. Published "
                    "results (Kimi K2-Thinking INT4 QAT, DeepSeek-V3 FP8) "
                    "show accuracy matching full precision. Energy/area "
                    "architecture study for this variant.",
        },
        "metrics": {
            "total_flops": t["total_flops"],
            "useful_macs": t["total_macs"],
            "counted_macs": counted,
            "weight_bits_per_param": wb,
            "weight_bytes_stored": weight_bytes_stored,
            "rom_compression_vs_bf16": bf16_ref / max(1, weight_bits_total // 8),
            "act_bytes": t["act_bytes"],
            "cycles": cycles,
            "latency_s": latency_s,
            "compute_energy_j": mac_j,
            "weight_energy_j": rom_j,
            "act_energy_j": sram_j,
            "total_energy_j": mac_j + rom_j + sram_j,
            "transistors": rom_t + sram_t + compute_t,
            "rom_transistors": rom_t,
            "sram_transistors": sram_t,
            "compute_transistors": compute_t,
            "area_mm2": None,
        },
    }



def lpu_estimate(model: dict[str, Any], seq: int) -> dict[str, Any]:
    """Groq LPU: weights resident in on-chip SRAM, deterministic dataflow.

    Like our ASIC, the LPU never streams weights from off-chip memory at
    inference — the memory-vs-compute energy advantage the ASIC wins also
    applies to the LPU. The differences vs our ASIC: (1) the LPU is a
    general tensor processor sized for many models, so its compute is ~300x
    wider; (2) its batch-1 latency is dominated by scheduling the full
    weight set through the SRAM fabric; (3) it burns far more static power
    (750 W TDP for the system).
    """
    t = totals(model, seq)
    flops = t["total_flops"]
    compute_s = flops / LPU["bf16_flops_per_s"]
    # Weights traverse the SRAM fabric once per pass (like our ROM reads);
    # fabric bandwidth estimated at ~64B per FLOP/s-unit.
    weight_s = t["weight_bytes"] / (LPU["bf16_flops_per_s"] / 64)
    latency_s = max(compute_s, weight_s)
    compute_j = LPU["tdp_watts"] * latency_s * 0.6
    weight_j = t["weight_bytes"] * 8 * LPU["sram_pj_per_bit"] / 1e12
    act_j = t["act_bytes"] * 8 * LPU["sram_pj_per_bit"] / 1e12
    return {
        "platform": LPU["name"],
        "kind": "lpu",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "sources": "datasheet-class: BF16 TFLOPS, SRAM-resident weights, TDP",
        },
        "metrics": {
            "total_flops": flops,
            "weight_bytes": t["weight_bytes"],
            "act_bytes": t["act_bytes"],
            "compute_time_s": compute_s,
            "memory_time_s": weight_s,
            "latency_s": latency_s,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "act_energy_j": act_j,
            "total_energy_j": compute_j + weight_j + act_j,
            "transistors": None,
            "area_mm2": None,
        },
    }


def compare(model: dict[str, Any], seq: int) -> dict[str, Any]:
    t = totals(model, seq)
    platforms = (
        [asic_estimate(model, seq)]
        + [quantized_estimate(model, seq, v) for v in QUANT_VARIANTS.values()]
        + [lpu_estimate(model, seq)]
        + [gpu_estimate(model, seq, g) for g in GPUS.values()]
        + [gpu_estimate(model, seq, TPU)]
    )
    baseline = measured_baseline()
    fused = measured_fused()
    # The named claim: GPU energy is dominated by memory movement. Evaluate
    # it on the GPU platforms (H100 is platforms[4]; B200 platforms[5]),
    # not on the ASIC variants.
    gpu_platforms = [p for p in platforms if p["kind"] == "gpu"]
    memory_energy_dominates_gpu = all(
        p["metrics"]["weight_energy_j"] > p["metrics"]["compute_energy_j"]
        for p in gpu_platforms)
    return {
        "schema_version": 1,
        "model_id": model.get("model_id", "Qwen/Qwen3-0.6B"),
        "sequence_length": seq,
        "stages": t["rows"],
        "totals": {
            "total_macs": t["total_macs"],
            "total_flops": t["total_flops"],
            "weight_bytes": t["weight_bytes"],
            "act_bytes": t["act_bytes"],
        },
        "platforms": platforms,
        "measured_baseline": baseline,
        "measured_fused": fused,
        "headline": {
            "weight_transistors_dominates": platforms[0]["metrics"]["rom_transistors"]
            > 10 * platforms[0]["metrics"]["compute_transistors"],
            "memory_energy_dominates_gpu": memory_energy_dominates_gpu,
            "memory_energy_dominates_gpu_note": (
                "GPU weight-movement energy exceeds GPU compute energy "
                "(batch-1 prefill). At batch>32 GPUs amortize the weight "
                "stream; see lab/scale_1t.py for batch analysis."),
        },
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "warning": "All figures are analytical estimates, not measurements. "
                       "GPU numbers are datasheet-class; ASIC numbers follow repo "
                       "sweep conventions and need PDK characterization.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    result = compare(model, args.seq)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
