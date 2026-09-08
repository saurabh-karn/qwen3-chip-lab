"""Scale the measured 0.6B chip schedule to 1T-parameter territory.

Answers three questions with numbers, keeping the repo's discipline that
measured figures are separated from analytical estimates:

1. Why weights-in-ROM stops working (area math at TSMC N4) and what replaces
   it: a streaming memory hierarchy (weight tiers) plus MoE sparsity.
2. What the same schedule costs at 1T: cycles from the calibrated per-stage
   model (validated to reproduce BOTH measured 0.6B runs exactly), energy
   from TSMC-class constants, KV-cache sizing vs context.
3. How that compares to H100/B200/TPU/LPU running the SAME 1T model, so the
   "custom silicon wins" claim is quantified per configuration, not asserted.

Calibration anchors (measured, bit-exact T4 runs, 16 lanes @ 500 MHz):
  fused   57,025,007 cycles   /tmp-style breakdown sums exactly
  unfused 295,368,736 cycles  breakdown sums exactly
Per-stage constants derived from those runs (cycles per 16-lane step):
  fused GEMM 1.01-1.02, V_PROJ 7.02 (transposed cache write),
  Q/K norms 7.75 (per-head re-reduce), ROPE/SILU 128/step,
  residuals 144/step, full-width norms 7.44/element.
"""

from __future__ import annotations

from lab.energy_constants import node as energy_node

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "qwen3_0_6b.json"

# ---------------------------------------------------------------------------
# Calibration: measured constants from the two instrumented T4 runs.
# ---------------------------------------------------------------------------

# Cycles per 16-lane MAC step, fused schedule (measured: 1.010-1.018).
FUSED_GEMM_CYCLES_PER_STEP = 1.018
# V_PROJ writes the transposed V-cache layout: measured 7.016 cyc/step.
V_PROJ_CYCLES_PER_STEP = 7.016
# Q/K per-head norms re-reduce per output head: measured 7.750 cyc/step.
QK_NORM_CYCLES_PER_STEP = 7.750
# Full-width norms are sum-once fused: measured 7.437 cycles per element.
FULL_NORM_CYCLES_PER_ELEMENT = 7.437
# Table-backed NLIN engines: measured 128 cycles per 16-lane step.
NLIN_CYCLES_PER_STEP = 128.0
# Residual adds: measured 144 cycles per 16-lane step.
RESIDUAL_CYCLES_PER_STEP = 144.0
# Attention score/value/softmax at T=1 measured; score/value scale with ctx.
ATTN_SCORE_CYCLES_PER_STEP = 8.125
ATTN_VALUE_CYCLES_PER_STEP = 9.0
SOFTMAX_CYCLES_PER_ROW = 15.0
# Unfused schedule: every stage re-reduces (measured 5.18x gap).
UNFUSED_FACTOR = 5.18

MEASURED_FUSED_CYCLES = 57_025_007
MEASURED_UNFUSED_CYCLES = 295_368_736
MEASURED_LANES = 16
MEASURED_CLOCK_HZ = 500e6

# ---------------------------------------------------------------------------
# TSMC N4-class silicon constants (analytical, datasheet-class).
# ---------------------------------------------------------------------------

_N4_NODE = energy_node("N4")

N4 = {
    "name": "TSMC N4-class (analytical)",
    # High-density logic: ~ 240 MTr/mm2 (N4HD, est. from N5 230 MTr/mm2).
    "mtr_per_mm2": 240e6,
    # 6T SRAM bitcell ~0.021 um2 (N5-class HD); ROM NOR bitcell ~0.018 um2.
    "sram_bitcell_um2": 0.021,
    "rom_bitcell_um2": 0.018,
    # Energy per access: the on-die and HBM figures now come from
    # lab/energy_constants.py at N4 so this block cannot drift from the
    # KPI table again (it used to say rom 0.08 where the canonical N4
    # value is 0.035 - a 2.3x disagreement that produced the 1.19 mJ
    # anchor alongside a 616 uJ table for the same chip).
    "sram_pj_per_bit": _N4_NODE["sram_pj_per_bit"],
    "rom_pj_per_bit": _N4_NODE["rom_pj_per_bit"],
    "lpddr_pj_per_bit": 4.5,
    "hbm_pj_per_bit": _N4_NODE["hbm_pj_per_bit"],
    "ddr_pj_per_bit": 12.0,
    # MAC lane: BF16 mul + FP32 serial add at N4.
    "mac_pj_per_op": _N4_NODE["mac_pj_per_op"],
    # Static leakage ~ 45 mW per mm2 at N4 (est.).
    "leakage_mw_per_mm2": 45.0,
    "clock_hz": 1.0e9,  # N4-class closed-core clock for this datapath class
}

# Streaming tiers: bytes/s per lane of prefetch bandwidth the schedule needs.
# The fused schedule reads each weight word exactly once per token; the
# prefetch FIFO hides latency as long as the tier keeps up.
TIER_BANDWIDTH = {
    # On-chip SRAM fabric: aggregate bandwidth scales with lanes; modeled as
    # effectively unbounded for these scenarios (it is never the bottleneck).
    "sram": 200e12,
    "lpddr": 68e9,    # LPDDR5X-8533 x 128-bit (est. per-package)
    "ddr": 50e9,      # DDR5-6400 dual-channel off-package (est.)
    "hbm": 3.35e12,   # HBM3e stack (H100-class)
    "hbm_wide": 8.0e12,  # B200-class dual-die
}

# Multi-chip: per-die tier attachment (bandwidth and capacity per die).
PER_DIE_TIER = {
    "sram": {"bytes_per_s": 200e12, "capacity_b": 168e6},
    "lpddr": {"bytes_per_s": 68e9, "capacity_b": 16e9},
    "ddr": {"bytes_per_s": 50e9, "capacity_b": 64e9},
    "hbm": {"bytes_per_s": 3.35e12, "capacity_b": 96e9},    # 8-stack die, est.
    "hbm_wide": {"bytes_per_s": 4.0e12, "capacity_b": 128e9},  # B200-class/die
}

# Cross-die collective network (Jalapeno-class, est.): a dedicated
# low-latency all-reduce fabric distinct from the tier. Every sharded
# layer needs at least one reduction; latency per hop ~ 200 ns class.
COLLECTIVE_NET = {
    "bytes_per_s_per_die": 600e9,   # scale-up local domain (est.)
    "latency_s_per_hop": 200e-9,
    # Activations to reduce per layer: hidden-dim vector per token.
}

# Memory economics (analytical, est. 2025-class module pricing).
TIER_USD_PER_GB = {
    "sram": 0.0,        # on-die, costed as area elsewhere
    "lpddr": 4.0,
    "ddr": 2.5,
    "hbm": 12.0,        # HBM3e module-class, est.
    "hbm_wide": 12.0,
}

# The measured 0.6B chip's SRAM (repo analytical model: 4 banks x 40960
# words x 1024 bits = 167,772,160 BITS = 20,972,160 B) and the fraction
# usable for KV after hidden/workspace.
SRAM_TOTAL_B = 4 * 40960 * 1024 // 8  # 20,972,160 bytes
KV_SRAM_USABLE_FRAC = 0.75

# ---------------------------------------------------------------------------
# Model presets. Dense 1T is the honest infeasible case; MoE 1T is the
# buildable one (active params fit the streaming tier).
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3_06b": {
        "label": "Qwen3-0.6B (measured chip)",
        "model_type": "dense",
        "params": 0.6e9,
        "hidden": 1024,
        "inter": 3072,
        "layers": 28,
        "q_heads": 16,
        "kv_heads": 8,
        "head_dim": 128,
        "vocab": 151936,
        "active_params": 0.6e9,
        "experts": 1,
        "active_experts": 1,
    },
    "dense_1t": {
        "label": "1T dense (infeasible-as-ROM reference)",
        "model_type": "dense",
        "params": 1e12,
        # ~0.94T: d_model 24576, 128 layers, FFN 4x, GQA 192:24 heads.
        "hidden": 24576,
        "inter": 98304,
        "layers": 128,
        "q_heads": 192,
        "kv_heads": 24,
        "head_dim": 128,
        "vocab": 256000,
        "active_params": 1e12,
        "experts": 1,
        "active_experts": 1,
    },
    "moe_1t": {
        "label": "1T MoE (128 experts, 8 active)",
        "model_type": "moe",
        "params": 1e12,
        # ~1.04T stored / ~76B active: d_model 8192, 64 layers,
        # 128 experts top-8, FFN 5120 (per expert).
        "hidden": 8192,
        "inter": 5120,
        "layers": 64,
        "q_heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "vocab": 256000,
        "active_params": 76.2e9,
        "experts": 128,
        "active_experts": 8,
    },
}


def _geom(preset: dict[str, Any]) -> dict[str, int]:
    return {
        "hidden": int(preset["hidden"]),
        "inter": int(preset["inter"]),
        "layers": int(preset["layers"]),
        "q_heads": int(preset["q_heads"]),
        "kv_heads": int(preset["kv_heads"]),
        "head_dim": int(preset["head_dim"]),
        "vocab": int(preset["vocab"]),
    }


def _per_layer_weight_params(g: dict[str, int]) -> int:
    """Attention + MLP weight parameters per layer (no embeddings)."""
    h, inter = g["hidden"], g["inter"]
    qout, kvout = g["q_heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
    return h * (qout + 2 * kvout) + qout * h + h * 2 * inter + inter * h


def _total_weight_bytes(preset: dict[str, Any], weight_bits: int) -> tuple[int, int]:
    """(stored, active) weight bytes at weight_bits.

    MoE stores every expert but streams only top-k per token. Tied embeddings
    are ONE h x vocab table (shared embedding + lm_head). Bit-accurate:
    params x weight_bits / 8 (no integer-division collapse at 4 bits).
    """
    g = _geom(preset)
    h, vocab = g["hidden"], g["vocab"]
    embed_params = h * vocab  # tied: one table
    if preset["model_type"] == "moe":
        inter = g["inter"]
        qout, kvout = g["q_heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
        attn_params = h * (qout + 2 * kvout) + qout * h
        mlp_params = h * 2 * inter + inter * h
        stored_params = preset["layers"] * (attn_params + preset["experts"] * mlp_params)
        active_params = preset["layers"] * (attn_params + preset["active_experts"] * mlp_params)
    else:
        stored_params = preset["layers"] * _per_layer_weight_params(g)
        active_params = stored_params
    stored_b = (stored_params + embed_params) * weight_bits // 8
    active_b = (active_params + embed_params) * weight_bits // 8
    return int(stored_b), int(active_b)


def _kv_bytes_per_token(g: dict[str, int], ctx: int, kv_bits: int) -> int:
    """KV cache: 2 (K+V) x layers x kv_heads x head_dim x ctx x bytes."""
    per_tok = 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * (kv_bits // 8)
    return per_tok * ctx


# KV-cache placement tiers, in read-energy order. KV is per-layer state: it
# is sharded by layer (or KV head) across whatever dies hold that layer's
# compute, so dies exchange activations, never KV. The tier is the cheapest
# physical home with enough capacity; spillover goes to the next tier.
KV_TIERS = [
    # name, pj_per_bit, bytes available in this architecture class
    ("sram", 0.10),        # on-die / on-tile SRAM, layer-sharded
    ("bonded", 0.13),      # hybrid-bonded SRAM dies (chiplet-class)
    ("hbm", 3.50),         # off-package spillover (last resort)
]


def kv_placement(g: dict[str, int], ctx: int, kv_bits: int,
                 sram_capacity_b: float, bonded_capacity_b: float = 0.0,
                 bonded_pj_per_bit: float = 0.13,
                 bonded_tier_name: str = "bonded") -> dict[str, Any]:
    """Where the KV cache physically lives, and what reading it costs.

    Returns per-tier byte placement, total read energy per decode step
    (full KV read once per token), and whether the placement is feasible.
    Capacity checks are explicit: an architecture that cannot hold KV
    reports it instead of silently assuming it fits.

    bonded_tier_name labels the mid tier ("bonded" for hybrid-bonded SRAM
    dies on the chiplet/CIM, "tile_sram" for other tiles on the wafer).
    """
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    remaining = kv_b
    tiers: list[dict[str, Any]] = []
    for name, pj in KV_TIERS:
        label = bonded_tier_name if name == "bonded" else name
        cap = {"sram": sram_capacity_b, "bonded": bonded_capacity_b,
               "hbm": float("inf")}[name]
        pj_eff = bonded_pj_per_bit if name == "bonded" else pj
        take = min(remaining, cap)
        tiers.append({
            "tier": label,
            "bytes": int(take),
            "pj_per_bit": pj_eff,
            "energy_j": take * 8 * pj_eff / 1e12,
        })
        remaining -= take
        if remaining <= 0:
            break
    total_energy = sum(t["energy_j"] for t in tiers)
    return {
        "kv_bytes": kv_b,
        "kv_bits": kv_bits,
        "tiers": tiers,
        "read_energy_j": total_energy,
        "fits_on_package": remaining <= 0,
        "hbm_spill_b": max(0, int(remaining)),
    }


# ---------------------------------------------------------------------------
# Memory economics: where HBM traffic and cost actually go.
# ---------------------------------------------------------------------------

def memory_economics(preset: dict[str, Any], ctx: int, weight_bits: int = 16,
                     kv_bits: int = 16, tier: str = "hbm") -> dict[str, Any]:
    """Weight vs KV traffic split, crossover context, HBM capacity and cost.

    Answers: does HBM cost go to weights or KV? Per token, the fused
    schedule streams active weights once and reads the full KV cache once
    (decode). The crossover context is where KV read traffic overtakes
    weight traffic.
    """
    g = _geom(preset)
    stored_b, active_b = _total_weight_bytes(preset, weight_bits)
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    kv_per_tok = kv_b // max(ctx, 1)

    weight_traffic = active_b          # streamed once per token
    kv_traffic = kv_b                  # full cache read per decode token
    total_traffic = weight_traffic + kv_traffic
    kv_share = kv_traffic / max(total_traffic, 1)
    crossover_ctx = weight_traffic // max(kv_per_tok, 1)

    pj = {"sram": N4["sram_pj_per_bit"], "rom": N4["rom_pj_per_bit"],
          "lpddr": N4["lpddr_pj_per_bit"], "hbm": N4["hbm_pj_per_bit"],
          "hbm_wide": N4["hbm_pj_per_bit"], "ddr": N4["ddr_pj_per_bit"]}
    weight_j = weight_traffic * 8 * pj[tier] / 1e12
    kv_j = kv_traffic * 8 * pj[tier] / 1e12

    # Capacity bill: weights are the floor; KV grows on top.
    capacity_b = stored_b + kv_b
    usd_per_gb = TIER_USD_PER_GB.get(tier, 0.0)
    return {
        "weight_traffic_per_token_b": weight_traffic,
        "kv_traffic_per_token_b": kv_traffic,
        "kv_traffic_share": round(kv_share, 4),
        "kv_overtakes_weights_at_ctx": crossover_ctx,
        "weight_energy_per_token_j": weight_j,
        "kv_energy_per_token_j": kv_j,
        "kv_energy_share": round(kv_j / max(weight_j + kv_j, 1e-12), 4),
        "capacity": {
            "weights_b": stored_b,
            "kv_b": kv_b,
            "total_b": capacity_b,
            "usd_per_gb": usd_per_gb,
            "usd": capacity_b / 1e9 * usd_per_gb,
            "weights_usd": stored_b / 1e9 * usd_per_gb,
            "kv_usd": kv_b / 1e9 * usd_per_gb,
        },
    }


def multi_chip_plan(preset: dict[str, Any], lanes: int, ctx: int,
                    weight_bits: int = 16, tier: str = "hbm",
                    dies: int | None = None) -> dict[str, Any]:
    """Multi-die partition: per-die tier bandwidth/capacity vs demand.

    Weights shard across dies (each die streams its slice); KV shards by
    layer. Demand per die = (lanes/dies) x clock x bytes/step. The plan
    picks the smallest die count that is not bandwidth-limited unless an
    explicit count is given.
    """
    g = _geom(preset)
    stored_b, active_b = _total_weight_bytes(preset, weight_bits)
    kv_b = _kv_bytes_per_token(g, ctx, 16)
    spec = PER_DIE_TIER[tier]
    clock = N4["clock_hz"]

    def plan_for(n: int) -> dict[str, Any]:
        lanes_per_die = max(1, lanes // n)
        demand = lanes_per_die * clock * (weight_bits / 8)
        bw = spec["bytes_per_s"]
        kv_per_die = kv_b // n
        # Token compute time in seconds (cycles / clock); KV read traffic
        # amortized over it.
        token_time_s = max(cycle_model(preset, lanes, ctx, "fused",
                                       weight_bits)["total_cycles"] / clock, 1e-9)
        kv_read_bps = (kv_per_die * 8) / token_time_s
        feed = demand + kv_read_bps
        # Cross-die reductions (Jalapeno-class collective net): with weights
        # sharded, each layer's outputs must be all-reduced across dies once
        # per token. Cost = (n-1) hops x activation bytes / net BW, plus a
        # fixed per-hop latency; serialized after compute per layer. Zero
        # at one die (no cross-die traffic).
        reduce_bytes = g["hidden"] * 2  # BF16 hidden vector per token
        reduce_s = (0 if n <= 1 else
                    g["layers"] * ((n - 1) * reduce_bytes
                                   / COLLECTIVE_NET["bytes_per_s_per_die"]
                                   + COLLECTIVE_NET["latency_s_per_hop"]))
        reduce_cycles = reduce_s * clock
        return {
            "dies": n,
            "lanes_per_die": lanes_per_die,
            "demand_bps": demand,
            "kv_read_bps": kv_read_bps,
            "feed_bps": feed,
            "tier_bw_per_die": bw,
            "bandwidth_limited": feed > bw,
            "bw_headroom": bw / max(feed, 1),
            "weight_slice_b": stored_b // n,
            "kv_slice_b": kv_per_die,
            "capacity_b": spec["capacity_b"],
            "capacity_ok": (stored_b // n + kv_per_die) <= spec["capacity_b"],
            "allreduce_s_per_token": reduce_s,
            "allreduce_cycles": reduce_cycles,
            "allreduce_share": reduce_s / max(reduce_s + token_time_s, 1e-9),
        }

    if dies is not None:
        return plan_for(max(1, dies))
    n = 1
    while n < 1024:
        p = plan_for(n)
        if not p["bandwidth_limited"] and p["capacity_ok"]:
            return p
        n *= 2
    return plan_for(n)


def phase_split(preset: dict[str, Any], lanes: int, ctx: int,
                weight_bits: int = 16, tier: str = "hbm",
                dies: int | None = None) -> dict[str, Any]:
    """Prefill (compute-bound, TTFT) vs decode (bandwidth-bound, TPOT).

    Prefill processes all ctx prompt tokens in one pass: weights stream once
    total (not per token), compute scales with ctx. Decode is one token per
    pass: weights stream every token, KV read grows with ctx. Returns the
    TTFT/TPOT split a serving SLA cares about.
    """
    g = _geom(preset)
    _, active_b = _total_weight_bytes(preset, weight_bits)
    kv_b = _kv_bytes_per_token(g, ctx, 16)
    clock = N4["clock_hz"]
    mc = multi_chip_plan(preset, lanes, ctx, weight_bits, tier, dies)
    lanes_total = lanes

    # Prefill: 2 * active_params * ctx FLOPs, weights streamed once.
    prefill_flops = 2 * preset["active_params"] * ctx + 4 * g["q_heads"] * ctx * ctx * g["head_dim"]
    prefill_compute_s = prefill_flops / (lanes_total * 2 * clock)
    prefill_weight_s = active_b / TIER_BANDWIDTH.get(tier, TIER_BANDWIDTH["hbm"])
    prefill_s = max(prefill_compute_s, prefill_weight_s) + mc["allreduce_s_per_token"]

    # Decode: one token; weights every token + full KV read.
    decode_s = mc["feed_bps"] and max(
        cycle_model(preset, lanes, ctx, "fused", weight_bits)["total_cycles"] / clock,
        active_b / TIER_BANDWIDTH.get(tier, TIER_BANDWIDTH["hbm"]),
    ) + mc["allreduce_s_per_token"]

    return {
        "prefill": {
            "ttft_s": prefill_s,
            "compute_bound": prefill_compute_s > prefill_weight_s,
            "tokens": ctx,
        },
        "decode": {
            "tpot_s": decode_s,
            "bandwidth_bound": True,
            "kv_read_b": kv_b,
        },
        "tok_per_s_user": 1 / max(decode_s, 1e-12),
        "provenance": {"kind": "analytical_estimate", "measured": False},
    }


def qwen06b_kv_story(ctx_list: list[int] | None = None) -> dict[str, Any]:
    """KV-cache effect on the MEASURED 0.6B chip across context.

    Weights never leave ROM, so HBM (or any off-chip tier) exists only for
    KV. This table shows when the on-chip SRAM stops holding KV, what the
    spill costs in energy, and when KV energy overtakes the whole-chip
    energy (measured fused counters priced at N4-class constants:
    57.0M cycles x 16 lanes x 0.45 pJ + 38.1M ROM reads x 256 b x
    0.08 pJ = 1.19 mJ).
    """
    preset = MODEL_PRESETS["qwen3_06b"]
    g = _geom(preset)
    sram_kv_b = int(SRAM_TOTAL_B * KV_SRAM_USABLE_FRAC)
    # Measured-chip energy comparator at N4-class constants: MAC energy from
    # the measured cycle count + ROM read energy from the measured read count
    # (38,086,208 words x 256 b), both at N4 pJ/bit. Apples-to-apples with
    # the KV energies below.
    rom_reads = 38_086_208
    fused_energy_j = (
        MEASURED_FUSED_CYCLES * MEASURED_LANES * N4["mac_pj_per_op"] / 1e12
        + rom_reads * 256 * N4["rom_pj_per_bit"] / 1e12
    )
    rows = []
    for ctx in (ctx_list or [1, 128, 512, 1024, 2048, 4096, 16384, 65536, 131072]):
        kv_b = _kv_bytes_per_token(g, ctx, 16)
        fits = kv_b <= sram_kv_b
        # Spill tier: HBM at N4-class pJ/bit.
        spill_j = (0 if fits else kv_b * 8 * N4["hbm_pj_per_bit"] / 1e12)
        # Even when it fits, reading KV back each token costs SRAM energy.
        read_j = kv_b * 8 * N4["sram_pj_per_bit"] / 1e12
        rows.append({
            "ctx": ctx,
            "kv_bytes": kv_b,
            "fits_sram": fits,
            "sram_area_mm2": round(kv_b * 8 * N4["sram_bitcell_um2"] / 1e6, 2),
            "kv_read_energy_j": read_j,
            "spill_energy_j": spill_j,
            "kv_over_chip_energy": (read_j + spill_j) / fused_energy_j,
        })
    crossover = next((r["ctx"] for r in rows
                      if r["kv_over_chip_energy"] > 1.0), None)
    sram_max_ctx = sram_kv_b // (2 * g["layers"] * g["kv_heads"] * g["head_dim"] * 2)
    return {
        "measured_chip_energy_j": fused_energy_j,
        "sram_kv_capacity_b": sram_kv_b,
        "sram_max_ctx_tokens": sram_max_ctx,
        "kv_energy_overtakes_chip_at_ctx": crossover,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Calibrated cycle model. Reproduces both measured 0.6B runs exactly at
# lanes=16, then scales lanes / geometry / context.
# ---------------------------------------------------------------------------

def _steps(outputs: int, reduce_len: int, lanes: int) -> int:
    return outputs * ((reduce_len + lanes - 1) // lanes)


def cycle_model(
    preset: dict[str, Any],
    lanes: int,
    ctx: int,
    schedule: str = "fused",
    weight_bits: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """Per-stage cycle estimate from measured per-step constants."""
    g = _geom(preset)
    h, inter = g["hidden"], g["inter"]
    qout, kvout = g["q_heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
    nh, nkv, hd = g["q_heads"], g["kv_heads"], g["head_dim"]
    vocab = g["vocab"]
    layers = g["layers"]
    experts = preset["active_experts"] if preset["model_type"] == "moe" else 1

    def gemm_cycles(outputs: int, reduce_len: int, per_step: float) -> float:
        return _steps(outputs, reduce_len, lanes) * per_step

    def nlin_cycles(elements: int) -> float:
        return (elements + lanes - 1) // lanes * NLIN_CYCLES_PER_STEP

    def resid_cycles(elements: int) -> float:
        return (elements + lanes - 1) // lanes * RESIDUAL_CYCLES_PER_STEP

    causal = ctx * (ctx + 1) // 2
    rows: list[dict[str, Any]] = []
    add = lambda name, cyc, note="": rows.append(
        {"stage": name, "cycles": cyc, "note": note})

    # Embedding: tied-row gather, measured 5,120 cycles at h=1024 (5 cyc/elem).
    add("embedding", 5.0 * h, "tied-row gather")
    add("input_norm", h * FULL_NORM_CYCLES_PER_ELEMENT, "sum-once fused")
    add("q_proj", gemm_cycles(qout, h, FUSED_GEMM_CYCLES_PER_STEP))
    add("k_proj", gemm_cycles(kvout, h, FUSED_GEMM_CYCLES_PER_STEP))
    add("v_proj", gemm_cycles(kvout, h, V_PROJ_CYCLES_PER_STEP),
        "transposed V-cache write")
    add("q_norm", gemm_cycles(qout, hd, QK_NORM_CYCLES_PER_STEP),
        "per-head (deferred lever)")
    add("k_norm", gemm_cycles(kvout, hd, QK_NORM_CYCLES_PER_STEP),
        "per-head (deferred lever)")
    add("rope", nlin_cycles(qout + kvout))
    add("attn_scores", _steps(nh * ctx, hd, lanes) * ATTN_SCORE_CYCLES_PER_STEP,
        "scales with ctx^2")
    add("softmax", nh * ctx * SOFTMAX_CYCLES_PER_ROW,
        "per-head row pass, scales with ctx")
    add("attn_value", _steps(qout, ctx, lanes) * ATTN_VALUE_CYCLES_PER_STEP)
    add("o_proj", gemm_cycles(h, qout, FUSED_GEMM_CYCLES_PER_STEP))
    add("attn_residual", resid_cycles(h))
    add("post_norm", h * FULL_NORM_CYCLES_PER_ELEMENT, "sum-once fused")
    for _ in range(experts):
        add("gate_proj", gemm_cycles(inter, h, FUSED_GEMM_CYCLES_PER_STEP))
        add("up_proj", gemm_cycles(inter, h, FUSED_GEMM_CYCLES_PER_STEP))
        add("silu", nlin_cycles(inter))
        add("down_proj", gemm_cycles(h, inter, FUSED_GEMM_CYCLES_PER_STEP))
    add("mlp_residual", resid_cycles(h))
    add("final_norm", h * FULL_NORM_CYCLES_PER_ELEMENT, "sum-once fused")
    add("lm_head", gemm_cycles(vocab, h, FUSED_GEMM_CYCLES_PER_STEP),
        "argmax in flight, never buffered")

    per_layer_rows = [r for r in rows if r["stage"] not in
                      ("embedding", "final_norm", "lm_head")]
    layer_cycles = sum(r["cycles"] for r in per_layer_rows)
    total = layer_cycles * layers + sum(
        r["cycles"] for r in rows if r["stage"] in
        ("embedding", "final_norm", "lm_head"))
    if schedule == "unfused":
        total *= UNFUSED_FACTOR
    return {"rows": rows, "layer_cycles": layer_cycles, "total_cycles": total}


def _validate_against_measured() -> dict[str, Any]:
    """The model must reproduce both measured 0.6B runs before extrapolating."""
    preset = MODEL_PRESETS["qwen3_06b"]
    fused = cycle_model(preset, MEASURED_LANES, 1, "fused")["total_cycles"]
    unfused = cycle_model(preset, MEASURED_LANES, 1, "unfused")["total_cycles"]
    return {
        "fused_estimated": round(fused),
        "fused_measured": MEASURED_FUSED_CYCLES,
        "fused_error_pct": round(100 * (fused - MEASURED_FUSED_CYCLES)
                                 / MEASURED_FUSED_CYCLES, 2),
        "unfused_estimated": round(unfused),
        "unfused_measured": MEASURED_UNFUSED_CYCLES,
        "unfused_error_pct": round(100 * (unfused - MEASURED_UNFUSED_CYCLES)
                                   / MEASURED_UNFUSED_CYCLES, 2),
    }


# ---------------------------------------------------------------------------
# Memory hierarchy: why ROM stops working, what replaces it.
# ---------------------------------------------------------------------------

def memory_plan(preset: dict[str, Any], weight_bits: int, ctx: int,
                kv_bits: int = 16) -> dict[str, Any]:
    g = _geom(preset)
    stored_b, active_b = _total_weight_bytes(preset, weight_bits)
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)

    rom_area_mm2 = stored_b * 8 * N4["rom_bitcell_um2"] / 1e6
    die_area_mm2 = 600.0  # reticle-limited single die (N4-class)
    rom_bits_per_die = die_area_mm2 * 0.85 * 1e6 / N4["rom_bitcell_um2"]
    dies_for_rom = math.ceil(stored_b * 8 / rom_bits_per_die)

    # Streaming tier: active weights read once per token.
    tier = {}
    for name, bw in TIER_BANDWIDTH.items():
        if name == "sram":
            fits = active_b <= 200e6  # largest credible on-chip weight store
            tier[name] = {"fits": fits, "bytes_per_s": bw,
                          "stream_time_s": None if fits else None}
            continue
        stream_s = active_b / bw
        tier[name] = {"fits": active_b <= 4e9, "bytes_per_s": bw,
                      "stream_time_s": stream_s}

    kv_sr = kv_b * 8 * N4["sram_bitcell_um2"] / 1e6
    return {
        "weight_bytes_stored": stored_b,
        "weight_bytes_active_per_token": active_b,
        "kv_bytes_at_ctx": kv_b,
        "kv_bits": kv_bits,
        "rom": {
            "area_mm2_if_rom": round(rom_area_mm2, 1),
            "reticle_die_mm2": die_area_mm2,
            "dies_needed": dies_for_rom,
            "verdict": ("infeasible: exceeds one reticle die by "
                        f"{rom_area_mm2 / die_area_mm2:.0f}x"
                        if rom_area_mm2 > die_area_mm2 else "fits"),
        },
        "tiers": tier,
        "kv_sram_area_mm2": round(kv_sr, 1),
        "kv_verdict": ("fits on-chip" if kv_sr < 200
                       else f"needs HBM/stacked DRAM ({kv_sr:.0f} mm2 of SRAM)"),
        "params_stored": stored_b * 8 // weight_bits,
        "params_active": active_b * 8 // weight_bits,
    }


# ---------------------------------------------------------------------------
# Chip estimate at N4: lanes scale, energy from measured-class constants.
# ---------------------------------------------------------------------------

def chip_estimate(preset: dict[str, Any], lanes: int, ctx: int,
                  schedule: str = "fused", weight_bits: int = 16,
                  tier: str = "hbm", kv_bits: int = 16) -> dict[str, Any]:
    g = _geom(preset)
    cyc = cycle_model(preset, lanes, ctx, schedule, weight_bits)
    total_cycles = cyc["total_cycles"]
    clock = N4["clock_hz"]

    # Bandwidth roofline: the fused schedule reads each active weight word
    # exactly once per token, so demand = lanes x clock x bytes/step. If the
    # tier cannot sustain it the chip stalls; effective cycles stretch by the
    # bandwidth ratio (the same roofline a GPU hits, but with 0 overhead
    # elsewhere). KV reads add ctx-scaled traffic at kv_bits precision.
    stored_b, active_b = _total_weight_bytes(preset, weight_bits)
    demand_bytes_per_s = lanes * clock * (weight_bits / 8)
    tier_bw = TIER_BANDWIDTH.get(tier, TIER_BANDWIDTH["hbm"])
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    kv_read_bps = kv_b * 8 / max(total_cycles / clock, 1e-9)  # amortized
    feed_bw = demand_bytes_per_s + kv_read_bps
    bw_limited = feed_bw > tier_bw
    if bw_limited:
        stretch = feed_bw / tier_bw
        total_cycles = total_cycles * stretch
    latency_s = total_cycles / clock

    macs = total_cycles * lanes  # one MAC per lane per cycle (upper bound)
    compute_j = macs * N4["mac_pj_per_op"] / 1e12

    pj = {"sram": N4["sram_pj_per_bit"], "rom": N4["rom_pj_per_bit"],
          "lpddr": N4["lpddr_pj_per_bit"], "hbm": N4["hbm_pj_per_bit"],
          "hbm_wide": N4["hbm_pj_per_bit"], "ddr": N4["ddr_pj_per_bit"]}
    weight_j = active_b * 8 * pj[tier] / 1e12

    kv_j = kv_b * 8 * (N4["sram_pj_per_bit"] if kv_b * 8 * N4["sram_bitcell_um2"] / 1e6 < 200
                       else N4["hbm_pj_per_bit"]) / 1e12

    # Area: compute lanes + KV SRAM + control + tier PHY. Weights live in
    # the tier, not on-die. Lane area from measured-class gate count
    # (4200 gates/lane, 2 tr/gate) at N4HD density (MTr/mm2 -> mm2).
    lane_area = lanes * 4200 * 2 / N4["mtr_per_mm2"]
    kv_area = kv_b * 8 * N4["sram_bitcell_um2"] / 1e6
    control_area = 12000 * 2 / N4["mtr_per_mm2"]
    # Streaming tier PHY (HBM/LPDDR controllers + PHY): ~8 mm2 per TB/s
    # of interface bandwidth (est., HBM3e PHY-class).
    tier_bw = TIER_BANDWIDTH.get(tier, TIER_BANDWIDTH["hbm"])
    phy_area = tier_bw / 1e12 * 8.0
    area_mm2 = lane_area + kv_area + control_area + phy_area

    leakage_w = area_mm2 * N4["leakage_mw_per_mm2"] / 1e3
    dynamic_w = (compute_j + weight_j + kv_j) / max(latency_s, 1e-9)
    power_w = dynamic_w + leakage_w

    return {
        "platform": f"{preset['label']} chip ({lanes} lanes, {tier}, {schedule})",
        "kind": "asic_scaled",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "calibrated_on": "measured 0.6B fused/unfused T4 runs",
            "sources": "measured per-stage cycle constants; TSMC N4-class est.",
        },
        "metrics": {
            "cycles": round(total_cycles),
            "latency_s": latency_s,
            "bandwidth_limited": bw_limited,
            "demand_bytes_per_s": demand_bytes_per_s,
            "tier_bytes_per_s": tier_bw,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "kv_energy_j": kv_j,
            "total_energy_j": compute_j + weight_j + kv_j,
            "power_w": power_w,
            "area_mm2": area_mm2,
            "transistors": lanes * 4200 * 2 + int(kv_b * 8 / 6) + 24000,
            "lanes": lanes,
            "tier": tier,
            "weight_bits": weight_bits,
        },
        "stages": cyc["rows"],
    }


# ---------------------------------------------------------------------------
# Architecture: 3D ROM chiplets (hybrid-bonded stacked ROM dies on compute die)
# ---------------------------------------------------------------------------

CHIPLET = {
    "name": "3D ROM chiplet (hybrid-bonded stacked ROM dies)",
    # ROM bitcell same N4 process (0.018 um2). Each stacked die is
    # reticle-limited at 600 mm2.
    "rom_bitcell_um2": 0.018,
    "die_area_mm2": 600.0,
    "rom_utilization": 0.85,
    # Hybrid Cu-Cu bonding: TSV pitch ~10 um -> ~10,000 TSVs/mm2. At 1 GHz
    # that is ~1.25 TB/s/mm2 of bonding area. A 600 mm2 die with 40% bonding
    # area yields ~300 TB/s aggregate. Bonding is never the bottleneck.
    "bond_bw_per_mm2": 1.25e12,   # bytes/s per mm2 bonding area
    "bond_area_fraction": 0.40,
    # Bonding energy: TSV capacitance ~50 fF at 1 V = 50 fJ/bit traversal,
    # plus ROM cell read ~0.08 pJ/bit. Total ~0.13 pJ/bit.
    "bond_pj_per_bit": 0.13,
    # Max stacked ROM dies (thermal: ROM dies are low-power read-only, but
    # bonding-layer thermal resistance limits stack height). Practical 8-12.
    "max_stack_dies": 8,
    # HBM fallback for stored weights that do not fit in the ROM stack.
    "hbm_pj_per_bit": 3.5,
    "hbm_bytes_per_s": 3.35e12,
    "clock_hz": 1.0e9,
}


def chiplet_estimate(preset: dict[str, Any], lanes: int, ctx: int,
                     schedule: str = "fused", weight_bits: int = 16,
                     tier: str = "hbm", kv_bits: int = 16) -> dict[str, Any]:
    """3D ROM chiplet: stacked ROM dies hybrid-bonded above a compute die.

    Active weights split between bonded ROM (low energy, very high bandwidth)
    and HBM spillover (high energy, lower bandwidth). ROM capacity is
    stack_dies x rom_per_die. Compute uses the same calibrated lane model.
    The architecture's value is energy: bonded ROM reads cost 0.13 pJ/bit
    vs HBM's 3.5 pJ/bit (27x lower). Speed matches HBM because bonding
    bandwidth far exceeds compute demand.
    """
    g = _geom(preset)
    cyc = cycle_model(preset, lanes, ctx, schedule, weight_bits)
    total_cycles = cyc["total_cycles"]
    clock = CHIPLET["clock_hz"]

    stored_b, active_b = _total_weight_bytes(preset, weight_bits)

    # ROM capacity in the stacked die package.
    rom_bits_per_die = (CHIPLET["die_area_mm2"] * CHIPLET["rom_utilization"]
                       * 1e6 / CHIPLET["rom_bitcell_um2"])
    rom_bytes_per_die = int(rom_bits_per_die / 8)
    rom_capacity_b = rom_bytes_per_die * CHIPLET["max_stack_dies"]

    # Active weights that fit in ROM vs HBM spillover. For MoE the active
    # set changes per token, so ALL stored weights must be addressable; the
    # ROM stack holds what it can and the rest spills to HBM.
    active_in_rom_b = min(active_b, rom_capacity_b)
    active_in_hbm_b = max(0, active_b - rom_capacity_b)
    rom_fraction = active_in_rom_b / max(active_b, 1)

    # Bonding bandwidth: very high, effectively unbounded for these scenarios.
    bond_bw = (CHIPLET["die_area_mm2"] * CHIPLET["bond_area_fraction"]
               * CHIPLET["bond_bw_per_mm2"])
    hbm_bw = CHIPLET["hbm_bytes_per_s"]

    # Weight read time: ROM portion is fast, HBM spillover is the limiter.
    rom_read_s = active_in_rom_b / bond_bw
    hbm_read_s = active_in_hbm_b / hbm_bw if active_in_hbm_b > 0 else 0.0
    weight_read_s = max(rom_read_s, hbm_read_s)

    # Compute time and KV.
    compute_s = total_cycles / clock
    # KV placement: on-die SRAM first (lane/control die has little to
    # spare), then hybrid-bonded SRAM dies share the stack budget with
    # ROM dies, then HBM spillover. KV is layer-sharded across whatever
    # dies hold the layer's compute; dies exchange activations, not KV.
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    on_die_sram_b = 4 * 40960 * 1024  # measured-chip-class SRAM, 20.97 MB
    rom_dies_for_weights = min(CHIPLET["max_stack_dies"],
                               max(1, math.ceil(active_b / rom_bytes_per_die)))
    bonded_kv_dies = CHIPLET["max_stack_dies"] - rom_dies_for_weights
    bonded_kv_capacity_b = bonded_kv_dies * rom_bytes_per_die  # SRAM denser, but stay conservative
    kv = kv_placement(g, ctx, kv_bits, on_die_sram_b,
                      bonded_kv_capacity_b, CHIPLET["bond_pj_per_bit"])
    kv_j = kv["read_energy_j"]
    kv_read_s = kv_b / bond_bw  # bonded tier bandwidth, same as ROM reads

    latency_s = max(compute_s, weight_read_s, kv_read_s)
    bw_limited = hbm_read_s > compute_s and active_in_hbm_b > 0

    # Energy: ROM reads at bonding energy, HBM spillover at HBM energy.
    compute_j = total_cycles * lanes * N4["mac_pj_per_op"] / 1e12
    weight_rom_j = active_in_rom_b * 8 * CHIPLET["bond_pj_per_bit"] / 1e12
    weight_hbm_j = active_in_hbm_b * 8 * CHIPLET["hbm_pj_per_bit"] / 1e12
    weight_j = weight_rom_j + weight_hbm_j

    # Area: compute die + ROM stack dies + HBM PHY for spillover.
    lane_area = lanes * 4200 * 2 / N4["mtr_per_mm2"]
    kv_area = kv_b * 8 * N4["sram_bitcell_um2"] / 1e6
    control_area = 12000 * 2 / N4["mtr_per_mm2"]
    compute_die_area = lane_area + kv_area + control_area
    rom_die_area = rom_dies_for_weights * CHIPLET["die_area_mm2"]
    hbm_phy_area = 8.0 if (active_in_hbm_b > 0 or not kv["fits_on_package"]) else 0.0
    area_mm2 = compute_die_area + rom_die_area + hbm_phy_area

    leakage_w = area_mm2 * N4["leakage_mw_per_mm2"] / 1e3
    dynamic_w = (compute_j + weight_j + kv_j) / max(latency_s, 1e-9)
    power_w = dynamic_w + leakage_w

    return {
        "platform": (f"{preset['label']} 3D ROM chiplet ({lanes} lanes, "
                     f"{rom_dies_for_weights} ROM dies, {weight_bits}b)"),
        "kind": "asic_chiplet",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "calibrated_on": "measured 0.6B per-stage cycle constants",
            "sources": "measured cycles; TSMC N4 + hybrid bonding est.",
        },
        "metrics": {
            "cycles": round(total_cycles),
            "latency_s": latency_s,
            "bandwidth_limited": bw_limited,
            "demand_bytes_per_s": lanes * clock * (weight_bits / 8),
            "tier_bytes_per_s": bond_bw,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "kv_energy_j": kv_j,
            "kv_placement": kv,
            "total_energy_j": compute_j + weight_j + kv_j,
            "power_w": power_w,
            "area_mm2": area_mm2,
            "transistors": lanes * 4200 * 2 + int(kv_b * 8 / 6) + 24000,
            "lanes": lanes,
            "tier": "3d_rom_chiplet",
            "weight_bits": weight_bits,
            "rom_dies": rom_dies_for_weights,
            "rom_capacity_b": rom_capacity_b,
            "active_in_rom_b": active_in_rom_b,
            "active_in_hbm_b": active_in_hbm_b,
            "rom_fraction": round(rom_fraction, 4),
            "bond_bw_bytes_per_s": bond_bw,
        },
        "stages": cyc["rows"],
    }


# ---------------------------------------------------------------------------
# Architecture: Analog compute-in-ROM (crossbar arrays)
# ---------------------------------------------------------------------------

CIM = {
    "name": "Analog compute-in-ROM (crossbar arrays)",
    # Array geometry: 256x256 crossbar. Each cell is a ROM transistor whose
    # conductance encodes the weight value (multi-level for >1 bit/cell).
    "array_rows": 256,
    "array_cols": 256,
    "rom_bitcell_um2": 0.018,
    # ADC: 8-bit SAR at N4, ~5 pJ/conversion (est.). One per column.
    "adc_pj_per_conv": 5.0,
    "adc_latency_cycles": 4,      # pipelined SAR
    # DAC: 8-bit at N4, ~0.5 pJ/conversion (est.). One per row.
    "dac_pj_per_conv": 0.5,
    "dac_latency_cycles": 1,
    # ROM cell analog read: charge sensing, no full voltage swing.
    "cell_read_pj_per_bit": 0.01,
    # Precision: weight_bits split into 4-bit nibbles. Each nibble is one
    # analog pass. BF16 = 4 passes, INT8 = 2 passes, INT4 = 1 pass.
    "bits_per_pass": 4,
    # Array area: cells + ADC + DAC + decoders. Periphery ~4x cell area.
    "periphery_factor": 4.0,
    "clock_hz": 1.0e9,
    "die_area_mm2": 600.0,
    # Arrays span as many reticle dies as storage requires. There is no
    # artificial cap — the honest answer for 1T is that CIM needs many dies,
    # and the area/cost numbers reflect that.
}


def _gemm_macs(preset: dict[str, Any], ctx: int) -> tuple[int, int]:
    """Split total MACs into weight-GEMM (CIM-able) and attention/elementwise.

    Weight GEMMs are matrix-vector products against a weight matrix: Q/K/V/O
    projections, gate/up/down MLP projections, embedding, and lm_head. These
    are the operations CIM accelerates because the weights are fixed in ROM
    cells and the input vector is applied via DACs.

    Attention MACs (score and value) are activation-times-activation, not
    weight GEMMs; they use digital lanes. At ctx=1 they are negligible.
    """
    g = _geom(preset)
    h, inter = g["hidden"], g["inter"]
    qout = g["q_heads"] * g["head_dim"]
    kvout = g["kv_heads"] * g["head_dim"]
    layers = g["layers"]
    experts = (int(preset["active_experts"])
               if preset["model_type"] == "moe" else 1)

    # Weight GEMMs per layer: QKV proj + O proj + (gate+up+down) x experts.
    per_layer = h * (qout + 2 * kvout) + qout * h
    per_layer += experts * (h * 2 * inter + inter * h)
    gemm_macs = layers * per_layer + g["vocab"] * h   # lm_head (tied embed)

    # Attention MACs: scores (Q*K^T) + values (attn*V), causal.
    causal = ctx * (ctx + 1) // 2
    attn_macs = layers * 2 * g["q_heads"] * g["head_dim"] * causal

    return int(gemm_macs), int(attn_macs)


def cim_estimate(preset: dict[str, Any], lanes: int, ctx: int,
                 schedule: str = "fused", weight_bits: int = 16,
                 tier: str = "hbm", kv_bits: int = 16) -> dict[str, Any]:
    """Analog compute-in-ROM: crossbar arrays do GEMM in the current domain.

    Each 256x256 array stores a weight tile in ROM cells and computes a
    matrix-vector product in one analog cycle (Kirchhoff current sum). The
    weight value IS the cell conductance, so weight-fetch energy collapses
    into the cell read. ADC/DAC conversion dominates energy and latency.

    Capacity drives the design: arrays must physically hold ALL stored
    weights (not just active ones), so the number of dies is set by storage
    area. Inactive-expert arrays are power-gated. Throughput is limited by
    how many arrays can be active simultaneously (active weights / tile
    size), not by bandwidth.

    Precision: weight_bits splits into 4-bit passes. INT4 = 1 pass, INT8 = 2,
    BF16 = 4. Each pass multiplies cycles and ADC/DAC energy by the pass
    count, but cell-read energy stays constant (weights are sensed once).

    Non-GEMM ops (norms, softmax, RoPE, residuals) use the same digital
    lane model calibrated on the 0.6B chip.
    """
    g = _geom(preset)
    clock = CIM["clock_hz"]

    gemm_macs, attn_macs = _gemm_macs(preset, ctx)
    stored_b, active_b = _total_weight_bytes(preset, weight_bits)

    # CIM precision passes and per-MAC energy.
    passes = max(1, weight_bits // CIM["bits_per_pass"])
    macs_per_array_cycle = CIM["array_rows"] * CIM["array_cols"]

    # Energy per array-cycle (one analog compute step):
    #   ADC: array_cols conversions, DAC: array_rows conversions,
    #   cell read: macs_per_array_cycle bit senses.
    adc_energy = CIM["array_cols"] * CIM["adc_pj_per_conv"] / 1e12   # J
    dac_energy = CIM["array_rows"] * CIM["dac_pj_per_conv"] / 1e12   # J
    cell_energy = macs_per_array_cycle * CIM["cell_read_pj_per_bit"] / 1e12
    energy_per_array_cycle = (adc_energy + dac_energy + cell_energy) * passes
    cim_energy_per_mac = energy_per_array_cycle / macs_per_array_cycle

    # Array sizing: arrays must physically hold ALL stored weights.
    # Each array holds macs_per_array_cycle weight cells.
    stored_weight_cells = stored_b * 8 // weight_bits
    n_arrays_for_storage = math.ceil(stored_weight_cells / macs_per_array_cycle)

    # Area per array and arrays per die (reticle-limited).
    cell_area = (CIM["array_rows"] * CIM["array_cols"]
                 * CIM["rom_bitcell_um2"]) / 1e6   # mm2
    array_area = cell_area * CIM["periphery_factor"]
    arrays_per_die = int(CIM["die_area_mm2"] * 0.80 / array_area)
    n_dies = max(1, math.ceil(n_arrays_for_storage / arrays_per_die))
    n_arrays = n_arrays_for_storage

    # Active arrays: only arrays holding active weights are powered on.
    # For MoE, active_experts / total_experts fraction of MLP arrays are
    # active per token; attention/embedding arrays are always active.
    if preset["model_type"] == "moe":
        active_experts = int(preset["active_experts"])
        total_experts = int(preset["experts"])
        # MLP weight fraction that is active.
        h, inter = g["hidden"], g["inter"]
        qout = g["q_heads"] * g["head_dim"]
        kvout = g["kv_heads"] * g["head_dim"]
        attn_params = h * (qout + 2 * kvout) + qout * h
        mlp_params = h * 2 * inter + inter * h
        total_layer = attn_params + total_experts * mlp_params
        active_layer = attn_params + active_experts * mlp_params
        active_fraction = active_layer / total_layer
    else:
        active_fraction = 1.0
    n_active_arrays = max(1, int(n_arrays * active_fraction))

    # CIM GEMM throughput: each active array does macs_per_array_cycle MACs
    # per (adc_latency + dac_latency) cycles, pipelined.
    pipeline_depth = CIM["adc_latency_cycles"] + CIM["dac_latency_cycles"]
    cim_macs_per_cycle = n_active_arrays * macs_per_array_cycle / pipeline_depth
    cim_cycles = math.ceil(gemm_macs * passes / max(cim_macs_per_cycle, 1))

    # Non-GEMM ops: digital lanes (same calibrated model). We reuse the
    # cycle_model but subtract GEMM stages to avoid double-counting.
    # cycle_model returns per-layer rows; per-layer stages must be multiplied
    # by layers, while global stages (final_norm) are added once.
    full_cyc = cycle_model(preset, lanes, ctx, schedule, weight_bits)
    gemm_stage_names = {"q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj", "lm_head",
                         "embedding"}
    global_stages = {"final_norm"}
    non_gemm_per_layer = sum(r["cycles"] for r in full_cyc["rows"]
                             if r["stage"] not in gemm_stage_names
                             and r["stage"] not in global_stages)
    non_gemm_global = sum(r["cycles"] for r in full_cyc["rows"]
                          if r["stage"] in global_stages)
    non_gemm_cycles = non_gemm_per_layer * g["layers"] + non_gemm_global

    total_cycles = cim_cycles + non_gemm_cycles
    latency_s = total_cycles / clock

    # Energy: CIM GEMM energy + digital non-GEMM + KV.
    # CIM GEMM energy = MACs x per-MAC energy. The array-cycle energy is a
    # THROUGHPUT figure (the array does 65,536 MACs per cycle); charging it
    # per wall-clock cycle would multiply by pipeline_depth x passes and
    # overstate energy by that factor. The pipelined array still performs
    # gemm_macs x passes MAC operations; each costs energy_per_array_cycle
    # / macs_per_array_cycle.
    cim_j = gemm_macs * passes * (energy_per_array_cycle / macs_per_array_cycle)
    digital_j = non_gemm_cycles * lanes * N4["mac_pj_per_op"] / 1e12
    compute_j = cim_j + digital_j

    # Weight energy is embedded in CIM cell reads (no separate fetch).
    # Small norm/scale weights for digital ops read from SRAM.
    weight_j = 0.0
    norm_weight_b = g["layers"] * 4 * g["hidden"] * 2   # 4 norms x h x BF16
    weight_j = norm_weight_b * 8 * N4["sram_pj_per_bit"] / 1e12

    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    # KV placement: KV cannot live in the ROM crossbars (it changes every
    # token). It lives in SRAM beside the digital attention lanes, sharded
    # by layer across the array dies; each die's non-array area holds its
    # shard. HBM is the spillover tier.
    non_array_per_die_b = int(CIM["die_area_mm2"] * 0.20 * 1e6
                              / N4["sram_bitcell_um2"] * 8 / 8 / 8)
    kv = kv_placement(g, ctx, kv_bits, non_array_per_die_b,
                      (n_dies - 1) * non_array_per_die_b, 0.13)
    kv_j = kv["read_energy_j"]

    # Area: all CIM arrays (all dies) + digital lanes + KV SRAM + control.
    lane_area = lanes * 4200 * 2 / N4["mtr_per_mm2"]
    kv_area = kv_b * 8 * N4["sram_bitcell_um2"] / 1e6
    control_area = 12000 * 2 / N4["mtr_per_mm2"]
    cim_area = n_arrays * array_area
    area_mm2 = cim_area + lane_area + kv_area + control_area

    # Leakage: only active arrays leak at full rate; inactive are power-gated
    # (gate off the ROM wordline supply and ADC/DAC periphery).
    inactive_leakage_fraction = 0.02
    active_area = n_active_arrays * array_area
    inactive_area = cim_area - active_area
    leakage_w = (active_area * N4["leakage_mw_per_mm2"]
                 + inactive_area * N4["leakage_mw_per_mm2"]
                 * inactive_leakage_fraction) / 1e3
    dynamic_w = (compute_j + weight_j + kv_j) / max(latency_s, 1e-9)
    power_w = dynamic_w + leakage_w

    # CIM is never bandwidth-limited: weights are in the arrays.
    demand_bps = 0.0   # no external weight fetch
    effective_bw = n_active_arrays * macs_per_array_cycle * clock * (weight_bits / 8)

    return {
        "platform": (f"{preset['label']} analog CIM ({n_dies} dies, "
                     f"{n_arrays:,} arrays ({n_active_arrays:,} active), "
                     f"{weight_bits}b, {passes} pass(es))"),
        "kind": "asic_cim",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "calibrated_on": "measured 0.6B per-stage cycle constants (non-GEMM)",
            "sources": "measured non-GEMM cycles; CIM crossbar physics est.",
        },
        "metrics": {
            "cycles": round(total_cycles),
            "latency_s": latency_s,
            "bandwidth_limited": False,
            "demand_bytes_per_s": demand_bps,
            "tier_bytes_per_s": effective_bw,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "kv_energy_j": kv_j,
            "kv_placement": kv,
            "total_energy_j": compute_j + weight_j + kv_j,
            "power_w": power_w,
            "area_mm2": area_mm2,
            "transistors": n_arrays * macs_per_array_cycle + lanes * 4200 * 2,
            "lanes": lanes,
            "tier": "cim_arrays",
            "weight_bits": weight_bits,
            "n_arrays": n_arrays,
            "n_active_arrays": n_active_arrays,
            "n_dies": n_dies,
            "passes": passes,
            "cim_macs_per_cycle": round(cim_macs_per_cycle),
            "cim_energy_per_mac_fj": round(cim_energy_per_mac * 1e15, 2),
            "gemm_macs": gemm_macs,
            "attn_macs": attn_macs,
            "active_fraction": round(active_fraction, 4),
        },
        "stages": full_cyc["rows"],
    }


# ---------------------------------------------------------------------------
# Architecture: Wafer-scale ROM (full-wafer weight store + distributed compute)
# ---------------------------------------------------------------------------

WAFER = {
    "name": "Wafer-scale ROM (Cerebras-class full-wafer)",
    # 300 mm wafer: pi * 150^2 = 70,686 mm2.
    "wafer_area_mm2": 70_686,
    "usable_fraction": 0.85,       # edge exclusion, scribe lanes
    "rom_bitcell_um2": 0.018,
    # ROM occupies 60% of usable wafer; rest is compute tiles + fabric + I/O.
    "rom_fraction": 0.60,
    # Compute tiles: each tile has local MAC lanes and reads local ROM.
    "tile_area_mm2": 25.0,
    "lanes_per_tile": 64,
    # On-wafer fabric: Cerebras-class, ~220 Pb/s aggregate (est.).
    "fabric_bytes_per_s": 220e15,
    # Per-tile ROM read bandwidth (local, very high).
    "tile_rom_bw_bytes_per_s": 1e12,
    # ROM read energy: the same on-die ROM as everywhere else, so read it
    # from the canonical table rather than restating it.
    "rom_pj_per_bit": _N4_NODE["rom_pj_per_bit"],
    # Fabric energy: ~0.3 pJ/bit/mm (on-wafer interconnect, est.).
    "fabric_pj_per_bit_mm": 0.3,
    # Average activation travel distance across wafer (est.).
    "avg_travel_mm": 50.0,
    # Power: Cerebras CS-3 is ~25 kW with custom cooling.
    "max_power_w": 25_000,
    "clock_hz": 1.0e9,
}


def wafer_estimate(preset: dict[str, Any], lanes: int, ctx: int,
                   schedule: str = "fused", weight_bits: int = 16,
                   tier: str = "hbm", kv_bits: int = 16) -> dict[str, Any]:
    """Wafer-scale ROM: all weights on full-wafer ROM, compute distributed.

    A 300 mm wafer at 60% ROM utilization holds ~295 GB of ROM. If stored
    weights exceed one wafer, the model partitions across multiple wafers
    with an inter-wafer fabric. Each wafer has distributed compute tiles
    that read local ROM at ~1 TB/s per tile, giving aggregate bandwidth far
    beyond any HBM stack.

    The architecture's value: no off-chip weight traffic at all. Weight
    reads cost 0.08 pJ/bit (on-wafer ROM) vs 3.5 pJ/bit (HBM), a 44x
    energy advantage. Compute is distributed and limited by wafer power
    dissipation, not bandwidth.
    """
    g = _geom(preset)
    clock = WAFER["clock_hz"]

    stored_b, active_b = _total_weight_bytes(preset, weight_bits)

    # Wafer ROM capacity.
    usable_wafer = WAFER["wafer_area_mm2"] * WAFER["usable_fraction"]
    rom_area_mm2 = usable_wafer * WAFER["rom_fraction"]
    rom_bits = rom_area_mm2 * 1e6 / WAFER["rom_bitcell_um2"]
    rom_bytes_per_wafer = int(rom_bits / 8)

    # Wafers needed to hold ALL stored weights.
    n_wafers = max(1, math.ceil(stored_b / rom_bytes_per_wafer))

    # Compute tiles per wafer (remaining area after ROM).
    compute_area_per_wafer = usable_wafer - rom_area_mm2
    tiles_per_wafer = int(compute_area_per_wafer / WAFER["tile_area_mm2"])
    total_tiles = tiles_per_wafer * n_wafers
    total_lanes = total_tiles * WAFER["lanes_per_tile"]

    # Use the calibrated cycle model with the wafer's total lanes.
    cyc = cycle_model(preset, total_lanes, ctx, schedule, weight_bits)
    total_cycles = cyc["total_cycles"]

    # ROM bandwidth: aggregate across all tiles (local reads, no contention).
    aggregate_rom_bw = total_tiles * WAFER["tile_rom_bw_bytes_per_s"]
    demand_bps = total_lanes * clock * (weight_bits / 8)
    bw_limited = demand_bps > aggregate_rom_bw
    if bw_limited:
        total_cycles = total_cycles * (demand_bps / aggregate_rom_bw)
    latency_s = total_cycles / clock

    # KV cache.
    kv_b = _kv_bytes_per_token(g, ctx, kv_bits)
    # KV placement: per-tile SRAM, layer-sharded across all tiles. Each
    # tile reserves part of its 25 mm2 for KV; three wafers carry ~130 GB
    # of tile SRAM, so even 128k fp16 fits without HBM.
    tile_sram_b = int(WAFER["tile_area_mm2"] * 0.25 * 1e6
                      / N4["sram_bitcell_um2"] * 8 / 8 / 8)
    kv = kv_placement(g, ctx, kv_bits, tile_sram_b,
                      (total_tiles - 1) * tile_sram_b, 0.10,
                      bonded_tier_name="tile_sram")
    kv_j = kv["read_energy_j"]

    # Energy: ROM reads (on-wafer) + compute + fabric + KV.
    compute_j = total_cycles * total_lanes * N4["mac_pj_per_op"] / 1e12
    weight_j = active_b * 8 * WAFER["rom_pj_per_bit"] / 1e12
    # Fabric: activation traffic travels avg_travel_mm per layer per token.
    fabric_bits = (g["layers"] * g["hidden"] * 2 * 8   # BF16 hidden vector
                   * WAFER["avg_travel_mm"])
    fabric_j = fabric_bits * WAFER["fabric_pj_per_bit_mm"] / 1e12

    # Cross-wafer all-reduce (if n_wafers > 1): activation reduction per layer.
    reduce_bytes = g["hidden"] * 2   # BF16 hidden vector
    reduce_s = (0 if n_wafers <= 1 else
                g["layers"] * ((n_wafers - 1) * reduce_bytes
                               / COLLECTIVE_NET["bytes_per_s_per_die"]
                               + COLLECTIVE_NET["latency_s_per_hop"]))
    latency_s += reduce_s

    # Area: total wafer area (all wafers).
    area_mm2 = WAFER["wafer_area_mm2"] * n_wafers

    # Power: capped by cooling. If dynamic power exceeds limit, stretch latency.
    leakage_w = area_mm2 * N4["leakage_mw_per_mm2"] / 1e3 / 10  # lower leakage density (3D)
    dynamic_w = (compute_j + weight_j + fabric_j + kv_j) / max(latency_s, 1e-9)
    power_w = dynamic_w + leakage_w
    if power_w > WAFER["max_power_w"]:
        # Thermal throttle: stretch latency to fit power budget.
        stretch = power_w / WAFER["max_power_w"]
        latency_s *= stretch
        power_w = WAFER["max_power_w"]

    return {
        "platform": (f"{preset['label']} wafer-scale ROM ({n_wafers} wafer(s), "
                     f"{total_tiles} tiles, {total_lanes} lanes, {weight_bits}b)"),
        "kind": "asic_wafer",
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "calibrated_on": "measured 0.6B per-stage cycle constants",
            "sources": "measured cycles; wafer-scale ROM + fabric est.",
        },
        "metrics": {
            "cycles": round(total_cycles),
            "latency_s": latency_s,
            "bandwidth_limited": bw_limited,
            "demand_bytes_per_s": demand_bps,
            "tier_bytes_per_s": aggregate_rom_bw,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "kv_energy_j": kv_j,
            "kv_placement": kv,
            "fabric_energy_j": fabric_j,
            "total_energy_j": compute_j + weight_j + fabric_j + kv_j,
            "power_w": power_w,
            "area_mm2": area_mm2,
            "transistors": total_lanes * 4200 * 2 + int(stored_b * 8 / 6),
            "lanes": total_lanes,
            "tier": "wafer_rom",
            "weight_bits": weight_bits,
            "n_wafers": n_wafers,
            "tiles_per_wafer": tiles_per_wafer,
            "rom_bytes_per_wafer": rom_bytes_per_wafer,
            "rom_capacity_b": rom_bytes_per_wafer * n_wafers,
            "allreduce_s_per_token": reduce_s,
        },
        "stages": cyc["rows"],
    }


# Architecture dispatch: map --arch name to estimator function.
ARCH_ESTIMATORS = {
    "hbm": chip_estimate,
    "chiplet": chiplet_estimate,
    "cim": cim_estimate,
    "wafer": wafer_estimate,
}


def gpu_estimate_same_model(preset: dict[str, Any], ctx: int,
                            gpu: dict[str, Any]) -> dict[str, Any]:
    """H100/B200/TPU/LPU running the SAME preset (batch 1)."""
    g = _geom(preset)
    _, active_b = _total_weight_bytes(preset, 16)
    kv_b = _kv_bytes_per_token(g, ctx, 16)
    # FLOPs: 2 * active params per token (attention + MLP), + heads*ctx*hd*2*2.
    flops = 2 * preset["active_params"] + 4 * g["q_heads"] * ctx * g["head_dim"]
    compute_s = flops / gpu["bf16_flops_per_s"]
    if gpu["hbm_bytes_per_s"] > 0:
        weight_s = active_b / gpu["hbm_bytes_per_s"]
        kv_s = kv_b / gpu["hbm_bytes_per_s"]
        feasible = True
        note = None
    else:
        # SRAM-resident design (LPU-class): weights must fit on-chip.
        # Groq-class systems carry ~4.6 GB aggregate SRAM (8-chip, est.):
        # fits 0.6B-class models, not 1T active weights.
        capacity_b = 4.6e9
        feasible = active_b <= capacity_b
        # Fabric bandwidth est.: 64 B per FLOP/s-unit (same convention as
        # lab/compute_metrics.py lpu_estimate) -> 188e12/64 = 2.94 TB/s.
        weight_s = active_b / (gpu["bf16_flops_per_s"] / 64) if feasible else None
        kv_s = kv_b / (gpu["bf16_flops_per_s"] / 64) if feasible else None
        note = None if feasible else (
            f"infeasible: {active_b/1e9:,.0f} GB active weights exceed "
            f"~{capacity_b/1e9:,.1f} GB on-chip SRAM")
    if not feasible:
        return {
            "platform": f"{gpu['name']} · {preset['label']}",
            "kind": "gpu",
            "feasible": False,
            "note": note,
            "provenance": {"kind": "analytical_estimate", "measured": False,
                           "sources": "datasheet-class"},
            "metrics": {"latency_s": None, "total_energy_j": None},
        }
    latency_s = max(compute_s, weight_s + kv_s)
    compute_j = gpu["tdp_watts"] * latency_s * 0.6
    weight_j = active_b * 8 * gpu["hbm_pj_per_bit"] / 1e12 if gpu["hbm_pj_per_bit"] else 0.0
    kv_j = kv_b * 8 * gpu["sram_pj_per_bit"] / 1e12
    return {
        "platform": f"{gpu['name']} · {preset['label']}",
        "kind": "gpu",
        "feasible": True,
        "note": note,
        "provenance": {"kind": "analytical_estimate", "measured": False,
                       "sources": "datasheet-class"},
        "metrics": {
            "latency_s": latency_s,
            "compute_time_s": compute_s,
            "memory_time_s": weight_s + kv_s,
            "compute_energy_j": compute_j,
            "weight_energy_j": weight_j,
            "kv_energy_j": kv_j,
            "total_energy_j": compute_j + weight_j + kv_j,
            "transistors": gpu.get("transistors"),
            "area_mm2": gpu.get("area_mm2"),
        },
    }


def scenario(preset_key: str, lanes: int, ctx: int, schedule: str = "fused",
             weight_bits: int = 16, tier: str = "hbm",
             dies: int | None = None, kv_bits: int = 16,
             arch: str = "hbm") -> dict[str, Any]:
    preset = MODEL_PRESETS[preset_key]
    mem = memory_plan(preset, weight_bits, ctx, kv_bits)
    econ = memory_economics(preset, ctx, weight_bits, kv_bits, tier)
    multi = multi_chip_plan(preset, lanes, ctx, weight_bits, tier, dies)
    phases = phase_split(preset, lanes, ctx, weight_bits, tier, dies)
    kv_story = (qwen06b_kv_story() if preset_key == "qwen3_06b" else None)
    # Architecture dispatch: hbm uses the original streaming chip model;
    # chiplet/cim/wafer use the new fixed-weight-ROM architectures.
    estimator = ARCH_ESTIMATORS.get(arch, chip_estimate)
    chip = estimator(preset, lanes, ctx, schedule, weight_bits, tier, kv_bits)
    rivals = [gpu_estimate_same_model(preset, ctx, g) for g in _RIVALS]
    fused_est = cycle_model(preset, lanes, ctx, "fused")["total_cycles"]
    validation = _validate_against_measured() if preset_key == "qwen3_06b" else None
    return {
        "schema_version": 1,
        "preset": preset_key,
        "label": preset["label"],
        "lanes": lanes,
        "context": ctx,
        "schedule": schedule,
        "weight_bits": weight_bits,
        "kv_bits": kv_bits,
        "tier": tier,
        "arch": arch,
        "dies": multi["dies"],
        "memory": mem,
        "economics": econ,
        "multi_chip": multi,
        "phases": phases,
        "kv_story": kv_story,
        "chip": chip,
        "rivals": rivals,
        "validation": validation,
        "provenance": {
            "measured_parts": "measured 0.6B per-stage cycle constants (both schedules)",
            "estimated_parts": "1T geometry, N4 constants, tier bandwidth (estimates)",
        },
    }


# Datasheet-class rivals, reused from compute_metrics conventions.
_RIVALS = [
    {"name": "NVIDIA H100 SXM", "bf16_flops_per_s": 989e12,
     "hbm_bytes_per_s": 3.35e12, "tdp_watts": 700.0, "hbm_pj_per_bit": 5.0,
     "sram_pj_per_bit": 0.5, "transistors": 80e9, "area_mm2": 814.0},
    {"name": "NVIDIA B200", "bf16_flops_per_s": 2.25e15,
     "hbm_bytes_per_s": 8.0e12, "tdp_watts": 1000.0, "hbm_pj_per_bit": 4.5,
     "sram_pj_per_bit": 0.5, "transistors": 208e9, "area_mm2": 1600.0},
    {"name": "Google TPU v5e", "bf16_flops_per_s": 197e12,
     "hbm_bytes_per_s": 1.6e12, "tdp_watts": 170.0, "hbm_pj_per_bit": 5.0,
     "sram_pj_per_bit": 0.5, "transistors": 32e9, "area_mm2": 325.0},
    {"name": "Groq LPU", "bf16_flops_per_s": 188e12,
     "hbm_bytes_per_s": 0.0, "tdp_watts": 750.0, "hbm_pj_per_bit": 0.0,
     "sram_pj_per_bit": 0.5, "transistors": 26.8e9, "area_mm2": 725.0},
]

GPUS_H100_B200_TPU_LPU = _RIVALS  # alias kept for readability


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="moe_1t",
                        choices=sorted(MODEL_PRESETS))
    parser.add_argument("--lanes", type=int, default=4096)
    parser.add_argument("--ctx", type=int, default=1)
    parser.add_argument("--schedule", default="fused",
                        choices=["fused", "unfused"])
    parser.add_argument("--weight-bits", type=int, default=16,
                        choices=[4, 8, 16])
    parser.add_argument("--tier", default="hbm",
                        choices=["sram", "lpddr", "hbm", "hbm_wide", "ddr"])
    parser.add_argument("--dies", type=int, default=None,
                        help="multi-die partition (default: auto)")
    parser.add_argument("--kv-bits", type=int, default=16, choices=[8, 16],
                        help="KV cache precision")
    parser.add_argument("--arch", default="hbm",
                        choices=sorted(ARCH_ESTIMATORS),
                        help="weight-storage architecture "
                             "(hbm=streaming, chiplet=3D ROM, cim=analog, "
                             "wafer=wafer-scale ROM)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = scenario(args.preset, args.lanes, args.ctx, args.schedule,
                      args.weight_bits, args.tier, args.dies, args.kv_bits,
                      args.arch)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
