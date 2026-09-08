#!/usr/bin/env python3
"""Fully-fused (no time-multiplexing) constant-coefficient chip scaling.

Stdlib-only (QWEN-PY-001).

This models a DIFFERENT architecture than lab/multi_die.py's near-memory
ROM design: instead of storing weight bits and reading them through a
reused MAC array (time-multiplexed), every weight gets its own
permanently-wired shift-add network (NAF constant-coefficient logic,
lab/mx_format.py), and every network runs simultaneously (spatial, not
time-multiplexed). There is no read step and no ROM bitcell -- the
"weight" is the shape of the gates, not stored data.

Key consequence for area: capacity is NOT bits-per-weight x bitcell
density (that's the ROM-based model's scaling law). It is
adders-per-weight x transistors-per-adder. Two adders exist per weight:

  1. Weight-constant-multiply adders: format-dependent (this is exactly
     lab/mx_format.py's NAF adder count -- 0.375 mean for MXFP4, 1.125
     for MXFP8, computed here for BF16 too).
  2. Accumulation-tree adder: every weight's partial product still has to
     be folded into its output neuron's running sum. In a time-
     multiplexed design this is one shared, reused adder; here, with no
     reuse possible, every weight needs its OWN accumulate-adder instance
     -- a flat, FORMAT-INDEPENDENT +1 adder per weight. This is the
     "activation fan-out and partial-sum accumulation... doesn't
     disappear" cost.

That +1 is why precision doesn't buy the same area win here that it buys
in a bits-stored-in-ROM or bits-moved-over-HBM model: for MXFP4 (0.375
weight-multiply adders) the flat +1 accumulate adder is already ~2.7x
BIGGER than the format-dependent part, so quantizing further barely moves
total area. Contrast with ROM/HBM energy, where going 16->4 bits is a
clean 4x win because there is no accumulate-adder analog in a bits-moved
accounting.

References (reused from elsewhere in this repo, not re-derived):
- lab.scale_1t.N4["mtr_per_mm2"] = 240e6 (N4HD, docs/references.md §3:
  WikiChip Fuse / Schor interpolation, [S]+[E]).
- lab.multi_die.DIE_CAPACITY["N4"]: 600 mm² reticle-limited die, 70%
  usable (existing citation for the ROM design; reused here for the
  fused-logic fabric on the same die-area/utilization assumption).
New citation, not previously used in this repo:
- Full-adder transistor count: 28 T/bit, static CMOS "mirror adder" --
  standard textbook figure (Weste & Harris, "CMOS VLSI Design", 4th ed.).

This is a structural estimate, not a synthesized number -- this repo has
no synthesis/place-and-route flow. It answers "how many chips, roughly,
under a specific cited set of assumptions" -- not "what a real tapeout
would measure."
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.mx_format import naf_adder_count
from lab.multi_die import DIE_CAPACITY
from lab.scale_1t import N4

ROOT = Path(__file__).resolve().parents[1]

TRANSISTORS_PER_ADDER_BIT = 28  # static CMOS full adder ("mirror adder")
ADDER_WIDTH_BITS = 16          # shared activation/accumulate datapath width (stated assumption)
MISC_TRANSISTORS_PER_WEIGHT = 12  # sign XOR + digit-valid muxes; minor vs the adders above

FORMATS = ("bf16", "mxfp8", "mxfp4")


def weight_multiply_adders_mean(fmt: str) -> float:
    """Exhaustive NAF adder count over every representable magnitude in
    the normal (non-subnormal) range -- the range real trained weights
    land in almost exclusively once a block/tensor scale is chosen."""
    if fmt == "mxfp4":
        numerators = (0, 1, 2, 3, 4, 6, 8, 12)          # E2M1, doubled magnitude
    elif fmt == "mxfp8":
        numerators = tuple(8 + m for m in range(8))      # E4M3, 3-bit mantissa
    elif fmt == "bf16":
        numerators = tuple(range(128, 256))               # 7-bit mantissa + implicit 1
    else:
        raise ValueError(f"unknown format: {fmt}")
    counts = [naf_adder_count(n) for n in numerators]
    return sum(counts) / len(counts)


@dataclass(frozen=True)
class FusedFormatCost:
    format: str
    weight_multiply_adders: float
    accumulate_adders: float  # always 1.0 -- format independent
    total_adders: float
    transistors_per_weight: float


def format_cost(fmt: str) -> FusedFormatCost:
    wa = weight_multiply_adders_mean(fmt)
    acc = 1.0
    total = wa + acc
    transistors = (
        total * ADDER_WIDTH_BITS * TRANSISTORS_PER_ADDER_BIT + MISC_TRANSISTORS_PER_WEIGHT
    )
    return FusedFormatCost(fmt, wa, acc, total, transistors)


def die_capacity_billions(fmt: str, tech_node: str = "N4") -> float:
    die = DIE_CAPACITY[tech_node]
    usable_mm2 = die["die_area_mm2"] * die["rom_utilization"]
    usable_transistors = usable_mm2 * N4["mtr_per_mm2"]
    cost = format_cost(fmt)
    return usable_transistors / cost.transistors_per_weight / 1e9


def dies_needed(total_params_billions: float, fmt: str, tech_node: str = "N4") -> int:
    cap = die_capacity_billions(fmt, tech_node)
    return max(1, math.ceil(total_params_billions / cap))


def main() -> int:
    formats_report: dict[str, Any] = {}
    for fmt in FORMATS:
        cost = format_cost(fmt)
        formats_report[fmt] = {
            "weight_multiply_adders_mean": cost.weight_multiply_adders,
            "accumulate_adders_flat": cost.accumulate_adders,
            "total_adders_per_weight": cost.total_adders,
            "transistors_per_weight": cost.transistors_per_weight,
            "die_capacity_billion_params": die_capacity_billions(fmt),
            "accumulate_share_of_area": cost.accumulate_adders / cost.total_adders,
        }

    scenarios_report: dict[str, Any] = {}
    for size_b, label in ((30, "30B_params"), (1000, "1T_params")):
        scenarios_report[label] = {
            fmt: {
                "dies_needed": dies_needed(size_b, fmt),
                "die_capacity_billion_params": die_capacity_billions(fmt),
            }
            for fmt in FORMATS
        }

    report = {
        "method": (
            "Structural estimate: capacity_per_die = usable_transistors / "
            "transistors_per_weight; transistors_per_weight = "
            "(weight_multiply_adders + 1_accumulate_adder) * 16-bit width * "
            "28 T/adder-bit + 12 T misc. No synthesis/P&R flow exists in "
            "this repo -- this is not a measured area."
        ),
        "die_area_mm2": DIE_CAPACITY["N4"]["die_area_mm2"],
        "usable_fraction": DIE_CAPACITY["N4"]["rom_utilization"],
        "mtr_per_mm2": N4["mtr_per_mm2"],
        "transistors_per_adder_bit": TRANSISTORS_PER_ADDER_BIT,
        "adder_width_bits": ADDER_WIDTH_BITS,
        "formats": formats_report,
        "scenarios": scenarios_report,
    }

    out_path = ROOT / "evidence" / "fused_logic_scaling_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Fully-fused constant-coefficient chip: per-weight cost")
    for fmt in FORMATS:
        f = formats_report[fmt]
        print(
            f"  {fmt:6s}: mult_adders={f['weight_multiply_adders_mean']:.3f} "
            f"+ accumulate=1.0 -> total={f['total_adders_per_weight']:.3f}  "
            f"({f['transistors_per_weight']:.0f} T/weight, "
            f"accumulate={f['accumulate_share_of_area']*100:.0f}% of area)  "
            f"capacity={f['die_capacity_billion_params']:.1f}B params/die"
        )
    print()
    for label, row in scenarios_report.items():
        print(f"{label}: " + ", ".join(f"{fmt}={row[fmt]['dies_needed']} dies" for fmt in FORMATS))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
