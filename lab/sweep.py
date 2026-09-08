#!/usr/bin/env python3
"""Dependency-free analytical Qwen chip configuration comparison."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate(model: dict[str, Any], chip: dict[str, Any], seq: int) -> dict[str, Any]:
    h = int(model["hidden_size"])
    inter = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    nh = int(model["num_attention_heads"])
    nkv = int(model["num_key_value_heads"])
    hd = int(model["head_dim"])
    vocab = int(model["vocab_size"])
    qout, kvout = nh * hd, nkv * hd

    projections_per_token = h * (qout + 2 * kvout) + h * qout + 3 * h * inter
    projection_macs = layers * seq * projections_per_token
    attention_macs = layers * 2 * nh * hd * seq * (seq + 1) // 2
    head_macs = vocab * h
    total_macs = projection_macs + attention_macs + head_macs

    compute = chip["compute"]
    rom = chip["rom"]
    sram = chip["sram"]
    lanes = int(compute["mac_lanes"])
    pipeline = int(compute["pipeline_depth"])
    compute_cycles = (total_macs + lanes - 1) // lanes + pipeline

    # Each projection MAC consumes one BF16 ROM word. Attention MACs use SRAM.
    rom_words = projection_macs + head_macs + vocab * h + h
    rom_words_per_cycle = (
        int(rom["banks"])
        * int(rom["ports_per_bank"])
        * int(rom["read_width_bits"])
        // 16
    )
    rom_cycles = (rom_words + rom_words_per_cycle - 1) // rom_words_per_cycle
    sram_words = attention_macs * 2 + layers * seq * (12 * h + 4 * inter)
    sram_words_per_cycle = int(sram["banks"]) * int(sram["ports_per_bank"])
    sram_cycles = (sram_words + sram_words_per_cycle - 1) // sram_words_per_cycle
    cycles = max(compute_cycles, rom_cycles, sram_cycles)
    clock_hz = int(chip["clock_hz"])

    # Technology-neutral normalized estimates. They are intentionally not
    # presented as square millimetres or joules without a PDK characterization.
    compute_area_units = lanes * (42 if compute["accumulator"] == "fp32" else 24)
    rom_area_units = rom_words * 16 / max(1, int(rom["banks"])) ** 0.08
    sram_capacity_words = 2 * seq * h + seq * (qout + 2 * kvout + 2 * inter)
    sram_area_units = sram_capacity_words * int(sram["word_width_bits"]) * 0.45
    energy_units = total_macs + rom_words * 0.35 + sram_words * 0.18

    return {
        "schema_version": 1,
        "name": chip["name"],
        "correctness": "not_run",
        "sequence_length": seq,
        "metrics": {
            "total_macs": total_macs,
            "projection_macs": projection_macs,
            "attention_macs": attention_macs,
            "head_macs": head_macs,
            "cycles": cycles,
            "compute_cycles": compute_cycles,
            "rom_cycles": rom_cycles,
            "sram_cycles": sram_cycles,
            "latency_seconds": cycles / clock_hz,
            "ideal_mac_utilization": total_macs / max(1, cycles * lanes),
            "rom_words_read": rom_words,
            "sram_words_accessed": sram_words,
            "sram_capacity_words": sram_capacity_words,
            "compute_area_units": compute_area_units,
            "rom_area_units": rom_area_units,
            "sram_area_units": sram_area_units,
            "energy_units": energy_units,
        },
        "provenance": {
            "kind": "analytical_estimate",
            "measured": False,
            "warning": "Normalized estimates require macro/PDK characterization before physical conclusions.",
        },
        "configuration": chip,
    }


def variants(base: dict[str, Any], lanes: Iterable[int], rom_banks: Iterable[int],
             sram_banks: Iterable[int]) -> Iterable[dict[str, Any]]:
    for lane_count, rb, sb in itertools.product(lanes, rom_banks, sram_banks):
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"mac{lane_count}_rom{rb}_sram{sb}"
        cfg["compute"]["mac_lanes"] = lane_count
        cfg["rom"]["banks"] = rb
        cfg["sram"]["banks"] = sb
        yield cfg


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    am, bm = a["metrics"], b["metrics"]
    keys = ("cycles", "compute_area_units", "rom_area_units", "sram_area_units", "energy_units")
    return all(am[k] <= bm[k] for k in keys) and any(am[k] < bm[k] for k in keys)


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto"] = not any(other is not row and dominates(other, row) for other in rows)


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "configs/qwen3_0_6b.json"))
    parser.add_argument("--base", default=str(ROOT / "configs/chip_default.json"))
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--mac-lanes", default="64,128,256,512")
    parser.add_argument("--rom-banks", default="8,16,32")
    parser.add_argument("--sram-banks", default="8,16,32")
    parser.add_argument("--output", default=str(ROOT / "evidence/sweep.json"))
    args = parser.parse_args()
    if args.seq < 1:
        parser.error("--seq must be positive")
    model, base = load_json(args.model), load_json(args.base)
    rows = [
        estimate(model, cfg, args.seq)
        for cfg in variants(
            base,
            parse_ints(args.mac_lanes),
            parse_ints(args.rom_banks),
            parse_ints(args.sram_banks),
        )
    ]
    mark_pareto(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "designs": rows}, indent=2) + "\n")
    print(f"wrote {len(rows)} designs ({sum(r['pareto'] for r in rows)} Pareto) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
