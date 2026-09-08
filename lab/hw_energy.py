"""Energy from real RTL counters, using only already-cited constants.

Every backend's measured counters (rom_read_count, sram_read_count,
mac_count from rtl_meta) are converted to energy with the N4-class
constants from lab/scale_1t.py (the same constants behind the 1.19 mJ
fused-anchor):

    rom_pj_per_bit  = 0.08   on-die mask-ROM read
    sram_pj_per_bit = 0.10   on-die SRAM read
    hbm_pj_per_bit  = 3.5    HBM3e weight stream
    mac_pj_per_op   = 0.45   BF16 MAC lane op

No new constants are introduced here. The point: the deck's claim
("eliminating HBM for fixed weights gives better model economics") stops
being an analytical estimate and becomes arithmetic over measured RTL
counters.

Memory-scenario semantics:
  - memory="rom" backends (mask_rom, mask_rom_fused) read weights from the
    on-die ROM: rom_read_count x 256b at 0.08 pJ/bit.
  - memory="hbm" backends (gpu_hbm/tpu_hbm/lpu_hbm scenarios) pull the
    same weight words from an HBM-class store: rom_read_count x 256b at
    3.5 pJ/bit. The canonical gpu/tpu/lpu rows are the engines measured
    against the direct ROM port; their HBM-scenario twins carry the
    HBM-priced weight stream.
"""

from __future__ import annotations

from typing import Any

from lab.energy_constants import (
    ROM_WORD_BITS,
    node as energy_node,
)

# One canonical table for every module (lab/energy_constants.py), N4 block.
_N4 = energy_node("N4")
PJ_PER_BIT = {
    "rom": _N4["rom_pj_per_bit"],
    "sram": _N4["sram_pj_per_bit"],
    "hbm": _N4["hbm_pj_per_bit"],
}
MAC_PJ_PER_OP = _N4["mac_pj_per_op"]

PROVENANCE = (
    "lab/scale_1t.py N4 constants (rom 0.08 / sram 0.10 / hbm 3.5 pJ/bit, "
    "mac 0.45 pJ/op); counters from rtl_meta (measured)"
)


def energy_row(result: dict[str, Any]) -> dict[str, Any] | None:
    """Energy breakdown (Joules) for one backend result row, or None."""
    meta = result.get("rtl_meta") or {}
    rom_reads = int(meta.get("rom_read_count") or 0)
    sram_reads = int(meta.get("sram_read_count") or 0)
    macs = int(meta.get("mac_count") or 0)
    if not (rom_reads or sram_reads or macs):
        return None

    memory = result.get("memory", "rom")
    weight_pj_per_bit = PJ_PER_BIT["hbm"] if memory == "hbm" else PJ_PER_BIT["rom"]

    weight_j = rom_reads * ROM_WORD_BITS * weight_pj_per_bit / 1e12
    sram_j = sram_reads * ROM_WORD_BITS * PJ_PER_BIT["sram"] / 1e12
    mac_j = macs * MAC_PJ_PER_OP / 1e12
    total_j = weight_j + sram_j + mac_j

    return {
        "memory": memory,
        "weight_energy_j": weight_j,
        "sram_energy_j": sram_j,
        "mac_energy_j": mac_j,
        "total_energy_j": total_j,
        "weight_pj_per_bit": weight_pj_per_bit,
        "constants": {
            "rom_pj_per_bit": PJ_PER_BIT["rom"],
            "sram_pj_per_bit": PJ_PER_BIT["sram"],
            "hbm_pj_per_bit": PJ_PER_BIT["hbm"],
            "mac_pj_per_op": MAC_PJ_PER_OP,
        },
        "provenance": PROVENANCE,
    }
