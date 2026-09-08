"""Single source of truth for per-node energy constants.

Every module that turns a measured counter into joules imports from here.
Before this module existed the repo carried three independent lineages and
the same physical quantity had four different values:

    ROM read energy (pJ/bit)
      0.8    lab/compute_metrics.py ASIC        (labelled "28nm-class")
      0.8    lab/machine_timelines.py            (sourced from the above)
      0.08   lab/scale_1t.py N4 + lab/hw_energy  (the "1.19 mJ" anchor)
      0.035  lab/exec_models.py TECH_NODES N4    (the "616 uJ" KPI table)

Those disagreements were invisible in the UI but showed up as the same chip
being quoted at 616 uJ, 1.19 mJ and 1.71 mJ per token on different slides,
and as "this chip vs H100" ranging from 116x to 320x.

The table below is the exec_models lineage, kept because it is the only one
with a documented per-node derivation and published references. Consumers
now differ only in WHICH NODE they ask for, never in the value of a node.

Choosing a node is a claim about the silicon:
  - "28nm" is the lab die that the measured RTL counters came from.
  - "N4" is the production projection.
An energy figure is only "measured" in the sense that the COUNTERS are
measured; the joules are always counters x these analytical constants.

METHODOLOGY (unchanged from the original exec_models table):
  - mac_pj_per_op: BF16 MAC for standard digital logic (not CIM). Derived
    from the Horowitz 45nm baseline (1.5 pJ) scaled ~0.7x per node.
  - rom_pj_per_bit: scaled from SRAM by ~0.6x (simpler read path, no write
    circuitry).
  - sram_pj_per_bit: Horowitz 8KB SRAM = 10 pJ / 64Kbit ~ 0.15 pJ/bit at
    45nm, scaled ~0.65x per node (empirical ISSCC trend).
  - hbm_pj_per_bit: Horowitz ~6 pJ/bit at 45nm; HBM3e improves to
    ~3.5-5 pJ/bit (package-level, I/O dominated, scales slowly).
  - logic_density: TSMC published relative transistor densities.

These are representative estimates for SYSTEM-LEVEL modelling, not
transistor-level precision.
"""

from __future__ import annotations

from typing import Any

TECH_NODES: dict[str, dict[str, Any]] = {
    "28nm": {
        "name": "28nm (lab/educational)",
        # Horowitz 2014 scaled from 45nm by ~0.8x
        "mac_pj_per_op": 1.2,       # BF16: ~1.5 pJ x 0.8 (from 45nm baseline)
        "rom_pj_per_bit": 0.10,     # SRAM x 0.6 (simpler read, no write circuitry)
        "sram_pj_per_bit": 0.15,    # Horowitz: 10 pJ / 64Kb ~ 0.15 pJ/bit
        "hbm_pj_per_bit": 5.0,      # Horowitz: ~6 pJ/bit, improved slightly
        "logic_density": 1.0,       # Reference baseline
        "ref": "Horowitz ISSCC 2014, scaled from 45nm",
    },
    "N7": {
        "name": "TSMC N7 (7nm)",
        # ~0.5x from 28nm (2 node generations)
        "mac_pj_per_op": 0.6,       # 1.2 x 0.5
        "rom_pj_per_bit": 0.05,     # 0.10 x 0.5
        "sram_pj_per_bit": 0.08,    # 0.15 x 0.5 (rounded)
        "hbm_pj_per_bit": 4.5,      # I/O-dominated, scales slowly
        "logic_density": 3.0,       # TSMC: ~3x vs 28nm
        "ref": "Scaled from 28nm; TSMC density from public roadmap",
    },
    "N5": {
        "name": "TSMC N5 (5nm)",
        # ~0.4x from 28nm; validated against ISSCC22-5nm CIM range
        "mac_pj_per_op": 0.5,       # 1.2 x 0.4
        "rom_pj_per_bit": 0.04,     # 0.10 x 0.4
        "sram_pj_per_bit": 0.06,    # 0.15 x 0.4
        "hbm_pj_per_bit": 4.0,      # HBM3 improvement
        "logic_density": 5.0,       # TSMC: ~5x vs 28nm
        "ref": "Scaled from 28nm; cross-checked with ISSCC 2022 5nm CIM",
    },
    "N4": {
        "name": "TSMC N4 (4nm production)",
        # N4 is N5 enhanced; ~0.35x from 28nm
        "mac_pj_per_op": 0.45,      # 1.2 x 0.375
        "rom_pj_per_bit": 0.035,    # 0.10 x 0.35
        "sram_pj_per_bit": 0.05,    # 0.15 x 0.35
        "hbm_pj_per_bit": 3.5,      # HBM3e
        "logic_density": 6.0,       # TSMC: ~6x vs 28nm (N4 ≈ N5P)
        "ref": "Scaled from 28nm; TSMC N4 ≈ N5 enhanced",
    },
    "N3": {
        "name": "TSMC N3 (3nm)",
        # ~0.3x from 28nm; ISSCC24 3nm validates ballpark
        "mac_pj_per_op": 0.35,      # 1.2 x 0.29
        "rom_pj_per_bit": 0.03,     # 0.10 x 0.3
        "sram_pj_per_bit": 0.04,    # 0.15 x 0.3 (SRAM scaling stalls at N3)
        "hbm_pj_per_bit": 3.0,      # HBM3e, marginal improvement
        "logic_density": 8.0,       # TSMC: ~8x vs 28nm
        "ref": "Scaled from 28nm; cross-checked with ISSCC 2024 3nm neural engine",
    },
}

DEFAULT_TECH_NODE = "N4"  # Production-class default

#: The node the measured RTL counters were actually produced on. Energy quoted
#: at any other node is a projection, and callers should say so.
MEASURED_TECH_NODE = "28nm"

#: Width of one mask-ROM read on the datapath (16 lanes x BF16).
ROM_WORD_BITS = 256


def node(name: str | None = None) -> dict[str, Any]:
    """Return the constant block for `name`, falling back to the default."""
    return TECH_NODES.get(name or DEFAULT_TECH_NODE, TECH_NODES[DEFAULT_TECH_NODE])


def energy_label(name: str | None = None) -> str:
    """Human-readable provenance string for an energy figure at `name`."""
    t = node(name)
    return (
        f"{t['name']} ({t['mac_pj_per_op']} pJ/MAC, "
        f"{t['rom_pj_per_bit']} pJ/bit ROM, {t['sram_pj_per_bit']} pJ/bit SRAM)"
    )
