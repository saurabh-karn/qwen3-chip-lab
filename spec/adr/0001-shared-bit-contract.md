# ADR 0001: Shared bit contract and pluggable physical architecture

Status: accepted

## Context

The Python implementation is an oracle for chip exploration, while the RTL
must support different ROM, SRAM, and compute organizations. Comparing a
framework float result to an unspecified hardware approximation cannot
identify whether an architecture is correct.

## Decision

Model geometry, ROM layout, operation order, arithmetic profile, and mandatory
checkpoints are invariant. Python and RTL consume the same versioned
known-answer vectors. ROM, SRAM, and compute implementations connect through
stable request/response interfaces and may be replaced independently.

Native BF16 weights are constants. Products use BF16 operands and
`qwen-ref-v2` uses a serial binary32 accumulator with no fused multiply-add.
Scratch-persisted activations and KV values are BF16; operator checkpoints are
binary32 pre-commit values. Subnormals are preserved, rounding is
round-to-nearest ties-to-even, and NaNs are canonicalized.

`qwen-ref-v2` replaces runtime platform math with generated, hash-pinned
binary32 ROMs: linearly interpolated reciprocal square root and negative
exponential tables, plus direct RoPE cosine/sine values for the canonical 128
positions and 64 rotate-half pairs. Its complete domain, exceptional-value,
interpolation, and per-operation rounding rules are normative in
`spec/arithmetic/qwen-ref-v2/`.

The contract is authoritative over incomplete RTL. The current MAC's
flush-to-zero and exceptional-value behavior is explicitly non-compliant and
must be corrected during the RTL arithmetic phase rather than reflected in a
weaker profile.

## Consequences

- A faster design is rejected if it changes mandatory checkpoint values.
- Different reduction trees require distinct arithmetic profile identifiers.
- Technology-neutral area and energy values are estimates until characterized
  macro and standard-cell data are supplied.
- Simulation ROM initialization is allowed, but a runtime host weight-write
  interface is prohibited.
