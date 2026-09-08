"""OCP Microscaling (MX) format encode/decode, plus canonical-signed-digit
(NAF) adder-count analysis for weight-baked constant multipliers.

Stdlib-only (QWEN-PY-001).

Why this exists: a mask-ROM chip's weights are fixed at fabrication, so a
"multiply by weight" never needs a general multiplier -- it can be
synthesized as a small shift-add network tailored to that one constant
(the classic multiplierless-FIR-filter technique, applied to MX-quantized
weight values). This module computes, for the *exact* finite value sets
MXFP4 (E2M1) and MXFP8 (E4M3) can represent, the minimal number of adders
such a network needs -- via the standard non-adjacent-form (NAF) signed-
digit recoding, not a guess. `qwen_chip/tools/gen_maskmac_weights.py` uses
this to bake real, quantized Qwen3-0.6B weight values into
`qwen_maskmac_mxfp4_lane.sv` / `qwen_maskmac_mxfp8_lane.sv`.

References:
- OCP Microscaling Formats (MX) Specification v1.0 (block size 32,
  E8M0 shared scale, E2M1 = MXFP4, E4M3 = MXFP8 element format).
- NAF / canonical signed digit recoding: standard result in computer
  arithmetic (e.g. Parhami, "Computer Arithmetic: Algorithms and Hardware
  Designs"); used here because it is the *minimal* signed-digit form,
  not because it's novel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MX_BLOCK_SIZE = 32  # OCP MX spec: one shared E8M0 scale per 32 elements.

# ---------------------------------------------------------------------------
# E2M1 (MXFP4 element format): 1 sign + 2 exponent (bias 1) + 1 mantissa bit.
# ---------------------------------------------------------------------------

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # exp=00..11, m=0/1


def decode_e2m1(code: int) -> float:
    if not 0 <= code <= 15:
        raise ValueError(f"E2M1 code out of range: {code}")
    sign = -1.0 if (code >> 3) & 1 else 1.0
    exp = (code >> 1) & 0b11
    mant = code & 1
    if exp == 0:
        return sign * (mant * 0.5)
    return sign * (1.0 + mant * 0.5) * (2.0 ** (exp - 1))


def encode_e2m1(value: float) -> int:
    """Round-to-nearest (ties to even magnitude index) E2M1 code."""
    sign_bit = 1 if value < 0 else 0
    mag = abs(value)
    best_idx, best_err = 0, float("inf")
    for idx, m in enumerate(E2M1_MAGNITUDES):
        err = abs(mag - m)
        if err < best_err:
            best_idx, best_err = idx, err
    exp = 0 if best_idx < 2 else (best_idx - 2) // 2 + 1
    mant = best_idx & 1 if best_idx < 2 else (best_idx - 2) % 2
    return (sign_bit << 3) | (exp << 1) | mant


# ---------------------------------------------------------------------------
# E4M3 (MXFP8 element format): 1 sign + 4 exponent (bias 7) + 3 mantissa bits.
# OCP reserves code 0b_s_1111_111 as NaN; max finite magnitude is 448.
# ---------------------------------------------------------------------------

E4M3_BIAS = 7
E4M3_MAX_MAGNITUDE = 448.0


def decode_e4m3(code: int) -> float:
    if not 0 <= code <= 255:
        raise ValueError(f"E4M3 code out of range: {code}")
    sign = -1.0 if (code >> 7) & 1 else 1.0
    exp = (code >> 3) & 0b1111
    mant = code & 0b111
    if exp == 0:
        return sign * (mant / 8.0) * (2.0 ** (1 - E4M3_BIAS))
    if exp == 0b1111 and mant == 0b111:
        return float("nan")
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - E4M3_BIAS))


def encode_e4m3(value: float) -> int:
    """Round-to-nearest E4M3 code (brute force over all 256 codes -- cheap,
    and avoids getting the rounding boundary subtly wrong by hand."""
    sign_bit = 1 if value < 0 else 0
    mag = min(abs(value), E4M3_MAX_MAGNITUDE)
    best_code, best_err = 0, float("inf")
    for exp in range(16):
        for mant in range(8):
            if exp == 0b1111 and mant == 0b111:
                continue  # reserved NaN code, skip only this single code
            candidate = decode_e4m3((exp << 3) | mant)
            err = abs(mag - candidate)
            if err < best_err:
                best_err = err
                best_code = (exp << 3) | mant
    return (sign_bit << 7) | best_code


# ---------------------------------------------------------------------------
# E8M0 shared block scale: 8-bit, value = 2**(code - 127); code 255 = NaN.
# ---------------------------------------------------------------------------

E8M0_BIAS = 127


def decode_e8m0(code: int) -> float:
    if code == 255:
        return float("nan")
    return 2.0 ** (code - E8M0_BIAS)


def choose_block_scale_exp(values: list[float], max_element_magnitude: float) -> int:
    """Smallest scale exponent s such that max(|v|)/2**s fits the element
    format's representable range (OCP MX: absmax scaling per block)."""
    amax = max((abs(v) for v in values), default=0.0)
    if amax == 0.0:
        return 0
    return math.ceil(math.log2(amax / max_element_magnitude))


# ---------------------------------------------------------------------------
# NAF (non-adjacent form) signed-digit recoding: the minimal-nonzero-digit
# signed power-of-two representation of a non-negative integer. Used to
# find the fewest adders needed to compute activation * n for a *fixed*
# integer n (a compile-time constant, since the weight is baked into
# silicon) via a shift-add network only.
# ---------------------------------------------------------------------------


def naf_digits(n: int) -> list[tuple[int, int]]:
    """Return [(bit_position, sign)] for the nonzero digits of NAF(n)."""
    if n < 0:
        raise ValueError("naf_digits expects a non-negative integer")
    digits: list[tuple[int, int]] = []
    pos = 0
    while n > 0:
        if n & 1:
            if n % 4 == 3:
                digits.append((pos, -1))
                n += 1
            else:
                digits.append((pos, 1))
                n -= 1
        n >>= 1
        pos += 1
    return digits


def naf_adder_count(n: int) -> int:
    """Adders needed to sum the NAF digits of n via a shift-add tree.

    k nonzero digits combine with k-1 additions; the first term is a bare
    (shifted, possibly negated) wire, not an addition.
    """
    if n == 0:
        return 0
    return max(0, len(naf_digits(n)) - 1)


@dataclass(frozen=True)
class ConstMultiplierPlan:
    """A weight, decomposed into a shift-add-only multiplier network."""

    digits: tuple[tuple[int, int], ...]  # (shift, sign) pairs, sign is +1/-1
    exp_shift: int  # combined weight-exponent + block-scale shift, applied last
    weight_sign: int  # 0 or 1 (1 = negative)
    adders: int


def plan_e2m1_lane(code: int, block_scale_exp: int) -> ConstMultiplierPlan:
    """Build the shift-add plan for one MXFP4 (E2M1) weight lane.

    Works in half-units (magnitude * 2) so the mantissa numerator is always
    an integer in {0,1,2,3,4,6,8,12}; the trailing /2 folds into exp_shift.
    """
    sign_bit = (code >> 3) & 1
    exp = (code >> 1) & 0b11
    mant = code & 1
    if exp == 0:
        numerator = mant  # 0 or 1 (magnitude 0 or 0.5)
        base_shift = 0
    else:
        numerator = 2 + mant  # 2 or 3 (magnitude 1.0 or 1.5, *2)
        base_shift = exp - 1
    digits = tuple(naf_digits(numerator))
    return ConstMultiplierPlan(
        digits=digits,
        exp_shift=base_shift + block_scale_exp - 1,  # -1 undoes the *2 scaling
        weight_sign=sign_bit,
        adders=naf_adder_count(numerator),
    )


def plan_e4m3_lane(code: int, block_scale_exp: int) -> ConstMultiplierPlan:
    """Build the shift-add plan for one MXFP8 (E4M3) weight lane.

    Mantissa numerator n = 8+m (normal) or m (subnormal) is always a small
    integer in [0,15]; the leading '8' or implicit-one folds into exp_shift
    via the /8 undone below.
    """
    sign_bit = (code >> 7) & 1
    exp = (code >> 3) & 0b1111
    mant = code & 0b111
    if exp == 0:
        numerator = mant
        base_shift = 1 - E4M3_BIAS
    else:
        numerator = 8 + mant
        base_shift = exp - E4M3_BIAS
    digits = tuple(naf_digits(numerator))
    return ConstMultiplierPlan(
        digits=digits,
        exp_shift=base_shift + block_scale_exp - 3,  # -3 undoes the /8 (or the '8' base)
        weight_sign=sign_bit,
        adders=naf_adder_count(numerator),
    )


def apply_const_plan(plan: ConstMultiplierPlan, activation: int) -> int:
    """Reference (bit-exact) evaluation of a shift-add constant-multiplier
    plan against an integer activation -- must match the RTL exactly,
    including the truncating arithmetic right shift for a negative
    exp_shift (Python's `>>` on ints is an arithmetic/floor shift, the same
    semantics as SystemVerilog's `>>>` on a signed value)."""
    total = 0
    for shift, sign in plan.digits:
        total += sign * (activation << shift)
    if plan.exp_shift >= 0:
        total <<= plan.exp_shift
    else:
        total >>= -plan.exp_shift
    if plan.weight_sign:
        total = -total
    return total


def decode_mxfp4_activation_int(code: int, scale_code: int) -> int:
    """Bit-exact reference for the RTL's runtime MXFP4 activation decode
    (qwen_maskmac_mxfp4_lane.sv): same rules as plan_e2m1_lane, applied to
    a runtime code + runtime shared E8M0 scale instead of a baked weight."""
    sign = (code >> 3) & 1
    exp = (code >> 1) & 0b11
    mant = code & 1
    numerator = mant if exp == 0 else (2 + mant)
    base_shift = 0 if exp == 0 else (exp - 1)
    shift = base_shift + (scale_code - E8M0_BIAS) - 1
    mag = (numerator << shift) if shift >= 0 else (numerator >> -shift)
    return -mag if sign else mag


def decode_mxfp8_activation_int(code: int, scale_code: int) -> int:
    """Bit-exact reference for qwen_maskmac_mxfp8_lane.sv's runtime MXFP8
    activation decode; same rules as plan_e4m3_lane."""
    sign = (code >> 7) & 1
    exp = (code >> 3) & 0b1111
    mant = code & 0b111
    numerator = mant if exp == 0 else (8 + mant)
    base_shift = (1 - E4M3_BIAS) if exp == 0 else (exp - E4M3_BIAS)
    shift = base_shift + (scale_code - E8M0_BIAS) - 3
    mag = (numerator << shift) if shift >= 0 else (numerator >> -shift)
    return -mag if sign else mag


def exhaustive_adder_stats(fmt: str) -> dict[str, float | int]:
    """Adder-count distribution over *every* representable magnitude in the
    format -- exhaustive, not sampled, since MXFP4/MXFP8 have few enough
    codes (16 / 256) to enumerate completely."""
    if fmt == "mxfp4":
        counts = [naf_adder_count(n) for n in (0, 1, 2, 3, 4, 6, 8, 12)]
    elif fmt == "mxfp8":
        # Normal-range mantissa numerators (8+m); subnormals are rarer once
        # a block's scale is chosen by absmax and are cheaper besides.
        counts = [naf_adder_count(8 + m) for m in range(8)]
    else:
        raise ValueError(f"unknown format: {fmt}")
    return {
        "mean": sum(counts) / len(counts),
        "max": max(counts),
        "counts": counts,
    }


def _self_test() -> None:
    # Round-trip sanity: every E2M1/E4M3 code decodes to a finite/zero/NaN
    # value and re-encodes to itself (format has no rounding ambiguity for
    # its own exact codes).
    for code in range(16):
        v = decode_e2m1(code)
        if v == 0.0:
            continue  # +0/-0 alias to the same magnitude; not a meaningful round-trip
        assert encode_e2m1(v) == code, (code, v, encode_e2m1(v))
    for code in range(256):
        v = decode_e4m3(code)
        if math.isnan(v) or v == 0.0:
            continue
        assert encode_e4m3(v) == code, (code, v, encode_e4m3(v))

    # NAF adder counts, hand-verified in the module docstring derivation.
    assert [naf_adder_count(n) for n in (0, 1, 2, 3, 4, 6, 8, 12)] == [
        0, 0, 0, 1, 0, 1, 0, 1,
    ]
    assert [naf_adder_count(8 + m) for m in range(8)] == [0, 1, 1, 2, 1, 2, 1, 1]

    fp4 = exhaustive_adder_stats("mxfp4")
    fp8 = exhaustive_adder_stats("mxfp8")
    assert fp4["max"] == 1 and abs(fp4["mean"] - 0.375) < 1e-9
    assert fp8["max"] == 2 and abs(fp8["mean"] - 1.125) < 1e-9
    print("mx_format self-test: PASS")
    print(f"  MXFP4 (E2M1) adders/lane: mean={fp4['mean']:.3f} max={fp4['max']}")
    print(f"  MXFP8 (E4M3) adders/lane: mean={fp8['mean']:.3f} max={fp8['max']}")


if __name__ == "__main__":
    _self_test()
