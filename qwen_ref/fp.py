"""Bit-defined qwen-ref-v2 floating-point and nonlinear arithmetic."""

import struct
from typing import Iterable

from .fp_tables import EXP_NEG_F32_BITS, RSQRT_F32_BITS


ARITHMETIC_PROFILE = "qwen-ref-v2"
CANONICAL_NAN_BITS = 0x7FC0_0000


def _packed_f32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        sign = struct.unpack("<Q", struct.pack("<d", value))[0] >> 63
        return (sign << 31) | 0x7F80_0000


def f32(value: float) -> float:
    """Round to binary32 RNE, canonicalizing every NaN."""
    bits = _packed_f32_bits(value)
    if bits & 0x7F80_0000 == 0x7F80_0000 and bits & 0x007F_FFFF:
        bits = CANONICAL_NAN_BITS
    return bits_f32(bits)


def f32_bits(value: float) -> int:
    bits = _packed_f32_bits(value)
    if bits & 0x7F80_0000 == 0x7F80_0000 and bits & 0x007F_FFFF:
        return CANONICAL_NAN_BITS
    return bits


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def bf16_bits(value: float) -> int:
    """Convert to BF16 RNE and canonicalize NaNs."""
    bits = f32_bits(value)
    exponent = bits & 0x7F80_0000
    fraction = bits & 0x007F_FFFF
    if exponent == 0x7F80_0000 and fraction:
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bits_bf16(bits: int) -> float:
    return bits_f32((bits & 0xFFFF) << 16)


def bf16(value: float) -> float:
    return bits_bf16(bf16_bits(value))


def f32_add(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def f32_mul(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def _round_ratio_pow2(numerator: int, denominator: int, shift: int) -> int:
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder << 1
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def _finite_parts(bits: int) -> tuple[int, int]:
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x007F_FFFF
    if exponent:
        return 0x0080_0000 | fraction, exponent - 127 - 23
    return fraction, -149


def f32_div(a: float, b: float) -> float:
    """Correctly rounded binary32 division, implemented with integers."""
    a_bits, b_bits = f32_bits(a), f32_bits(b)
    sign = ((a_bits ^ b_bits) >> 31) << 31
    a_abs, b_abs = a_bits & 0x7FFF_FFFF, b_bits & 0x7FFF_FFFF
    a_exp, b_exp = a_abs & 0x7F80_0000, b_abs & 0x7F80_0000
    if (
        (a_exp == 0x7F80_0000 and a_abs & 0x007F_FFFF)
        or (b_exp == 0x7F80_0000 and b_abs & 0x007F_FFFF)
        or (a_abs == 0 and b_abs == 0)
        or (a_abs == 0x7F80_0000 and b_abs == 0x7F80_0000)
    ):
        return bits_f32(CANONICAL_NAN_BITS)
    if a_abs == 0x7F80_0000 or b_abs == 0:
        return bits_f32(sign | 0x7F80_0000)
    if a_abs == 0 or b_abs == 0x7F80_0000:
        return bits_f32(sign)

    a_sig, a_power = _finite_parts(a_abs)
    b_sig, b_power = _finite_parts(b_abs)
    power = a_power - b_power
    exponent = a_sig.bit_length() - b_sig.bit_length() + power
    compare_shift = exponent - power
    if (
        (a_sig < (b_sig << compare_shift))
        if compare_shift >= 0
        else ((a_sig << -compare_shift) < b_sig)
    ):
        exponent -= 1
    if exponent > 127:
        return bits_f32(sign | 0x7F80_0000)
    if exponent >= -126:
        significand = _round_ratio_pow2(
            a_sig, b_sig, power - exponent + 23
        )
        if significand == 0x0100_0000:
            significand >>= 1
            exponent += 1
            if exponent > 127:
                return bits_f32(sign | 0x7F80_0000)
        return bits_f32(sign | ((exponent + 127) << 23) | (significand & 0x007F_FFFF))
    fraction = _round_ratio_pow2(a_sig, b_sig, power + 149)
    if fraction >= 0x0080_0000:
        return bits_f32(sign | 0x0080_0000)
    return bits_f32(sign | fraction)


def _require_positive_finite(value: float, operation: str) -> tuple[float, int]:
    value = f32(value)
    bits = f32_bits(value)
    if not 0 < bits < 0x7F80_0000:
        raise ValueError(f"{operation} requires a positive finite binary32 input")
    return value, bits


def reciprocal_sqrt(value: float) -> float:
    """qwen-ref-v2 piecewise-linear reciprocal square root."""
    _, bits = _require_positive_finite(value, "reciprocal_sqrt")
    raw_exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x007F_FFFF
    if raw_exponent:
        exponent = raw_exponent - 127
        mantissa_fraction = fraction
    else:
        leading = fraction.bit_length() - 1
        exponent = leading - 149
        mantissa_fraction = (fraction << (23 - leading)) & 0x007F_FFFF

    quotient, parity = divmod(exponent, 2)
    if parity:
        table_index = 256 + (mantissa_fraction >> 14)
        remainder = mantissa_fraction & 0x3FFF
    else:
        table_index = mantissa_fraction >> 15
        remainder = mantissa_fraction & 0x7FFF
    interpolation = f32_div(float(remainder), float(1 << (14 if parity else 15)))
    lower = bits_f32(RSQRT_F32_BITS[table_index])
    upper = bits_f32(RSQRT_F32_BITS[table_index + 1])
    estimate = f32_add(lower, f32_mul(f32_add(upper, -lower), interpolation))
    scale_exponent = -quotient
    scale = bits_f32((scale_exponent + 127) << 23)
    return f32_mul(estimate, scale)


def exp_negative(value: float) -> float:
    """Approximate exp(x) for finite x <= 0 using the pinned v2 ROM."""
    value = f32(value)
    bits = f32_bits(value)
    if bits & 0x7F80_0000 == 0x7F80_0000 or value > 0.0:
        raise ValueError("exp_negative requires a finite binary32 input <= 0")
    if value < -16.0:
        return 0.0
    scaled = f32_mul(-value, 256.0)
    index = int(scaled)
    if index == 4096:
        return bits_f32(EXP_NEG_F32_BITS[index])
    fraction = f32_add(scaled, -float(index))
    lower = bits_f32(EXP_NEG_F32_BITS[index])
    upper = bits_f32(EXP_NEG_F32_BITS[index + 1])
    return f32_add(lower, f32_mul(f32_add(upper, -lower), fraction))


def dot_f32_bf16(left: Iterable[float], right: Iterable[float]) -> float:
    """Multiply BF16-rounded operands and round after every FP32 operation."""
    acc = 0.0
    sentinel = object()
    li = iter(left)
    ri = iter(right)
    while True:
        a = next(li, sentinel)
        b = next(ri, sentinel)
        if a is sentinel or b is sentinel:
            if a is not sentinel or b is not sentinel:
                raise ValueError("dot operands have different lengths")
            return acc
        acc = f32_add(acc, f32_mul(bf16(a), bf16(b)))


def pack_bf16(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<H", bf16_bits(value)) for value in values)
