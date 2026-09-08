"""Standard-library-only Qwen3-0.6B reference and RTL oracle."""

from .config import Geometry, QWEN3_0_6B
from .fp import ARITHMETIC_PROFILE, bf16, bf16_bits, bits_bf16, f32
from .model import causal_gqa, forward, linear, qk_norm, rms_norm, rope, silu, swiglu
from .rom import FlatBF16ROM
from .safetensors import SafeTensorFile, ShardedSafeTensors, open_checkpoint

__all__ = [
    "Geometry",
    "ARITHMETIC_PROFILE",
    "FlatBF16ROM",
    "QWEN3_0_6B",
    "SafeTensorFile",
    "ShardedSafeTensors",
    "bf16",
    "bf16_bits",
    "bits_bf16",
    "causal_gqa",
    "f32",
    "forward",
    "linear",
    "open_checkpoint",
    "qk_norm",
    "rms_norm",
    "rope",
    "silu",
    "swiglu",
]
