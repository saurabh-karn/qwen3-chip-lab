"""Canonical Qwen3-0.6B geometry.

These values are intentionally not configurable.  Small tests should exercise
the shape-inferred operators rather than constructing a non-canonical model.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Geometry:
    layers: int = 28
    hidden_size: int = 1024
    intermediate_size: int = 3072
    attention_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151_936
    max_position_embeddings: int = 40_960
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 1_000_000.0

    def __post_init__(self) -> None:
        expected = {
            "layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "attention_heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 151_936,
            "max_position_embeddings": 40_960,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1_000_000.0,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"canonical geometry requires {field}={value}")

    @property
    def q_size(self) -> int:
        return self.attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.attention_heads // self.kv_heads


QWEN3_0_6B = Geometry()
