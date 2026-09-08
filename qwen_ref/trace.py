"""Stable NDJSON tracing and tensor checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .fp import ARITHMETIC_PROFILE, bf16_bits, f32

Progress = Callable[[int, int, str], None]


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _flatten(values: object):
    if _is_sequence(values):
        for value in values:  # type: ignore[union-attr]
            yield from _flatten(value)
    else:
        yield float(values)  # type: ignore[arg-type]


def _shape(values: object) -> list[int]:
    if not _is_sequence(values):
        return []
    length = len(values)  # type: ignore[arg-type]
    if not length:
        return [0]
    child = _shape(values[0])  # type: ignore[index]
    for value in values:  # type: ignore[union-attr]
        if _shape(value) != child:
            raise ValueError("trace tensors must be rectangular")
    return [length, *child]


def tensor_digest(values: object, dtype: str = "F32") -> str:
    digest = hashlib.sha256()
    for value in _flatten(values):
        if dtype == "BF16":
            digest.update(struct.pack("<H", bf16_bits(value)))
        elif dtype == "F32":
            digest.update(struct.pack("<f", f32(value)))
        else:
            raise ValueError("trace dtype must be BF16 or F32")
    return digest.hexdigest()


class TraceWriter:
    def __init__(
        self, destination: str | Path | TextIO, checkpoint_dir: str | Path | None = None
    ) -> None:
        self._owned = not hasattr(destination, "write")
        self.stream = (
            Path(destination).open("w", encoding="utf-8")
            if self._owned
            else destination
        )
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def event(
        self, name: str, values: object, **fields: object
    ) -> None:
        shape = _shape(values)
        flattened = _flatten(values)
        sample = []
        for _, value in zip(range(32), flattened):
            sample.append(f32(value))
        record = {
            "schema_version": 1,
            "arithmetic_profile": ARITHMETIC_PROFILE,
            "event": name,
            "stage": name.rsplit(".", 1)[-1],
            "dtype": "F32",
            "shape": shape,
            "elements": math.prod(shape) if shape else 1,
            "f32_sha256": tensor_digest(values),
            "values": sample,
            **fields,
        }
        self.stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        self.stream.flush()
        if self.checkpoint_dir:
            safe_name = name.replace("/", "_").replace(".", "_")
            path = self.checkpoint_dir / f"{safe_name}.f32le"
            with path.open("wb") as output:
                for value in _flatten(values):
                    output.write(struct.pack("<f", f32(value)))

    def close(self) -> None:
        if self._owned:
            self.stream.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
