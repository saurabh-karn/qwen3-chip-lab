"""Memory-mapped access to the canonical flat BF16 ROM."""

from __future__ import annotations

import hashlib
import hmac
import json
import mmap
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path

from .fp import ARITHMETIC_PROFILE, bits_bf16
from .manifest import LAYOUT_SHA256, ROM_BY_HF_NAME, ROM_BYTES, TensorLayout


class ROMTensor(Sequence[float]):
    def __init__(self, owner: "FlatBF16ROM", layout: TensorLayout) -> None:
        self.owner = owner
        self.layout = layout
        self.shape = layout.shape
        self.size = layout.byte_length // 2

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        bits = struct.unpack_from(
            "<H", self.owner.mapping, self.layout.byte_offset + index * 2
        )[0]
        return bits_bf16(bits)

    def __iter__(self) -> Iterator[float]:
        for index in range(self.size):
            yield self[index]  # type: ignore[misc]


class FlatBF16ROM:
    """Weight source implementing direct HF-name-to-ROM-offset lookup."""

    def __init__(
        self, path: str | Path, manifest_path: str | Path | None = None
    ) -> None:
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            size = self.path.stat().st_size
            if size != ROM_BYTES:
                raise ValueError(f"ROM is {size} bytes; expected {ROM_BYTES}")
            if manifest_path is not None:
                self._verify_manifest(Path(manifest_path))
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.file.close()
            raise

    def _verify_manifest(self, path: Path) -> None:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ROM manifest: {path}") from exc
        if not isinstance(document, dict):
            raise ValueError("ROM manifest must be an object")
        expected = {
            "format": "qwen_ref.flat-bf16-rom.v1",
            "dtype": "BF16",
            "endianness": "little",
            "arithmetic_profile": ARITHMETIC_PROFILE,
            "layout_sha256": LAYOUT_SHA256,
            "rom_bytes": ROM_BYTES,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"ROM manifest has invalid {key}")
        expected_digest = document.get("rom_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("ROM manifest has invalid rom_sha256")
        digest = hashlib.sha256()
        self.file.seek(0)
        for chunk in iter(lambda: self.file.read(1024 * 1024), b""):
            digest.update(chunk)
        self.file.seek(0)
        if not hmac.compare_digest(digest.hexdigest(), expected_digest.lower()):
            raise ValueError("ROM SHA-256 does not match its manifest")

    def __getitem__(self, hf_name: str) -> ROMTensor:
        return ROMTensor(self, ROM_BY_HF_NAME[hf_name])

    def close(self) -> None:
        self.mapping.close()
        self.file.close()

    def __enter__(self) -> "FlatBF16ROM":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
