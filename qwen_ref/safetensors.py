"""Minimal, read-only safetensors support using only the standard library."""

from __future__ import annotations

import json
import mmap
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


_ITEM_SIZE = {"BF16": 2, "F32": 4}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class SafeTensor(Sequence[float]):
    """A zero-copy tensor view whose owner must remain open."""

    def __init__(
        self, owner: "SafeTensorFile", name: str, dtype: str,
        shape: tuple[int, ...], start: int, end: int,
    ) -> None:
        self.owner = owner
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.start = start
        self.end = end
        size = 1
        for extent in shape:
            size *= extent
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        pos = self.start + index * _ITEM_SIZE[self.dtype]
        if self.dtype == "BF16":
            bits = struct.unpack_from("<H", self.owner.mapping, pos)[0]
            return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        return struct.unpack_from("<f", self.owner.mapping, pos)[0]

    def __iter__(self) -> Iterator[float]:
        for index in range(self.size):
            yield self[index]  # type: ignore[misc]


class SafeTensorFile:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self._parse()
            except Exception:
                self.mapping.close()
                raise
        except Exception:
            self.file.close()
            raise

    def _parse(self) -> None:
        if len(self.mapping) < 8:
            raise ValueError(f"{self.path}: truncated safetensors file")
        header_len = struct.unpack_from("<Q", self.mapping, 0)[0]
        data_start = 8 + header_len
        if header_len > 100_000_000 or header_len > len(self.mapping) - 8:
            raise ValueError(f"{self.path}: invalid header length")
        if not header_len or self.mapping[8] != ord("{"):
            raise ValueError(f"{self.path}: header must start with '{{'")
        try:
            header: dict[str, Any] = json.loads(
                self.mapping[8:data_start], object_pairs_hook=_unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{self.path}: invalid JSON header") from exc
        if not isinstance(header, dict):
            raise ValueError(f"{self.path}: header must be an object")
        self.metadata = header.pop("__metadata__", {})
        if (
            not isinstance(self.metadata, dict)
            or not all(isinstance(key, str) and isinstance(value, str)
                       for key, value in self.metadata.items())
        ):
            raise ValueError(f"{self.path}: metadata must map strings to strings")
        self.tensors: dict[str, SafeTensor] = {}
        occupied: list[tuple[int, int, str]] = []
        for name, info in header.items():
            if not isinstance(info, dict):
                raise ValueError(f"{name}: malformed tensor metadata")
            dtype = info.get("dtype")
            shape = info.get("shape")
            offsets = info.get("data_offsets")
            if dtype not in _ITEM_SIZE:
                raise ValueError(f"{name}: unsupported dtype {dtype!r}")
            if (
                not isinstance(shape, list)
                or not all(type(n) is int and n >= 0 for n in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(type(n) is int for n in offsets)
            ):
                raise ValueError(f"{name}: malformed tensor metadata")
            begin, end = offsets
            count = 1
            for extent in shape:
                count *= extent
            if begin < 0 or end < begin or end - begin != count * _ITEM_SIZE[dtype]:
                raise ValueError(f"{name}: inconsistent data offsets")
            absolute_begin, absolute_end = data_start + begin, data_start + end
            if absolute_end > len(self.mapping):
                raise ValueError(f"{name}: tensor extends past end of file")
            if end > begin:
                occupied.append((begin, end, name))
            self.tensors[name] = SafeTensor(
                self, name, dtype, tuple(shape), absolute_begin, absolute_end
            )
        cursor = 0
        for begin, end, name in sorted(occupied):
            if begin != cursor:
                raise ValueError(f"{name}: tensor data has a gap or overlap")
            cursor = end
        if cursor != len(self.mapping) - data_start:
            raise ValueError(f"{self.path}: unclaimed bytes in tensor data")

    def __getitem__(self, name: str) -> SafeTensor:
        return self.tensors[name]

    def close(self) -> None:
        self.mapping.close()
        self.file.close()

    def __enter__(self) -> "SafeTensorFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ShardedSafeTensors:
    """Resolve tensors through a HF model.safetensors.index.json."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        try:
            document = json.loads(
                self.index_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid safetensors index JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("safetensors index must be an object")
        weight_map = document.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not all(isinstance(name, str) and isinstance(filename, str)
                       for name, filename in weight_map.items())
        ):
            raise ValueError("safetensors index has no weight_map")
        self.weight_map: dict[str, str] = weight_map
        self.files: dict[str, SafeTensorFile] = {}
        root = self.index_path.parent.resolve()
        try:
            for filename in sorted(set(self.weight_map.values())):
                path = (root / filename).resolve()
                if root not in path.parents:
                    raise ValueError(f"shard escapes model directory: {filename}")
                self.files[filename] = SafeTensorFile(path)
            for name, filename in self.weight_map.items():
                if name not in self.files[filename].tensors:
                    raise ValueError(f"{name}: missing from mapped shard {filename}")
            for filename, file in self.files.items():
                for name in file.tensors:
                    if self.weight_map.get(name) != filename:
                        raise ValueError(
                            f"{name}: shard contents disagree with index weight_map"
                        )
        except Exception:
            self.close()
            raise

    def __getitem__(self, name: str) -> SafeTensor:
        filename = self.weight_map[name]
        tensor = self.files[filename][name]
        return tensor

    def close(self) -> None:
        for file in self.files.values():
            file.close()

    def __enter__(self) -> "ShardedSafeTensors":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_checkpoint(path: str | Path) -> SafeTensorFile | ShardedSafeTensors:
    path = Path(path)
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            return ShardedSafeTensors(index)
        path = path / "model.safetensors"
    if path.name.endswith(".index.json"):
        return ShardedSafeTensors(path)
    return SafeTensorFile(path)
