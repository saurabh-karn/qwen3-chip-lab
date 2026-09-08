"""Small standard-library tokenizer.json byte-level BPE implementation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def _byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAD + 1))
    visible += list(range(0xAE, 0xFF + 1))
    encoded = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            encoded.append(256 + extra)
            extra += 1
    encoder = dict(zip(visible, map(chr, encoded)))
    return encoder, {character: byte for byte, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_alphabet()
_QWEN3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _pretokenize(text: str) -> list[str]:
    """Implement the canonical Qwen3 Split regex without a regex dependency."""
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        lower = text[cursor:].lower()
        contraction = next((item for item in _CONTRACTIONS if lower.startswith(item)), None)
        if contraction is not None:
            end = cursor + len(contraction)
        else:
            character = text[cursor]
            letter_start = cursor
            if (
                character not in "\r\n"
                and not _is_letter(character)
                and not _is_number(character)
                and cursor + 1 < len(text)
                and _is_letter(text[cursor + 1])
            ):
                letter_start += 1
            if _is_letter(text[letter_start]):
                end = letter_start + 1
                while end < len(text) and _is_letter(text[end]):
                    end += 1
            elif _is_number(character):
                # The published expression deliberately isolates each number.
                end = cursor + 1
            else:
                symbol_start = cursor
                if character == " " and cursor + 1 < len(text):
                    following = text[cursor + 1]
                    if not following.isspace() and not _is_letter(following) and not _is_number(following):
                        symbol_start += 1
                if (
                    symbol_start < len(text)
                    and not text[symbol_start].isspace()
                    and not _is_letter(text[symbol_start])
                    and not _is_number(text[symbol_start])
                ):
                    end = symbol_start + 1
                    while (
                        end < len(text)
                        and not text[end].isspace()
                        and not _is_letter(text[end])
                        and not _is_number(text[end])
                    ):
                        end += 1
                    while end < len(text) and text[end] in "\r\n":
                        end += 1
                elif character.isspace():
                    whitespace_end = cursor + 1
                    while whitespace_end < len(text) and text[whitespace_end].isspace():
                        whitespace_end += 1
                    last_newline = max(
                        text.rfind("\r", cursor, whitespace_end),
                        text.rfind("\n", cursor, whitespace_end),
                    )
                    if last_newline >= cursor:
                        end = last_newline + 1
                    elif whitespace_end == len(text):
                        end = whitespace_end
                    elif whitespace_end - cursor > 1:
                        # \s+(?!\S) backtracks one character before non-space.
                        end = whitespace_end - 1
                    else:
                        end = whitespace_end
                else:
                    # The alternatives cover every Unicode scalar.
                    end = cursor + 1
        chunks.append(text[cursor:end])
        cursor = end
    return chunks


class Tokenizer:
    """Byte-level BPE subset used by the published Qwen tokenizer."""

    def __init__(self, tokenizer_json: str | Path) -> None:
        document = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
        model = document.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError("only tokenizer.json BPE models are supported")
        supported_options = {
            "dropout": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
        }
        if any(
            key in model and model[key] != value
            for key, value in supported_options.items()
        ):
            raise ValueError("unsupported BPE model options")
        self.vocab: dict[str, int] = model["vocab"]
        if (
            not isinstance(self.vocab, dict)
            or not all(
                isinstance(token, str) and type(index) is int
                for token, index in self.vocab.items()
            )
            or len(set(self.vocab.values())) != len(self.vocab)
        ):
            raise ValueError("BPE vocabulary must map strings to unique integer ids")
        self.inverse = {value: key for key, value in self.vocab.items()}
        self.unk_token = model.get("unk_token")
        merges = model.get("merges", [])
        pairs = []
        for merge in merges:
            parts = merge.split() if isinstance(merge, str) else merge
            if len(parts) != 2:
                raise ValueError(f"invalid BPE merge: {merge!r}")
            pairs.append((parts[0], parts[1]))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate BPE merge")
        self.ranks = {pair: rank for rank, pair in enumerate(pairs)}
        normalizer = document.get("normalizer")
        if normalizer not in (None, {"type": "NFC"}):
            raise ValueError("only the canonical NFC normalizer is supported")
        self.normalize_nfc = normalizer == {"type": "NFC"}
        expected_pre_tokenizer = {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _QWEN3_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        }
        pre_tokenizer = document.get("pre_tokenizer")
        # Small operator fixtures historically omitted the graph; in that case
        # use the same canonical split rather than the unrelated GPT-2 regex.
        if pre_tokenizer is not None and pre_tokenizer != expected_pre_tokenizer:
            raise ValueError("unsupported tokenizer.json pre_tokenizer graph")
        expected_byte_level = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        decoder = document.get("decoder")
        if decoder is not None and decoder != expected_byte_level:
            raise ValueError("unsupported tokenizer.json decoder")
        added = document.get("added_tokens", [])
        if not isinstance(added, list):
            raise ValueError("added_tokens must be a list")
        self.added: dict[str, int] = {}
        self.special: dict[str, int] = {}
        for item in added:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("content"), str)
                or type(item.get("id")) is not int
            ):
                raise ValueError("malformed added token")
            if (
                item.get("single_word", False)
                or item.get("lstrip", False)
                or item.get("rstrip", False)
                or item.get("normalized", False)
            ):
                raise ValueError("added-token boundary modifiers are unsupported")
            content, token_id = item["content"], item["id"]
            if (
                content in self.added
                or token_id in self.inverse
                or token_id in self.added.values()
            ):
                raise ValueError("duplicate added token content or id")
            self.added[content] = token_id
            if item.get("special"):
                self.special[content] = token_id
        self._added_inverse = {value: key for key, value in self.added.items()}
        self._cache: dict[str, tuple[str, ...]] = {}

    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        while len(word) > 1:
            candidates = [
                (self.ranks[(word[i], word[i + 1])], i)
                for i in range(len(word) - 1)
                if (word[i], word[i + 1]) in self.ranks
            ]
            if not candidates:
                break
            _, first_index = min(candidates)
            target = (word[first_index], word[first_index + 1])
            merged = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == target:
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        self._cache[token] = word
        return word

    def _ordinary(self, text: str) -> list[int]:
        ids = []
        if self.normalize_nfc:
            text = unicodedata.normalize("NFC", text)
        for chunk in _pretokenize(text):
            encoded = "".join(_BYTE_ENCODER[byte] for byte in chunk.encode("utf-8"))
            for piece in self._bpe(encoded):
                token_id = self.vocab.get(piece)
                if token_id is None:
                    if self.unk_token is None:
                        raise ValueError(f"BPE piece missing from vocabulary: {piece!r}")
                    token_id = self.vocab[self.unk_token]
                ids.append(token_id)
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not allowed_special or not self.added:
            return self._ordinary(text)
        pattern = re.compile("|".join(re.escape(x) for x in sorted(self.added, key=len, reverse=True)))
        result, cursor = [], 0
        for match in pattern.finditer(text):
            result.extend(self._ordinary(text[cursor:match.start()]))
            result.append(self.added[match.group()])
            cursor = match.end()
        result.extend(self._ordinary(text[cursor:]))
        return result

    def decode(self, token_ids: list[int], skip_special: bool = False) -> str:
        special_ids = set(self.special.values())
        pieces = []
        for token_id in token_ids:
            if skip_special and token_id in special_ids:
                continue
            if token_id not in self.inverse:
                added = self._added_inverse.get(token_id)
                if added is None:
                    raise ValueError(f"unknown token id {token_id}")
                pieces.append(added.encode())
                continue
            token = self.inverse[token_id]
            pieces.append(bytes(_BYTE_DECODER[character] for character in token))
        return b"".join(pieces).decode("utf-8", errors="replace")
