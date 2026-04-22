import ast
import json
from collections.abc import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.bytes_to_tokens = {v: k for k, v in vocab.items()}
        if special_tokens is not None:
            self.special_tokens.sort(key=len, reverse=True)
            pattern = "|".join(map(re.escape, self.special_tokens))
            self.pattern = f"({pattern})"
        else:
            self.pattern = None
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.merge_pairs_to_tokens = {
            (self.bytes_to_tokens[bytes1], self.bytes_to_tokens[bytes2]): self.bytes_to_tokens[(bytes1 + bytes2)]
            for (bytes1, bytes2) in merges
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as f:
            vocab_str = json.load(f)
            vocab = {int(k): ast.literal_eval(v) for k, v in vocab_str.items()}
        with open(merges_filepath) as f:
            merges = []
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                parts = re.findall(r"b'[^']*'|b\"[^\"]*\"", line)
                if len(parts) != 2:
                    raise ValueError(f"The line {i}'s content {parts} is wrong")
                else:
                    byte1 = ast.literal_eval(parts[0])
                    byte2 = ast.literal_eval(parts[1])
                    merges.append((byte1, byte2))
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_tokens = special_token.encode("utf-8")
                if byte_encoded_special_tokens not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_tokens
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_encoding = []
        if self.pattern is not None:
            chunks = re.split(self.pattern, text)
        else:
            chunks = [text]
        for chunk in chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                token_encoding.append(self.bytes_to_tokens[chunk.encode("utf-8")])
            else:
                for iter in re.finditer(self.PAT, chunk):
                    word = iter.group()
                    word_tokens = [self.bytes_to_tokens[bytes([b])] for b in word.encode("utf-8")]
                    while True:
                        merge_pair_tokens = float("inf")
                        merge_pair = (None, None)
                        for token1, token2 in zip(word_tokens, word_tokens[1:]):
                            new_token = self.merge_pairs_to_tokens.get((token1, token2), -1)
                            if new_token != -1 and new_token < merge_pair_tokens:
                                merge_pair_tokens = new_token
                                merge_pair = (token1, token2)
                        if merge_pair_tokens == float("inf"):
                            break
                        else:
                            new_word_tokens = []
                            index = 0
                            word_size = len(word_tokens)
                            while index < word_size:
                                if (
                                    index < word_size - 1
                                    and word_tokens[index] == merge_pair[0]
                                    and word_tokens[index + 1] == merge_pair[1]
                                ):
                                    new_word_tokens.append(merge_pair_tokens)
                                    index += 2
                                else:
                                    new_word_tokens.append(word_tokens[index])
                                    index += 1
                            word_tokens = new_word_tokens
                    token_encoding.extend(word_tokens)
        return token_encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_list = self.encode(text)
            yield from token_list

    def decode(self, ids: list[int]) -> str:
        decoded_string = ""
        decoded_bytes = b""
        for token in ids:
            decoded_bytes += self.vocab[token]
        decoded_string = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_string
