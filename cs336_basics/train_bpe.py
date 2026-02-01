import os
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # read data, separate them by special tokens and decode it as utf-8 words for pre-tokenization to different words
    # with open(input_path, "rb") as f:
    #     data = f.read()
    #     text = data.decode("utf-8", errors="ignore")
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
        pattern = "|".join(map(re.escape, special_tokens))
        docs = re.split(pattern, text)  # separate them by special tokens and delete the special tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # get all words after pre-tokenization
        allwords = []
        for doc in docs:
            words = re.findall(PAT, doc)
            allwords.extend(words)
        # print(allwords)
        # change all words to bytes and then to tokens
        allwords_tokens = [list(word.encode("utf-8")) for word in allwords]
        # print(allwords_tokens)
        # tokens_to_bytes dict and BPE merges list
        tokens_to_bytes = {idx: bytes([idx]) for idx in range(256)}
        merges_list = []

        num_merges = vocab_size - len(special_tokens) - 256

        # iterate num_merges times to merge bytes
        for _ in range(num_merges):
            bytes_count_dict = {}
            for word_tokens in allwords_tokens:
                # this is more memory efficient because do not create another slice of word_tokens list
                for token1, token2 in zip(word_tokens, word_tokens[1:]):
                    bytes_count_dict[(token1, token2)] = bytes_count_dict.get((token1, token2), 0) + 1
            # if not finding any adjacent bytes pair, break the merges
            if not bytes_count_dict:
                break
            else:
                # merge_pair = max(bytes_count_dict, key=bytes_count_dict.get)
                merge_pair = max(
                    bytes_count_dict,
                    key=lambda pair: (bytes_count_dict[pair], tokens_to_bytes[pair[0]], tokens_to_bytes[pair[1]]),
                )

            # add the merge_pair to the merges_list
            merges_list.append((tokens_to_bytes[merge_pair[0]], tokens_to_bytes[merge_pair[1]]))
            new_token_id = len(tokens_to_bytes)
            tokens_to_bytes[new_token_id] = tokens_to_bytes[merge_pair[0]] + tokens_to_bytes[merge_pair[1]]

            # replace the new token back to the original words
            new_allwords_tokens = []
            for word_tokens in allwords_tokens:
                new_word_tokens = []
                index = 0
                while index < len(word_tokens):
                    if (
                        index < len(word_tokens) - 1
                        and word_tokens[index] == merge_pair[0]
                        and word_tokens[index + 1] == merge_pair[1]
                    ):
                        new_word_tokens.append(new_token_id)
                        index += 2
                    else:
                        new_word_tokens.append(word_tokens[index])
                        index += 1
                new_allwords_tokens.append(new_word_tokens)
            allwords_tokens = new_allwords_tokens

        # add the special tokens back into the vocab
        for special_token in special_tokens:
            tokens_to_bytes[len(tokens_to_bytes)] = special_token.encode("utf-8")

        # print(tokens_to_bytes)
        # print(merges_list)

        return tokens_to_bytes, merges_list
