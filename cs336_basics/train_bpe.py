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
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
        pattern = "|".join(map(re.escape, special_tokens))
        docs = re.split(pattern, text)  # separate them by special tokens and delete the special tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # get all words after pre-tokenization
        allwords_count_dict = {}
        allwords_token = []
        for doc in docs:
            for iter in re.finditer(PAT, doc):
                word_bytes = iter.group().encode("utf-8")
                # print(word_tokens)
                allwords_count_dict[word_bytes] = allwords_count_dict.get(word_bytes, 0) + 1

        for word_bytes in allwords_count_dict.keys():
            allwords_token.append(list(word_bytes))
        # change all words to bytes and then to tokens
        # TODO: optimizing for parallel

        # tokens_to_bytes dict and BPE merges list
        tokens_to_bytes = {idx: bytes([idx]) for idx in range(256)}
        merges_list = []
        len_special_tokens = len(special_tokens)

        num_merges = vocab_size - len_special_tokens - 256

        # count the frequency of adjacent bytes
        bytes_count_dict = {}
        for word_bytes in allwords_count_dict.keys():
            word_tokens = list(word_bytes)
            for token1, token2 in zip(word_tokens, word_tokens[1:]):
                bytes_count_dict[(token1, token2)] = (
                    bytes_count_dict.get((token1, token2), 0) + allwords_count_dict[word_bytes]
                )

        # iterate num_merges times to merge bytes
        for i in range(num_merges):
            # if not finding any adjacent bytes pair, break the merges
            if not bytes_count_dict:
                break
            else:
                merge_pair = max(
                    bytes_count_dict,
                    key=lambda pair: (bytes_count_dict[pair], tokens_to_bytes[pair[0]], tokens_to_bytes[pair[1]]),
                )

            # add the merge_pair to the merges_list
            merges_list.append((tokens_to_bytes[merge_pair[0]], tokens_to_bytes[merge_pair[1]]))
            new_token_id = i + 256
            tokens_to_bytes[new_token_id] = tokens_to_bytes[merge_pair[0]] + tokens_to_bytes[merge_pair[1]]

            # replace the new token back to the original words
            new_allwords_token = []
            for word_tokens in allwords_token:
                new_word_tokens = []
                index = 0
                change = False
                while index < len(word_tokens):
                    if (
                        index < len(word_tokens) - 1
                        and word_tokens[index] == merge_pair[0]
                        and word_tokens[index + 1] == merge_pair[1]
                    ):
                        new_word_tokens.append(new_token_id)
                        index += 2
                        change = True
                    else:
                        new_word_tokens.append(word_tokens[index])
                        index += 1
                new_allwords_token.append(new_word_tokens)
                if change:
                    original_word = b"".join(tokens_to_bytes[t] for t in word_tokens)
                    for token1, token2 in zip(word_tokens, word_tokens[1:]):
                        bytes_count_dict[(token1, token2)] -= allwords_count_dict[original_word]
                    for token1, token2 in zip(new_word_tokens, new_word_tokens[1:]):
                        bytes_count_dict[(token1, token2)] = (
                            bytes_count_dict.get((token1, token2), 0) + allwords_count_dict[original_word]
                        )
            allwords_token = new_allwords_token

        # add the special tokens back into the vocab
        for i in range(len_special_tokens):
            new_token_id = vocab_size - len_special_tokens
            tokens_to_bytes[i + new_token_id] = special_tokens[i].encode("utf-8")

        # print(tokens_to_bytes)
        # print(merges_list)

        return tokens_to_bytes, merges_list
