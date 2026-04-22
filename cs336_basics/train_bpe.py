import os
from collections import Counter, defaultdict
from multiprocessing import Pool
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


def pretokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[str, int]:
    with open(input_path, "rb") as f:
        f.seek(start)  # jump to byte position
        raw = f.read(end - start)  # read raw bytes
        text = raw.decode("utf-8", errors="ignore")  # you decode manually
        pattern = "|".join(map(re.escape, special_tokens))
        docs = re.split(pattern, text)  # separate them by special tokens and delete the special tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # get all words after pre-tokenization
        allwords_count_dict = {}
        for doc in docs:
            for iter in re.finditer(PAT, doc):
                word = iter.group()
                allwords_count_dict[word] = allwords_count_dict.get(word, 0) + 1

    return allwords_count_dict


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # read data, separate them by special tokens and decode it as utf-8 words for pre-tokenization to different words
    with open(input_path, "rb") as f:
        num_processes = 8  # os.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunk_args = []
        for start, end in zip(boundaries, boundaries[1:]):
            chunk_args.append((input_path, start, end, special_tokens))

        with Pool(processes=num_processes) as pool:
            results = pool.starmap(pretokenize_chunk, chunk_args)

        # # get all words after pre-tokenization
        allwords_count_dict = Counter()
        allwords_token = []
        allwords_count = []
        for r in results:
            allwords_count_dict += Counter(r)

        for word, count in allwords_count_dict.items():
            allwords_token.append(list(word.encode("utf-8")))
            allwords_count.append(count)

        size = len(allwords_token)

        # tokens_to_bytes dict and BPE merges list
        tokens_to_bytes = {idx: bytes([idx]) for idx in range(256)}
        merges_list = []
        len_special_tokens = len(special_tokens)

        num_merges = vocab_size - len_special_tokens - 256

        # count the frequency of adjacent bytes
        bytes_count_dict = defaultdict(int)
        pair_to_word_indices = defaultdict(set)

        for i in range(size):
            word_tokens = allwords_token[i]
            count = allwords_count[i]
            for token1, token2 in zip(word_tokens, word_tokens[1:]):
                bytes_count_dict[(token1, token2)] += count
                pair_to_word_indices[(token1, token2)].add(i)

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

            set_with_merge_pair = pair_to_word_indices[merge_pair]
            for i in list(set_with_merge_pair):
                word_tokens = allwords_token[i]
                new_word_tokens = []
                index = 0
                word_size = len(word_tokens)
                while index < word_size:
                    if (
                        index < word_size - 1
                        and word_tokens[index] == merge_pair[0]
                        and word_tokens[index + 1] == merge_pair[1]
                    ):
                        new_word_tokens.append(new_token_id)
                        index += 2
                    else:
                        new_word_tokens.append(word_tokens[index])
                        index += 1
                allwords_token[i] = new_word_tokens
                for token1, token2 in zip(word_tokens, word_tokens[1:]):
                    bytes_count_dict[(token1, token2)] -= allwords_count[i]
                    pair_to_word_indices[(token1, token2)].discard(i)
                for token1, token2 in zip(new_word_tokens, new_word_tokens[1:]):
                    bytes_count_dict[(token1, token2)] = bytes_count_dict.get((token1, token2), 0) + allwords_count[i]
                    pair_to_word_indices[(token1, token2)].add(i)

        # add the special tokens back into the vocab
        for i in range(len_special_tokens):
            new_token_id = vocab_size - len_special_tokens
            tokens_to_bytes[i + new_token_id] = special_tokens[i].encode("utf-8")

        return tokens_to_bytes, merges_list
