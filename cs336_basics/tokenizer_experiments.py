import os
import pathlib
import random
import time

from tokenizer import Tokenizer


def get_tokenizer(dataset_name: str) -> Tokenizer:
    filepath = pathlib.Path(__file__).resolve()
    if dataset_name == "TinyStories":
        vocab_path = filepath.parent / "TinyStories_vocab.json"
        merges_path = filepath.parent / "TinyStories_merges.txt"
    elif dataset_name == "OpenWebText":
        vocab_path = filepath.parent / "OpenWebText_vocab.json"
        merges_path = filepath.parent / "OpenWebText_merges.txt"
    special_tokens = ["<|endoftext|>"]
    return Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)


def sample(dataset_name: str, times: int) -> int:
    tokenizer = get_tokenizer(dataset_name)
    if dataset_name == "TinyStories":
        DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data/TinyStoriesV2-GPT4-train.txt"
    elif dataset_name == "OpenWebText":
        DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data/owt_train.txt"
    sum_compression_ratio = 0
    with open(DATA_PATH, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        total_chunk_size = 0
        for i in range(times):
            while True:
                random_position = random.randint(0, file_size)
                file.seek(random_position)
                start = -1
                end = -1
                mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
                initial_position = random_position
                while end == -1:
                    mini_chunk = file.read(mini_chunk_size)

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        end = file_size
                        break

                    found_at = mini_chunk.find(b"<|endoftext|>")
                    if found_at != -1:
                        if start == -1:
                            start = initial_position + found_at + len(b"<|endoftext|>")
                        else:
                            end = initial_position + found_at + len(b"<|endoftext|>")

                    initial_position += mini_chunk_size

                if start != -1 and end != -1:
                    break

            file.seek(start)
            raw = file.read(end - start)
            text = raw.decode("utf-8", errors="ignore")
            tokens = tokenizer.encode(text)
            num_bytes = len(raw)
            num_tokens = len(tokens)
            # print(f"compression ratio (bytes/token) for {dataset_name} in run {i} is {num_bytes / num_tokens}")
            sum_compression_ratio += num_bytes / num_tokens
            total_chunk_size += end - start
    print(f"The average compression ratio (bytes/token) for {dataset_name} is {sum_compression_ratio / times}")
    return total_chunk_size


def cross_tokenize(dataset_name: str, times: int):
    tokenizer = get_tokenizer(dataset_name)
    if dataset_name == "TinyStories":
        DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data/owt_train.txt"
    elif dataset_name == "OpenWebText":
        DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data/TinyStoriesV2-GPT4-train.txt"
    sum_compression_ratio = 0
    with open(DATA_PATH, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        for i in range(times):
            while True:
                random_position = random.randint(0, file_size)
                file.seek(random_position)
                start = -1
                end = -1
                mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
                initial_position = random_position
                while end == -1:
                    mini_chunk = file.read(mini_chunk_size)

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        end = file_size
                        break

                    found_at = mini_chunk.find(b"<|endoftext|>")
                    if found_at != -1:
                        if start == -1:
                            start = initial_position + found_at + len(b"<|endoftext|>")
                        else:
                            end = initial_position + found_at + len(b"<|endoftext|>")

                    initial_position += mini_chunk_size

                if start != -1 and end != -1:
                    break

            file.seek(start)
            raw = file.read(end - start)
            text = raw.decode("utf-8", errors="ignore")
            tokens = tokenizer.encode(text)
            num_bytes = len(raw)
            num_tokens = len(tokens)
            # print(f"compression ratio (bytes/token) for {dataset_name} in run {i} is {num_bytes / num_tokens}")
            sum_compression_ratio += num_bytes / num_tokens
    print(f"The average compression ratio (bytes/token) for {dataset_name} is {sum_compression_ratio / times}")


def tokenizer_throughput(dataset_name: str):
    if dataset_name == "TinyStories":
        t0 = time.time()
        total_chunk_size = sample("TinyStories", 10)
        t1 = time.time()
        dt = t1 - t0
        throughput = total_chunk_size / (dt)
        print(f"the tokenizer_throughput for {dataset_name} is {throughput:.2f} bytes/second")
        calculate_time = 825 * 1e9 / throughput
        print(f"it take to {calculate_time:.2f} seconds to tokenize the Pile dataset")

    elif dataset_name == "OpenWebText":
        t0 = time.time()
        total_chunk_size = sample("OpenWebText", 10)
        t1 = time.time()
        dt = t1 - t0
        throughput = total_chunk_size / (dt)
        print(f"the tokenizer_throughput for {dataset_name} is {throughput:.2f} bytes/second")
        calculate_time = 825 * 1e9 / throughput
        print(f"it take to {calculate_time:.2f} seconds to tokenize the Pile dataset")


if __name__ == "__main__":
    tokenizer_throughput("TinyStories")
    tokenizer_throughput("OpenWebText")
    # total_chunk_size = sample("TinyStories", 10)
    # total_chunk_size = sample("OpenWebText", 10)
    # cross_tokenize("TinyStories", 10)
    # cross_tokenize("OpenWebText", 10)

    # test to make sure the get_tokenizer function works.
    # FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"
    # with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
    #     corpus_contents = f.read()
    #     ids = tokenizer.encode(corpus_contents)
    #     assert tokenizer.decode(ids) == corpus_contents

    # test for overlapping special tokens
    # tokenizer = get_tokenizer("TinyStories")
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    # ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    # print(tokenized_string)
    # assert tokenized_string.count("<|endoftext|>") == 1
    # assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # # Test roundtrip
    # assert tokenizer.decode(ids) == test_string
