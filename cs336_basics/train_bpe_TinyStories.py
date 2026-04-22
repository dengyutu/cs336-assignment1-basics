import json
import pathlib
import resource
import time

from train_bpe import train_bpe

if __name__ == "__main__":
    # Example call with test data
    BASE_PATH = pathlib.Path(__file__).resolve().parent.parent
    input_path = BASE_PATH / "data/TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    dt = end_time - start_time  # time difference in seconds
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")  # macOS: bytes to MB
    print(f"training time: {dt:.2f}s")

    with open("TinyStories_merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

    vocab_serializable = {str(k): str(v) for k, v in vocab.items()}
    with open("TinyStories_vocab.json", "w") as f:
        json.dump(vocab_serializable, f)

    longest_token = b""
    longest_token_len = 0
    for k in vocab.keys():
        bytes_token = vocab[k]
        if len(bytes_token) > longest_token_len:
            longest_token = bytes_token
            longest_token_len = len(bytes_token)

    print(longest_token)
