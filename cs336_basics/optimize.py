import pathlib
import time

from train_bpe import train_bpe

if __name__ == "__main__":
    # Example call with test data
    BASE_PATH = pathlib.Path(__file__).resolve().parent.parent
    input_path = BASE_PATH / "data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = BASE_PATH / "tests/fixtures/corpus.en"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    dt = (end_time - start_time) * 1000  # time difference in miliseconds
    print(f"dt: {dt:.2f}ms")
