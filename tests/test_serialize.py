from ngram import NGramModel
import time
import pytest
import numpy as np
from pathlib import Path


test_root = Path(__file__).parent.resolve()

def test_serialize():
    vocab = 50432
    bin_path = str(test_root / "data.bin")
    data = np.load(open(bin_path, "rb"))
    model = NGramModel(5, vocab, 0)
    model.train(data)

    b = model.serialize(as_list=False)
    model2 = NGramModel.from_bytes(b, 5, vocab, 0)

    dist = model.get_distribution(np.array([26514, 6901], dtype=np.uint16))
    dist2 = model.get_distribution(np.array([26514, 6901], dtype=np.uint16))
    assert np.all(dist == dist2)


