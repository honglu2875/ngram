from pathlib import Path

import numpy as np
import pytest

from ngram import NGramModel

test_root = Path(__file__).parent.resolve()

@pytest.fixture
def model():
    vocab = 50432
    bin_path = str(test_root / "data.bin")
    data = np.load(open(bin_path, "rb"))
    model = NGramModel(5, vocab, 0)
    model.train(data)
    return model

@pytest.fixture
def small_model():
    vocab = 50432
    new_vocab = 100
    bin_path = str(test_root / "data.bin")
    data = np.load(open(bin_path, "rb"))
    model = NGramModel(5, new_vocab, 0)
    model.train(data % new_vocab)
    return model

