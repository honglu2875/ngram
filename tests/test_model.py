import time
from pathlib import Path

import numpy as np
import pytest

from ngram import NGramModel

test_root = Path(__file__).parent.resolve()
@pytest.mark.parametrize(("vocab, pre, ans"), ((50432, 0, 3075), (50432, 1, 3075), (10, 1, 0), (10, 2, 0)))
def test_model(vocab, pre, ans):
    original_vocab = 50432
    bin_path = str(test_root / "data.bin")
    data = np.load(open(bin_path, "rb"))
    model = NGramModel(5, vocab, pre)
    print(data % vocab)
    start = time.perf_counter()
    model.train(data % vocab)
    print(time.perf_counter() - start)

    assert model.get_distribution(np.array([8735], dtype=np.uint16)).argmax() == ans

