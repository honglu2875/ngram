
import numpy as np
import os
import tempfile

from ngram import NGramModel


def test_serialize(model):
    b = model.serialize(as_list=False)
    model2 = NGramModel.from_bytes(b, 5, model.vocab_size, 0)

    dist = model.get_distribution(np.array([26514, 6901], dtype=np.uint16))
    dist2 = model2.get_distribution(np.array([26514, 6901], dtype=np.uint16))
    assert np.all(dist == dist2)


def test_save_load(small_model):
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "test")
        small_model.save(path)
        model2 = NGramModel.load(path)
        assert len(small_model.serialize(as_list=False)) == len(model2.serialize(as_list=False))
       
        v = small_model.vocab_size
        dist = small_model.get_distribution(np.array([26514 % v, 6901 % v], dtype=np.uint16))
        dist2 = model2.get_distribution(np.array([26514 % v, 6901 % v], dtype=np.uint16))
        assert np.all(dist == dist2)


