# Copyright 2024 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cimport cython
cimport openmp
from cython.operator cimport dereference, postincrement
from cython.parallel cimport prange
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.atomic cimport atomic
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np

import json
import os

ctypedef cython.float cfloat
ctypedef unsigned char cbyte
ctypedef unsigned short vocab_size_t
ctypedef np.uint16_t np_vocab_size_t
ctypedef unsigned long long count_t
ctypedef np_vocab_size_t[:] NGram
ctypedef np_vocab_size_t[:, :] NgramBatch

cdef extern from "nodes.h":
    cdef cppclass TrieNode:
        TrieNode() noexcept nogil
        TrieNode(vocab_size_t vocabSize) noexcept nogil
        void acquire() noexcept nogil
        void release() noexcept nogil
        void inc() noexcept nogil
        bool contains(vocab_size_t key) noexcept nogil
        TrieNode *get(vocab_size_t key) noexcept nogil
        void put(vocab_size_t key, TrieNode *new_child) noexcept nogil
        void serialize(vector[cbyte] output) noexcept nogil
        vector[cbyte] serialize() noexcept nogil
        count_t count() noexcept nogil
        @staticmethod
        TrieNode *deserialize(const cbyte* input, long long size) noexcept nogil

@cython.cfunc
@cython.exceptval(-1, check=True)
cdef int insert(TrieNode *root, vocab_size_t *ngram, count_t n, vocab_size_t vocab_size) nogil:
    cdef TrieNode *node = root
    cdef vocab_size_t token

    node.inc()
    for i in range(n):
        token = ngram[i]

        if not node.contains(token):
            node.put(token, new TrieNode())
        node = node.get(token)

        node.inc()

    return 0

cdef TrieNode* trace_node(TrieNode *root, NGram ngram) noexcept nogil:
    cdef TrieNode *node = root
    cdef vocab_size_t token

    for i in range(ngram.shape[0]):
        token = ngram[i]
        if not node.contains(token):
            return NULL
        node = node.get(token)
    return node

cdef void _preallocate(long preallocate_depth, TrieNode *root, vocab_size_t vocab_size) noexcept nogil:
    cdef vector[TrieNode *] queue
    cdef vector[int] depths
    cdef int p = 0

    if preallocate_depth != 0:
        queue.push_back(root)
        depths.push_back(0)

        while p < queue.size():
            if depths[p] < preallocate_depth:
                for i in range(vocab_size):
                    if depths[p] < preallocate_depth - 1:
                        queue[p].put(i, new TrieNode())
                    else:
                        queue[p].put(i, new TrieNode(vocab_size))
                    queue.push_back(queue[p].get(i))
                    depths.push_back(depths[p] + 1)
            p += 1

cdef _read_tree_size(char *s):
    cdef count_t c
    memcpy(&c, s, sizeof(count_t))
    return c

def _merge_serialized(py_list: list[bytes], root_count: int) -> bytes:
    cdef char* buffer
    vocab_size = len(py_list)
    singletons = []
    children_count = 0

    for i in range(vocab_size):
        if py_list[i] is not None:
            children_count += 1
            buffer = py_list[i]
            if _read_tree_size(buffer) == 1:
                singletons.append(i)

    def _key_bytes(c):
        return c.to_bytes(sizeof(vocab_size_t), byteorder='little')

    prefix = root_count.to_bytes(sizeof(count_t), byteorder='little') + \
             children_count.to_bytes(sizeof(vocab_size_t), byteorder='little') + \
             len(singletons).to_bytes(sizeof(vocab_size_t), byteorder='little')
    singleton_bytes = b"".join([_key_bytes(c) for c in singletons])
    children_bytes = [_key_bytes(i) + c for i, c in enumerate(py_list) if c is not None]
        
    return prefix + singleton_bytes + b"".join(children_bytes)


cdef class _NGramModel:
    cdef:
        TrieNode *root
        long n
        vocab_size_t vocab_size
        long preallocate_depth

    def __init__(self, long n, vocab_size_t vocab_size, long preallocate_depth):
        cdef Py_ssize_t i
        cdef int _pd

        self.n = n
        self.vocab_size = vocab_size
        self.root = new TrieNode(vocab_size)
        self.preallocate_depth = preallocate_depth
        if preallocate_depth > 0:
            _pd = preallocate_depth
            for i in prange(vocab_size, schedule='dynamic', nogil=True, chunksize=1):
                if _pd == 1:
                    self.root.put(i, new TrieNode())
                else:
                    self.root.put(i, new TrieNode(vocab_size))
                    _preallocate(_pd - 1, self.root.get(i), vocab_size)

       
    def train(self, np.ndarray[np_vocab_size_t, ndim=2] data):
        cdef Py_ssize_t batch_size = data.shape[0]
        cdef Py_ssize_t sequence_len = data.shape[1]

        cdef np.ndarray[ndim=2, dtype=np_vocab_size_t, mode='c'] cont_data = np.ascontiguousarray(data)

        cdef Py_ssize_t i, j, k
        cdef TrieNode *node

        for i in prange(batch_size, schedule='dynamic', nogil=True, chunksize=1):
            for j in range(sequence_len - self.n + 1):
                insert(self.root, <vocab_size_t *>&cont_data[i, j], self.n, self.vocab_size)

    @staticmethod
    cdef cfloat* _get_distribution(cfloat[:] buffer, TrieNode *root, NGram ngram, vocab_size_t vocab_size) nogil:
        cdef TrieNode *node
        cdef int i = 0

        node = trace_node(root, ngram)
        if node == NULL or node.count() == 0:
            for i in prange(vocab_size):
                buffer[i] = 0
        else:
            while i < vocab_size:
                if not node.contains(i):
                    buffer[i] = 0
                else:
                    buffer[i] = node.get(i).count() / node.count()
                i += 1

    def set_root_from_buffer(self, const cbyte[:] buffer, int size):
        self.root = TrieNode.deserialize(&buffer[0], size)

    cdef count_t get_size(self, NGram ngram):
        return trace_node(self.root, ngram).count()

    def __len__(self):
        return self.root.count()

    def get_distribution(self, NGram ngram):
        cdef np.ndarray[ndim=1, dtype=cfloat, mode='c'] prob_array = np.zeros((self.vocab_size,), dtype=np.float32)

        buffer = cython.declare(cfloat[:], prob_array) 
        _NGramModel._get_distribution(buffer, self.root, ngram, self.vocab_size)
        return prob_array
    
    def serialize_trie_as_list(self):
        cdef vector[vector[cbyte]] outputs
        cdef vocab_size_t i

        for i in range(self.vocab_size):
            outputs.push_back(vector[cbyte]())

        for i in prange(self.vocab_size, schedule='dynamic', nogil=True, chunksize=1):
            if self.root.contains(i):
                outputs[i] = self.root.get(i).serialize()

        py_list = []
        for arr in outputs:
            if arr.size() > 0:
                py_list.append(bytes(<cbyte[:arr.size():1]>arr.data()))
            else:
                py_list.append(None)
        return py_list

    def serialize_trie_as_bytes(self):
        py_list = self.serialize_trie_as_list()
        return _merge_serialized(py_list, len(self))

    @classmethod
    def from_bytes(cls, bytes input, long n, vocab_size_t vocab_size, long preallocate_depth):
        cdef const cbyte[:] buffer = input

        model = cls(n, vocab_size, preallocate_depth)
        model.set_root_from_buffer(buffer, len(input))

        return model
        
class NGramModel:
    """The Python wrapper of the cdef class _NGramModel"""
    def __init__(self, n: int, vocab_size: int, preallocate_depth: int = 0):
        self.n = n
        self.vocab_size = vocab_size
        self.preallocate_depth = preallocate_depth

        self._model = _NGramModel(n, vocab_size, preallocate_depth)

    def train(self, data: np.ndarray):
        self._model.train(data)

    def get_size(self, ngram: np.ndarray):
        self._model.get_size(ngram)

    def get_distribution(self, ngram: np.ndarray):
        return self._model.get_distribution(ngram)

    def __len__(self):
        return len(self._model)

    def serialize(self, as_list=False):
        if as_list:
            return self._model.serialize_trie_as_list()
        else:
            return self._model.serialize_trie_as_bytes()

    def save(self, path: str):
        if os.path.exists(path):
            raise ValueError(f"A file or directory already exists at the path {path}.")
        serialized = self.serialize(as_list=True)
        os.makedirs(path, exist_ok=True)
        #for i in prange(self.vocab_size, schedule='dynamic', nogil=True, chunksize=1):
        for i in range(self.vocab_size):
            with open(os.path.join(path, str(i)), "wb") as f:
                f.write(serialized[i])

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(
                {
                    "n": self.n,
                    "vocab_size": self.vocab_size,
                    "preallocate_depth": self.preallocate_depth,
                    "root_count": len(self),
                },
                f
            )

    @classmethod
    def from_bytes(cls, input: bytes, n: int, vocab_size: int, preallocate_depth: int = 0):
        wrapped = cls(n, vocab_size, preallocate_depth)
        wrapped._model = _NGramModel.from_bytes(input, n, vocab_size, preallocate_depth)
        return wrapped

    @classmethod
    def load(cls, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"load method expects a directory {path}.")
        config = json.load(open(os.path.join(path, "config.json"), "r"))
        n, vocab_size, preallocate_depth, root_count = config["n"], config["vocab_size"], config["preallocate_depth"], config["root_count"]
        serialized = [None] * vocab_size 
        #for i in prange(vocab_size, schedule="dynamic", nogil=True, chunksize=1):
        for i in range(vocab_size):
            with open(os.path.join(path, str(i)), "rb") as f:
                serialized[i] = bytes(f.read())

        return cls.from_bytes(_merge_serialized(serialized, root_count), n, vocab_size, preallocate_depth)
