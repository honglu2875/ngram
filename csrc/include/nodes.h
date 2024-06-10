// Copyright 2024 Honglu Fan (https://github.com/honglu2875).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <atomic>

// Type aliases for clarity
using vocab_size_t = unsigned short;
using count_t = unsigned long long;

class TrieNode {
    private:
	std::atomic<count_t> _count;
        std::unordered_map<vocab_size_t, TrieNode*> _children;
	bool _preallocated = false;

    public:
        omp_lock_t lock;

        TrieNode();
	TrieNode(vocab_size_t vocabSize);
        ~TrieNode();
        void acquire();
        void release();
        void inc();
	count_t count();
	void set_count(count_t);
        bool contains(vocab_size_t key);
        TrieNode *get(vocab_size_t key);
        void put(vocab_size_t key, TrieNode *newChild);
	std::vector<vocab_size_t> get_singleton_children();
	void serialize(std::vector<unsigned char> &output);
	std::vector<unsigned char> serialize();
	static TrieNode* deserialize(std::vector<unsigned char>::iterator &begin, const std::vector<unsigned char>::iterator &end);
	static TrieNode* deserialize(std::vector<unsigned char>input);
	static TrieNode* deserialize(const unsigned char *input, long long size);
};

