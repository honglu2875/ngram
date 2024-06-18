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
#include <cstdio>
#include <vector>
#include <atomic>
#include <omp.h>
#include <signal.h>
#include "nodes.h"


TrieNode::TrieNode() : _count(0) { 
	_preallocated = false;
	omp_init_lock(&lock); 
}

TrieNode::TrieNode(vocab_size_t vocabSize) : TrieNode() {
	_preallocated = true;
	for(auto i = 0; i < vocabSize; i++) {
		_children[i] = nullptr;
	}
}

TrieNode::~TrieNode() { 
	omp_destroy_lock(&lock); 
}

void TrieNode::acquire() {
	omp_set_lock(&lock);
}

void TrieNode::release() {
	omp_unset_lock(&lock);
}

void TrieNode::inc() {
	_count++;
}

count_t TrieNode::count() {
	return _count.load();
}

void TrieNode::set_count(count_t c) {
	_count.store(c);
} 

bool TrieNode::contains(vocab_size_t key) {
	if (_preallocated) {
		return _children[key] != nullptr;
	}
	acquire();
	bool exist = _children.find(key) != _children.end();
	release();
	return exist;
}

TrieNode *TrieNode::get(vocab_size_t key) {
	return _children[key];
}

void TrieNode::put(vocab_size_t key, TrieNode *newChild) {
	acquire();
	if (!_preallocated || !this->contains(key))
		_children[key] = newChild;
	release();
}

template <typename T>
void add_elem(std::vector<unsigned char> &buffer, T data) {
	unsigned char *d = static_cast<unsigned char*>(static_cast<void*>(&data));
	for (auto i = 0; i < sizeof(T); i++) {
		buffer.push_back(d[i]);
	}
}

template <typename T>
T read_elem_move(std::vector<unsigned char>::iterator &begin) {
	unsigned char out_byte[sizeof(T)];
	for (auto i = 0; i < sizeof(T); i++) {
		out_byte[i] = *begin++;
	}
	T out;
	memcpy(&out, out_byte, sizeof(T));
	return out;
}

std::vector<vocab_size_t> TrieNode::get_singleton_children() {
	// Singletons are defined as leaves who are visited only once.
	// Packing them together as 2-byte each reduces overhead significantly for 
	// certain types of ngram Trie.
	std::vector<vocab_size_t> output;
	for (auto it = this->_children.begin(); it != this->_children.end(); it++) {
		if (it->second != nullptr && it->second->count() == 1) {
			output.push_back(it->first);	
		}	
	}
	return output;
}

void TrieNode::serialize(std::vector<unsigned char> &output) {
	std::vector<vocab_size_t> singleton = this->get_singleton_children();

	add_elem<count_t>(output, this->count());
	add_elem<vocab_size_t>(output, this->_children.size());
	add_elem<vocab_size_t>(output, singleton.size());

	// Writing singleton first
	for (auto k: singleton) {
		add_elem<vocab_size_t>(output, k);
	}

	// Writing non-singletons
	for (auto it = this->_children.begin(); it != this->_children.end(); it++) {
		TrieNode* child = it->second;
		if (child != nullptr && child->count() != 1) {
			add_elem<vocab_size_t>(output, it->first);	
			child->serialize(output);
		}
	}
}

std::vector<unsigned char> TrieNode::serialize() {
	std::vector<unsigned char> output;
	serialize(output);
	return output;
}

TrieNode* TrieNode::deserialize(std::vector<unsigned char>::iterator &begin, const std::vector<unsigned char>::iterator &end) {
	if (begin == end) {
		return nullptr;
	}

	count_t count = read_elem_move<count_t>(begin);
	vocab_size_t tot_size = read_elem_move<vocab_size_t>(begin);
	vocab_size_t singleton_size = read_elem_move<vocab_size_t>(begin);

	TrieNode* new_node = new TrieNode();
	new_node->set_count(count);

	// Reconstruct singleton leaves
	for (auto i = 0; i < singleton_size; i++) {
		vocab_size_t key = read_elem_move<vocab_size_t>(begin);
		TrieNode* new_child = new TrieNode();
		new_child->inc();
		new_node->put(key, new_child);
	}

	// Reconstruct other branches
	for (auto i = 0; i < tot_size - singleton_size; i++) {
		vocab_size_t key = read_elem_move<vocab_size_t>(begin);	
		TrieNode* new_child = deserialize(begin, end);
		new_node->put(key, new_child);
	}

	return new_node;
}

TrieNode* TrieNode::deserialize(std::vector<unsigned char> input) {
	auto it = input.begin();
	return deserialize(it, input.end());
}

TrieNode* TrieNode::deserialize(const unsigned char* input, long long size) {
	std::vector<unsigned char> vt_input(input, input + size);
	return deserialize(vt_input);
}
