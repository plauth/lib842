#ifndef LIB842_STREAM_BLOCK_H
#define LIB842_STREAM_BLOCK_H

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/detail/free_unique_ptr.h>
#include <cstdint>
#include <cstddef>
#include <array>

#include <stdlib.h> // Hacky use of C11 aligned_alloc,
		    // since std::aligned_alloc is not available until C++17

namespace lib842 {

namespace stream {

struct Block {
	// Offset (in the original data) corresponding to the data in the block
	size_t offset;
	// Data for each (possibly compressed) chunk in the block
	std::array<const uint8_t *, NUM_CHUNKS_PER_BLOCK> datas;
	// Size for each (possibly compressed) chunk in the block
	std::array<size_t, NUM_CHUNKS_PER_BLOCK> sizes;

	uint8_t *allocate_buffer(size_t alignment, size_t size) {
		uint8_t *ptr = static_cast<uint8_t *>(aligned_alloc(alignment, size));
		assert(chunk_buffer.get() == nullptr);
		chunk_buffer.reset(ptr);
		return ptr;
	}

	void release_buffer() {
		chunk_buffer.reset();
	}

private:
	// This is a buffer associated with the block. Its purpose is to hold
	// (if necessary) the pointers in 'datas', and facilitate management
	// of the lifetime of this (so it is released alongside the block)
	detail::free_unique_ptr<const uint8_t> chunk_buffer;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_BLOCK_H
