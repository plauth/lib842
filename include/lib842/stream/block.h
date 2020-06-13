#ifndef LIB842_STREAM_BLOCK_H
#define LIB842_STREAM_BLOCK_H

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/detail/free_unique_ptr.h>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <array>
#include <new>

#include <stdlib.h> // Hacky use of C11 aligned_alloc,
		    // since std::aligned_alloc is not available until C++17

namespace lib842 {

namespace stream {

struct Block {
	// Offset (in the original data) corresponding to the data in the block
	size_t offset;
	// Separation between chunks in the chunk_buffer
	size_t chunk_padding;
	// Pointer to original source data (set when the block is compressed, can be nullptr)
	const void *source;
	// Data for each (possibly compressed) chunk in the block
	//std::array<const uint8_t *, NUM_CHUNKS_PER_BLOCK> datas;
	// Size for each (possibly compressed) chunk in the block
	std::array<size_t, NUM_CHUNKS_PER_BLOCK> sizes;

	uint8_t *allocate_buffer(size_t alignment, size_t chunk_padding) {
		uint8_t *ptr = static_cast<uint8_t *>(aligned_alloc(
			alignment, chunk_padding * NUM_CHUNKS_PER_BLOCK));
		if (ptr == nullptr)
			throw std::bad_alloc();
		assert(chunk_buffer.get() == nullptr);
		chunk_buffer.reset(ptr);
		return ptr;
	}

	void release_buffer() {
		chunk_buffer.reset();
	}

	// Get the pointer to (potentially uncompressed) data corresponding to a chunk
	const uint8_t *get_chunk(size_t chunk_no) const {
		if (sizes[chunk_no] == 0) // Chunk not present ('gap')
			return nullptr;
		if (sizes[chunk_no] == CHUNK_SIZE) { // Uncompressible chunk - read from source
			assert(source == nullptr);
			return static_cast<const uint8_t *>(source) + chunk_no * CHUNK_SIZE;
		}
		return chunk_buffer.get() + chunk_no * chunk_padding;
	}

	// Get a pointer to the beginning of the chunk buffer
	const uint8_t *get_chunk_buffer() const {
		return chunk_buffer.get();
	}

private:
	// This is a buffer associated with the block, where the compressed
	// chunks are stored. Using a unique_ptr here facilitates management
	// of the lifetime of this (so it is released alongside the block)
	detail::free_unique_ptr<const uint8_t> chunk_buffer;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_BLOCK_H
