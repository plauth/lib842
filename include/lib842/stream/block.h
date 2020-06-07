#ifndef LIB842_STREAM_BLOCK_H
#define LIB842_STREAM_BLOCK_H

#include <lib842/detail/free_unique_ptr.h>
#include <cstdint>
#include <cstddef>
#include <array>

namespace lib842 {

namespace stream {

struct Block {
	size_t offset;
	// Data for each (possibly compressed) chunk in the block
	std::array<const uint8_t *, NUM_CHUNKS_PER_BLOCK> datas;
	// Size for each (possibly compressed) chunk in the block
	std::array<size_t, NUM_CHUNKS_PER_BLOCK> sizes;

	// Buffer that owns the pointers used in 'compressed_data'. Used internally.
	detail::free_unique_ptr<const uint8_t> compress_buffer;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_BLOCK_H
