#ifndef LIB842_STREAM_COMP_H
#define LIB842_STREAM_COMP_H

// High-performance compressor for real-time streaming data
// (e.g. data coming from the network)

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/detail/barrier.h>
#include <lib842/detail/latch.h>

#include <lib842/stream/common.h>
#include <lib842/common.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <array>
#include <vector>

#include <cstdint>
#include <memory>
#include <functional>
#include <ostream>

namespace lib842 {

namespace stream {

class DataCompressionStream {
public:
	struct compress_block {
		// Offset into the source buffer where the data associated with the block comes from
		size_t source_offset;
		// Data for each (possibly compressed) chunk in the block
		std::array<const uint8_t *, NUM_CHUNKS_PER_BLOCK> datas;
		// Size for each (possibly compressed) chunk in the block
		std::array<size_t, NUM_CHUNKS_PER_BLOCK> sizes;

		// Buffer that owns the pointers used in 'datas'. Used internally.
		std::unique_ptr<uint8_t[]> compress_buffer;
	};

	DataCompressionStream(lib842_compress_func compress842_func,
			      unsigned int num_threads,
			      thread_policy thread_policy_,
			      std::function<std::ostream&(void)> error_logger,
			      std::function<std::ostream&(void)> debug_logger);
	~DataCompressionStream();

	/* Blocks until the stream is ready to actually start processing data
	   (the underlying threads have been spawned).
	   This isn't only for debugging and benchmarking */
	void wait_until_ready();

	void start(const void *ptr, size_t size, bool skip_compress_step,
		   std::function<void(compress_block &&)> block_available_callback);
	void finalize(bool cancel, std::function<void(bool)> finalize_callback);

	compress_block handle_block(size_t offset);

private:
	void loop_compress_thread(size_t thread_id);

	lib842_compress_func _compress842_func;
	std::function<std::ostream&(void)> _error_logger;
	std::function<std::ostream&(void)> _debug_logger;

	// Instance of the compression threads
	std::vector<std::thread> _threads;
	// Latch that is signaled once all threads have actually been spawned
	detail::latch _threads_ready;
	// Mutex for protecting concurrent accesses to
	// (_trigger, _error, _finalizing, _finalize_callback, _quit)
	std::mutex _mutex;
	// Number that increases if a new operation must be started in the compression threads
	unsigned _trigger;
	// Wakes up the compression threads when a new operation must be started
	std::condition_variable _trigger_changed;

	// Parameters for the compression operation in course
	const void *_ptr;
	size_t _size;
	bool _skip_compress_step;
	std::function<void(compress_block &&)> _block_available_callback;
	// Stores the offset of the next block to be compressed
	std::atomic<size_t> _current_offset;
	// Set to true if an error happens during 842 compression
	bool _error;

	// Set to true when the user wants to be notified when the queue is empty
	bool _finalizing;
	// Callback to be called after finalizing is done
	std::function<void(bool)> _finalize_callback;
	// Barrier for finalizing a compression operation, necessary for
	// ensuring all pending blocks are processed before notifying the user
	detail::barrier _finalize_barrier;

	// If set to true, causes the compression threads to quit (for cleanup)
	bool _quit;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_COMP_H
