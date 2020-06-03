#ifndef LIB842_STREAM_DECOMP_H
#define LIB842_STREAM_DECOMP_H

// High-performance decompressor for real-time streaming data
// (e.g. data coming from the network)

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/detail/barrier.h>
#include <lib842/detail/latch.h>
#include <lib842/detail/free_unique_ptr.h>

#include <lib842/stream/common.h>
#include <lib842/common.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#include <array>
#include <vector>
#include <queue>

#include <functional>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <ostream>
#ifdef LIB842_STREAM_INDEPTH_TRACE
#include <chrono>
#endif

namespace lib842 {

namespace stream {

class DataDecompressionStream {
public:
	struct decompress_chunk {
		const uint8_t *compressed_data;
		size_t compressed_length;
		void *destination;

		// Disable default copy constructor/assignment to prevent accidental performance hit
		decompress_chunk() :
			compressed_data(nullptr), compressed_length(0), destination(nullptr) { }
		decompress_chunk(const uint8_t *compressed_data, size_t compressed_length, void *destination) :
			compressed_data(compressed_data), compressed_length(compressed_length), destination(destination) { }
		decompress_chunk(const decompress_chunk &) = delete;
		decompress_chunk& operator=(const decompress_chunk &) = delete;
		decompress_chunk(decompress_chunk &&) = default;
		decompress_chunk& operator=(decompress_chunk &&) = default;
	};

	struct decompress_block {
		std::array<decompress_chunk, NUM_CHUNKS_PER_BLOCK> chunks;

		// Buffer that owns the pointers used in 'compressed_data'. Used internally.
		detail::free_unique_ptr<const uint8_t> compress_buffer;
	};

	DataDecompressionStream(const lib842_implementation &impl842,
				unsigned int num_threads,
				thread_policy thread_policy_,
				std::function<std::ostream&(void)> error_logger,
				std::function<std::ostream&(void)> debug_logger);
	~DataDecompressionStream();

	/* Blocks until the stream is ready to actually start processing data
	   (the underlying threads have been spawned).
	   This isn't only for debugging and benchmarking */
	void wait_until_ready();

	/* Starts a new decompression operation. */
	void start();
	/* Enqueues a new to be decompressed */
	bool push_block(DataDecompressionStream::decompress_block &&dm);
	/* Wait for the decompression queue to be cleared up and then call the specified callback.
	 * If cancel = false, the decompression queue will be fully processed before
	 *                    invoking the callback (unless an error happens).
	 * If cancel = true, the decompression operation will be finished as soon as possible,
	 *                   possibly dropping most or all of the decompression queue.
	 * The parameter of the callback specifies a success (true) / error (false) status. */
	void finalize(bool cancel, std::function<void(bool)> finalize_callback);

private:
	struct stats_per_thread_t {
#ifdef LIB842_STREAM_INDEPTH_TRACE
		size_t handled_blocks = 0;
		std::chrono::steady_clock::duration thread_duration{0};
		std::chrono::steady_clock::duration woken_duration{0};
		std::chrono::steady_clock::duration block_duration{0};
		std::chrono::steady_clock::duration decompress_duration{0};
#endif
	};

	void loop_decompress_thread(size_t thread_id);
	bool handle_block(const decompress_block &block,
			  stats_per_thread_t &stats);

	const lib842_implementation &_impl842;
	std::function<std::ostream&(void)> _error_logger;
	std::function<std::ostream&(void)> _debug_logger;

	// Instance of the decompression threads
	std::vector<std::thread> _threads;
	// Latch that is signaled once all threads have actually been spawned
	detail::latch _threads_ready;
	// Mutex for protecting concurrent accesses to
	// (_trigger, _queue, _error, _finalizing, _finalize_callback, _quit)
	std::mutex _mutex;

	// Number that increases if a new operation must be started in the decompression threads
	unsigned _trigger;
	// Wakes up the decompression threads when a new operation must be started
	std::condition_variable _trigger_changed;

	// Stores blocks pending to be decompressed
	std::queue<decompress_block> _queue;
	// Wakes up the decompression threads when new operations have been added to the queue
	std::condition_variable _queue_available;
	// Set to true if an error happens during 842 decompression
	bool _error;

	// Set to true when the user wants to be notified when the queue is empty
	bool _finalizing;
	// Callback to be called after finalizing is done
	std::function<void(bool)> _finalize_callback;
	// Barrier for finalizing a decompression operation, necessary for
	// ensuring all pending blocks are processed before notifying the user
	detail::barrier _finalize_barrier;

	// If set to true, causes the decompression threads to quit (for cleanup)
	bool _quit;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_DECOMP_H
