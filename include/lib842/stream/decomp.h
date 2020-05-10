#ifndef LIB842_STREAM_DECOMP_H
#define LIB842_STREAM_DECOMP_H

// High-performance decompressor for real-time streaming data
// (e.g. data coming from the network)

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/detail/barrier.h>

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
		std::array<decompress_chunk, NUM_CHUNKS_PER_NETWORK_BLOCK> chunks;

		// Buffer that owns the pointers used in 'compressed_data'. Used internally.
		std::unique_ptr<const uint8_t[]> compress_buffer;
	};

	DataDecompressionStream(lib842_decompress_func decompress842_func,
				unsigned int num_threads,
				std::function<std::ostream&(void)> error_logger,
				std::function<std::ostream&(void)> debug_logger);
	~DataDecompressionStream();
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
	void finalize(bool cancel, const std::function<void(bool)> &finalize_callback);

private:
	void loop_decompress_thread(size_t thread_id);

	enum class decompress_state {
		// The decompressor is working (or waiting for new blocks)
		processing,
		// The decompressor is working, but the user of this class is waiting for the
		// remaining decompression blocks to be processed.
		finalizing,
		// The decompressor is working, but the user of this class has ordered that the
		// current decompression process should be cancelled.
		cancelling,
		// The decompressor has found an error during decompression and the current
		// decompression process is being cancelled
		handling_error,
		// The thread pool is being destroyed
		quitting
	};

	lib842_decompress_func _decompress842_func;
	std::function<std::ostream&(void)> _error_logger;
	std::function<std::ostream&(void)> _debug_logger;

	// Instance of the decompression threads
	std::vector<std::thread> _threads;
	// Mutex for protecting concurrent accesses to
	// (_state, _queue, _report_error, _working_thread_count)
	std::mutex _queue_mutex;
	// Stores the current action being performed by the threads
	decompress_state _state;
	// Stores pending decompression operations
	std::queue<decompress_block> _queue;
	// Indicates that a decompression error happened and the user of this class should be notified
	bool _report_error;
	// Number of threads currently running decompression operations
	unsigned int _working_thread_count;
	// Callback to be called after finalizing or cancelling is done
	std::function<void(bool)> _finalize_callback;
	// Wakes up the decompression threads when new operations have been added to the queue
	std::condition_variable _queue_available;
	// Barrier for finishing decompression, necessary for ensuring that resources
	// are not released until all threads have finished
	barrier _finish_barrier;
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_DECOMP_H
