#ifndef __COMPSTREAM842_H__
#define __COMPSTREAM842_H__

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include "detail/barrier842.h"

#include <commonstream842.h>
#include <common842.h>

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
		std::array<const uint8_t *, NUM_CHUNKS_PER_NETWORK_BLOCK> datas;
		// Size for each (possibly compressed) chunk in the block
		std::array<size_t, NUM_CHUNKS_PER_NETWORK_BLOCK> sizes;

		// Buffer that owns the pointers used in 'datas'. Used internally.
		std::unique_ptr<uint8_t[]> compress_buffer;
	};

	DataCompressionStream(lib842_compress_func compress842_func,
			      unsigned int num_threads,
			      std::function<std::ostream&(void)> error_logger,
			      std::function<std::ostream&(void)> debug_logger);
	~DataCompressionStream();

	void start(const void *ptr, size_t size, bool skip_compress_step,
		   std::function<void(compress_block &&)> block_available_callback);
	void finish(bool cancel);

private:
	void loop_compress_thread(size_t thread_id);

	lib842_compress_func _compress842_func;
	std::function<std::ostream&(void)> _error_logger;
	std::function<std::ostream&(void)> _debug_logger;

	// Instance of the compression thread
	std::vector<std::thread> _threads;
	// Mutex for protecting concurrent accesses to
	// (_trigger, _quit)
	std::mutex _trigger_mutex;
	// true if a new operation must be started in the compression thread
	bool _trigger;
	// Wakes up the compression thread when a new operation must be started
	std::condition_variable _trigger_changed;
	// If set to true, causes the compression to quit (for cleanup)
	bool _quit;
	// Necessary data for triggering an asynchronous I/O write operation from the compression thread
	std::function<void(compress_block &&)> _block_available_callback;
	// Parameters for the compression operation in course
	const void *_ptr;
	size_t _size;
	bool _skip_compress_step;
	// Stores the offset of the next block to be compressed
	std::atomic<std::size_t> _current_offset;
	// true if an error has happened and the compression operation should be cancelled, false otherwise
	std::atomic<bool> _error;
	// Barrier for starting compression, necessary for ensuring that all compression
	// threads have seen the trigger to start compressing before unsetting it
	barrier _start_barrier;
	// Barrier for finishing compression, necessary for ensuring that resources
	// are not released until all threads have finished
	barrier _finish_barrier;
};

} // namespace stream

} // namespace lib842

#endif
