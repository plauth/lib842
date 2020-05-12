#include "numa_spread.h"

#include <lib842/stream/decomp.h>

#include <cassert>

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

namespace lib842 {

namespace stream {

DataDecompressionStream::DataDecompressionStream(
	lib842_decompress_func decompress842_func,
	unsigned int num_threads,
	thread_policy thread_policy_,
	std::function<std::ostream&(void)> error_logger,
	std::function<std::ostream&(void)> debug_logger) :
	_decompress842_func(decompress842_func),
	_error_logger(std::move(error_logger)),
	_debug_logger(std::move(debug_logger)),
	_threads_ready(num_threads),
	_trigger(false),
	_trigger_barrier(num_threads),
	_error(false),
	_finalizing(false),
	_finalize_barrier(num_threads),
	_quit(false) {
	_threads.reserve(num_threads);
	for (size_t i = 0; i < num_threads; i++)
		_threads.emplace_back(&DataDecompressionStream::loop_decompress_thread, this, i);
	if (thread_policy_ == thread_policy::spread_threads_among_numa_nodes)
		spread_threads_among_numa_nodes(_threads);
}

DataDecompressionStream::~DataDecompressionStream() {
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_quit = true;
		// Wake up everybody so they see the quick signal
		_trigger_changed.notify_all();
		_queue_available.notify_all();
		// The barriers *must* be interrupted, since otherwise if any
		// thread quits while another is waiting on the barrier, the
		// waiting thrread would wait forever and never quit
		_trigger_barrier.interrupt();
		_finalize_barrier.interrupt();
	}

	for (auto &t : _threads)
		t.join();
}

void DataDecompressionStream::wait_until_ready() {
	_threads_ready.wait();
}

void DataDecompressionStream::start() {
	std::lock_guard<std::mutex> lock(_mutex);
	assert(!_trigger);
	_trigger = true;
	_trigger_changed.notify_all();
}

bool DataDecompressionStream::push_block(DataDecompressionStream::decompress_block &&dm) {
	std::lock_guard<std::mutex> lock(_mutex);
	assert(!_finalizing);
	if (_error) {
		// If an error happened, report the error to the user,
		// and the operation is considered immediately finalized
		// (the user must not finalize the operation itself)
		_finalizing = true;
		_finalize_callback = [](bool){};
		_queue_available.notify_all();
		return false;
	}

	_queue.push(std::move(dm));
	_queue_available.notify_one();
	return true;
}

void DataDecompressionStream::finalize(bool cancel, std::function<void(bool)> finalize_callback) {
	std::lock_guard<std::mutex> lock(_mutex);
	if (cancel)
		_queue = std::queue<decompress_block>();
	_finalizing = true;
	_finalize_callback = std::move(finalize_callback);
	_queue_available.notify_all();
}

void DataDecompressionStream::loop_decompress_thread(size_t thread_id) {
#ifdef INDEPTH_TRACE
	_debug_logger()
		<< "(DataStream to " << _remote_endpoint << ") "
		<< "Start decompression thread with id " << thread_id
		<< std::endl;
	size_t stat_handled_blocks = 0;
#endif

	_threads_ready.count_down();

	while (!_quit) {
		// -------------
		// TRIGGER PHASE
		// -------------
		{
			std::unique_lock<std::mutex> lock(_mutex);
			_trigger_changed.wait(lock, [this] { return _trigger || _quit; });
		}

		_trigger_barrier.arrive_and_wait();
		if (thread_id == 0) { // Only a single 'leader' thread goes in
			std::lock_guard<std::mutex> lock(_mutex);
			// Unset the trigger once all threads have been modified
			// Note that the trigger *must* be unset early, not at the end.
			// Otherwise, since the user can immediately call
			// start() after push_block() reports an error,
			// a trigger for a further operation can be missed
			_trigger = false;
		}
		_trigger_barrier.arrive_and_wait();

		// -------------------
		// DECOMPRESSION PHASE
		// -------------------
		while (true) {
			// (Blocking) pop from the chunk queue
			std::unique_lock<std::mutex> lock(_mutex);
			_queue_available.wait(lock, [this] {
				return !_queue.empty() || _finalizing || _quit;
			});
			if ((_finalizing && _queue.empty()) || _quit) {
				break;
			}

			auto block = std::move(_queue.front());
			_queue.pop();

			lock.unlock();
#ifdef INDEPTH_TRACE
			stat_handled_blocks++;
#endif
			if (!handle_block(block)) {
				lock.lock();
				bool first_error = !_error;
				_error = true;
				_queue = std::queue<decompress_block>();
				lock.unlock();

				if (first_error) {
					_error_logger()
						<< "Data decompression failed, aborting operation"
						<< std::endl;
				}
			}
		}

		// ------------------
		// FINALIZATION PHASE
		// ------------------

		// Wait until all threads have got the "finalize" message
		_finalize_barrier.arrive_and_wait();

		if (thread_id == 0) { // Only a single 'leader' thread goes in
			bool quit, error;
			std::function<void(bool)> finalize_callback;
			{
				std::lock_guard<std::mutex> lock(_mutex);
				// If quiting, do NOT reset any variable, just quit
				// This is important so that if we arrive here due to
				// an interrupted barrier, the error flag does not get unset
				quit = _quit;
				if (!quit) {
					error = _error;
					_finalizing = false;
					_error = false;
					finalize_callback = std::move(_finalize_callback);
					_finalize_callback = std::function<void(bool)>();
				}
			}
			if (!quit)
				finalize_callback(!error);
		}

		// Once write is finalized, wait again
		_finalize_barrier.arrive_and_wait();
	}

#ifdef INDEPTH_TRACE
	_debug_logger()
		<< "(DataStream to " << _remote_endpoint << ") "
		<< "End decompression thread with id " << thread_id << " (stat_handled_blocks=" << stat_handled_blocks << ")"
		<< std::endl;
#endif
}

bool DataDecompressionStream::handle_block(const decompress_block &block) {
	for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
		const auto &chunk = block.chunks[i];
		if (chunk.compressed_data == nullptr && chunk.compressed_length == 0 &&
		    chunk.destination == nullptr) {
			// Chunk was transferred uncompressed, nothing to do
			continue;
		}

		auto destination = static_cast<uint8_t *>(chunk.destination);

		assert(chunk.compressed_length > 0 &&
		       chunk.compressed_length <= COMPRESSIBLE_THRESHOLD);

		size_t uncompressed_size = CHUNK_SIZE;
		int ret = _decompress842_func(chunk.compressed_data,
					      chunk.compressed_length,
					      destination, &uncompressed_size);
		if (ret != 0)
			return false;

		assert(uncompressed_size == CHUNK_SIZE);
	}

	return true;
}

} // namespace stream

} // namespace lib842
