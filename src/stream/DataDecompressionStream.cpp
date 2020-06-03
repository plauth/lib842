#include "numa_spread.h"

#include <lib842/stream/decomp.h>

#include <cassert>

namespace lib842 {

namespace stream {

DataDecompressionStream::DataDecompressionStream(
	const lib842_implementation &impl842,
	unsigned int num_threads,
	thread_policy thread_policy_,
	std::function<std::ostream&(void)> error_logger,
	std::function<std::ostream&(void)> debug_logger) :
	_impl842(impl842),
	_error_logger(std::move(error_logger)),
	_debug_logger(std::move(debug_logger)),
	_threads_ready(num_threads),
	_trigger(0),
	_error(false),
	_finalizing(false), _finalize_barrier(num_threads),
	_quit(false) {
	if ((CHUNK_SIZE % impl842.required_alignment) != 0) {
		_error_logger() << "CHUNK_SIZE must be a multiple of the required 842 alignment" << std::endl;
		throw std::runtime_error("CHUNK_SIZE must be a multiple of the required 842 alignment");
	}
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
	_trigger++;
	_trigger_changed.notify_all();
}

bool DataDecompressionStream::push_block(DataDecompressionStream::decompress_block &&dm) {
	std::lock_guard<std::mutex> lock(_mutex);
	assert(!_finalizing);
	if (_error) {
		// If an error happened, report the error to the user
		// (but the user must still finalize the operation itself,
		//  to make sure no threads are still working on the blocks)
		return false;
	}

	_queue.push(std::move(dm));
	_queue_available.notify_one();
	return true;
}

void DataDecompressionStream::finalize(bool cancel, std::function<void(bool)> finalize_callback) {
	std::lock_guard<std::mutex> lock(_mutex);
	_finalizing = true;
	_finalize_callback = std::move(finalize_callback);
	if (cancel)
		_queue = std::queue<decompress_block>();
	_queue_available.notify_all();
}

void DataDecompressionStream::loop_decompress_thread(size_t thread_id) {
	stats_per_thread_t stats;
#ifdef LIB842_STREAM_INDEPTH_TRACE
	_debug_logger()
		<< "Start decompression thread with id " << thread_id
		<< std::endl;
	auto stat_thread_start_time = std::chrono::steady_clock::now();
#endif

	_threads_ready.count_down();

	unsigned last_trigger = 0;
	while (!_quit) {
		// -------------
		// TRIGGER PHASE
		// -------------
		{
			std::unique_lock<std::mutex> lock(_mutex);
			_trigger_changed.wait(lock, [this, &last_trigger] {
				return _trigger != last_trigger || _quit;
			});
			last_trigger = _trigger;
		}

#ifdef LIB842_STREAM_INDEPTH_TRACE
		auto stat_woken_start_time = std::chrono::steady_clock::now();
#endif

		// -------------------
		// DECOMPRESSION PHASE
		// -------------------
		while (true) {
			// (Blocking) pop from the chunk queue
			decompress_block block;

			{
				std::unique_lock<std::mutex> lock(_mutex);
				_queue_available.wait(lock, [this] {
					return !_queue.empty() || _finalizing || _quit;
				});
				if ((_finalizing && _queue.empty()) || _quit) {
					break;
				}

				block = std::move(_queue.front());
				_queue.pop();
			}

#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.handled_blocks++;
			auto stat_block_start_time = std::chrono::steady_clock::now();
#endif
			auto block_success = handle_block(block, stats);
#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.block_duration += std::chrono::steady_clock::now() - stat_block_start_time;
#endif
			if (!block_success) {
				bool first_error;
				{
					std::lock_guard<std::mutex> lock(_mutex);
					first_error = !_error;
					_error = true;
					_queue = std::queue<decompress_block>();
				}

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

		// Wait until all threads have finished any decompression operations
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
					_error = false;
					_finalizing = false;
					finalize_callback = std::move(_finalize_callback);
					_finalize_callback = std::function<void(bool)>();
				}
			}
			if (!quit)
				finalize_callback(!error);
		}

#ifdef LIB842_STREAM_INDEPTH_TRACE
		stats.woken_duration += std::chrono::steady_clock::now() - stat_woken_start_time;
#endif
	}

#ifdef LIB842_STREAM_INDEPTH_TRACE
	stats.thread_duration += std::chrono::steady_clock::now() - stat_thread_start_time;
	_debug_logger()
		<< "End decompression thread with id " << thread_id << " (stats: "
		<< "handled_blocks=" << stats.handled_blocks << ", "
		<< "thread_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.thread_duration).count() << ", "
		<< "woken_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.woken_duration).count() << ", "
		<< "block_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.block_duration).count() << ", "
		<< "decompress_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.decompress_duration).count() << ")"
		<< std::endl;
#endif
}

bool DataDecompressionStream::handle_block(const decompress_block &block,
					   stats_per_thread_t &stats) {
	// TODOXXX use chunked mode
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
#ifdef LIB842_STREAM_INDEPTH_TRACE
		auto stat_decompress_start_time = std::chrono::steady_clock::now();
#endif
		int ret = _impl842.decompress(chunk.compressed_data,
					      chunk.compressed_length,
					      destination, &uncompressed_size);
#ifdef LIB842_STREAM_INDEPTH_TRACE
		stats.decompress_duration += std::chrono::steady_clock::now() - stat_decompress_start_time;
#endif
		if (ret != 0)
			return false;

		assert(uncompressed_size == CHUNK_SIZE);
	}

	return true;
}

} // namespace stream

} // namespace lib842
