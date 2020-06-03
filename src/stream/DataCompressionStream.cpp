#include "numa_spread.h"

#include <lib842/stream/comp.h>

#include <algorithm>
#include <stdexcept>
#include <climits>
#include <cerrno>
#include <stdlib.h> // Hacky use of C11 aligned_alloc,
		    // since std::aligned_alloc is not available until C++17

// A big offset value so all threads stop processing new work
// as soon as possible, i.e. to cancel all pending work of the operation
static size_t BIG_OFFSET_TO_STOP_NEW_WORK = static_cast<size_t>(1) << (sizeof(size_t)*8-1);

namespace lib842 {

namespace stream {

DataCompressionStream::DataCompressionStream(
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
	_ptr(nullptr), _size(0), _skip_compress_step(false),
	_current_offset(0), _error(false),
	_finalizing(false), _finalize_barrier(num_threads),
	_quit(false), _offset_sync_epoch_multiple_log2(UINT_MAX) {
	if ((CHUNK_SIZE % impl842.required_alignment) != 0) {
		_error_logger() << "CHUNK_SIZE must be a multiple of the required 842 alignment" << std::endl;
		throw std::runtime_error("CHUNK_SIZE must be a multiple of the required 842 alignment");
	}
	_threads.reserve(num_threads);
	for (size_t i = 0; i < num_threads; i++)
		_threads.emplace_back(&DataCompressionStream::loop_compress_thread, this, i);
	if (thread_policy_ == thread_policy::spread_threads_among_numa_nodes)
		spread_threads_among_numa_nodes(_threads);
}

DataCompressionStream::~DataCompressionStream() {
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_quit = true;
		// Set a big offset value so all threads stop processing new work soon
		_current_offset = BIG_OFFSET_TO_STOP_NEW_WORK;
		_trigger_changed.notify_all();
		_finalize_barrier.interrupt();
	}

	for (auto &t : _threads)
		t.join();
}

void DataCompressionStream::set_offset_sync_epoch_multiple(size_t offset_sync_epoch_multiple) {
	assert(offset_sync_epoch_multiple % BLOCK_SIZE == 0 &&
	       (offset_sync_epoch_multiple & (offset_sync_epoch_multiple - 1)) == 0);

	_offset_sync_epoch_multiple_log2 = UINT_MAX;
	while (offset_sync_epoch_multiple > 0) {
		_offset_sync_epoch_multiple_log2++;
		offset_sync_epoch_multiple >>= 1;
	}
}

void DataCompressionStream::wait_until_ready() {
	_threads_ready.wait();
}

void DataCompressionStream::start(
	const void *ptr, size_t size, bool skip_compress_step,
	std::function<void(compress_block &&)> block_available_callback) {
	_ptr = ptr;
	_size = size;
	_skip_compress_step = skip_compress_step;
	_block_available_callback = std::move(block_available_callback);

	std::lock_guard<std::mutex> lock(_mutex);
	_trigger++;
	_trigger_changed.notify_all();
}

void DataCompressionStream::finalize(bool cancel, std::function<void(bool)> finalize_callback) {
	std::lock_guard<std::mutex> lock(_mutex);
	_finalizing = true;
	_finalize_callback = std::move(finalize_callback);
	if (cancel)
		_current_offset = BIG_OFFSET_TO_STOP_NEW_WORK;
	_trigger_changed.notify_all();
}

void DataCompressionStream::loop_compress_thread(size_t thread_id) {
	stats_per_thread_t stats;
#ifdef LIB842_STREAM_INDEPTH_TRACE
	_debug_logger()
		<< "Start compression thread with id " << thread_id
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

		// -----------------
		// COMPRESSION PHASE
		// -----------------
		size_t last_offset = 0;

		auto last_valid_offset = _size & ~(BLOCK_SIZE-1);
		while (true) {
			size_t offset = _current_offset.fetch_add(BLOCK_SIZE);

			// If (last_offset / _offset_sync_epoch_multiple) and (offset / _offset_sync_epoch_multiple)
			// are different, stop compressing until all threads are on the same level
			if (_offset_sync_epoch_multiple_log2 != UINT_MAX) {
				size_t num_epochs = (std::min(offset, _size) >> _offset_sync_epoch_multiple_log2) -
						    (last_offset >> _offset_sync_epoch_multiple_log2);
				for (size_t i = 0; i < num_epochs; i++)
					_finalize_barrier.arrive_and_wait();
			}
			last_offset = offset;

			if (offset >= last_valid_offset)
				break;

#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.handled_blocks++;
			auto stat_block_start_time = std::chrono::steady_clock::now();
#endif
			compress_block block = handle_block(offset, stats);
#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.block_duration += std::chrono::steady_clock::now() - stat_block_start_time;
#endif
			if (block.source_offset == SIZE_MAX) {
				bool first_error;
				{
					std::lock_guard<std::mutex> lock(_mutex);
					first_error = !_error;
					_error = true;
					_current_offset = BIG_OFFSET_TO_STOP_NEW_WORK;
				}


				if (first_error) {
					_error_logger()
						<< "Data compression failed, aborting operation"
						<< std::endl;
				}
			}

			// NB: This can push 'spurious' correct or incorrect blocks after an error
			// The user is supposed to ignore those himself
			_block_available_callback(std::move(block));
		}

		{
			std::unique_lock<std::mutex> lock(_mutex);
			_trigger_changed.wait(lock, [this] { return _finalizing || _quit; });
		}

		// ------------------
		// FINALIZATION PHASE
		// ------------------

		// Wait until all threads have finished any compression operations
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
					_ptr = nullptr;
					_size = 0;
					_skip_compress_step = false;
					_block_available_callback = std::function<void(compress_block &&)>();
					_current_offset = 0;
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
		<< "End compression thread with id " << thread_id << " (stats: "
		<< "handled_blocks=" << stats.handled_blocks << ", "
		<< "thread_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.thread_duration).count() << ", "
		<< "woken_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.woken_duration).count() << ", "
		<< "block_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.block_duration).count() << ", "
		<< "compress_duration (ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(stats.compress_duration).count() << ")"
		<< std::endl;
#endif
}


DataCompressionStream::compress_block DataCompressionStream::handle_block(size_t offset,
									  stats_per_thread_t &stats) {
	compress_block block;
	block.source_offset = offset;
	if (_skip_compress_step) {
		for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
			auto source = static_cast<const uint8_t *>(_ptr) + offset + i * CHUNK_SIZE;

			auto is_compressed = std::equal(source,source + sizeof(LIB842_COMPRESSED_CHUNK_MARKER), LIB842_COMPRESSED_CHUNK_MARKER);

			auto chunk_buffer_size = is_compressed
				 ? *reinterpret_cast<const uint64_t *>((source + sizeof(LIB842_COMPRESSED_CHUNK_MARKER)))
				: CHUNK_SIZE;
			auto chunk_buffer = is_compressed
					? source + CHUNK_SIZE - chunk_buffer_size
					: source;

			block.datas[i] = chunk_buffer;
			block.sizes[i] = chunk_buffer_size;
		}
	} else {
		// TODOXXX use chunked mode

		// TODOXXX: This can be reduced to e.g. COMPRESSIBLE_THRESHOLD or CHUNK_SIZE,
		// as long as the lib842 compressor respects the destination buffer size
		// (input value of the olen parameter to the compression function)
		// However, currently the serial_optimized lib842 implementation does not
		// respect olen when compiled without ENABLE_ERROR_HANDLING (for performance),
		// so this is necessary to handle this case
		static constexpr size_t CHUNK_PADDING = 2 * CHUNK_SIZE;
		block.compress_buffer.reset(static_cast<uint8_t *>(aligned_alloc(
			_impl842.preferred_alignment, CHUNK_PADDING * NUM_CHUNKS_PER_BLOCK)));

		bool any_compressible = false;
		for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
			auto source = static_cast<const uint8_t *>(_ptr) + offset + i * CHUNK_SIZE;
			auto compressed_destination = block.compress_buffer.get() + i * CHUNK_PADDING;

			// Compress chunk
			size_t compressed_size = CHUNK_PADDING;

#ifdef LIB842_STREAM_INDEPTH_TRACE
			auto stat_compress_start_time = std::chrono::steady_clock::now();
#endif
			int ret = _impl842.compress(source, CHUNK_SIZE, compressed_destination, &compressed_size);
#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.compress_duration += std::chrono::steady_clock::now() - stat_compress_start_time;
#endif
			if (ret != 0 && ret != -ENOSPC) {
				block.source_offset = SIZE_MAX; // Indicates error to the user
				break;
			}

			// Push into the chunk queue
			auto compressible = ret == 0 && compressed_size <= COMPRESSIBLE_THRESHOLD;

			block.datas[i] = compressible ? compressed_destination : source;
			block.sizes[i] = compressible ? compressed_size : CHUNK_SIZE;
			any_compressible |= compressible;
		}

		// If no chunk is compressible, release the unused compression buffer now
		if (!any_compressible)
			block.compress_buffer.reset();
	}
	return block;
}

} // namespace stream

} // namespace lib842
