#include "numa_spread.h"

#include <lib842/stream/comp.h>

#include <array>
#include <algorithm>
#include <stdexcept>
#include <climits>
#include <cerrno>

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
	_ptr(nullptr), _size(0),
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
	const void *ptr, size_t size,
	std::function<void(Block &&)> block_available_callback) {
	_ptr = ptr;
	_size = size;
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
			Block block = handle_block(offset, stats);
#ifdef LIB842_STREAM_INDEPTH_TRACE
			stats.block_duration += std::chrono::steady_clock::now() - stat_block_start_time;
#endif
			if (block.offset == SIZE_MAX) {
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
					_block_available_callback = std::function<void(Block &&)>();
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


Block DataCompressionStream::handle_block(size_t offset, stats_per_thread_t &stats) const {
	// Use a padding of CHUNK_SIZE instead of COMPRESSIBLE_THRESHOLD to
	// hopefully get better alignment for HW compression
	// (CHUNK_SIZE is guaranteed to be a power of two)
	static constexpr size_t CHUNK_PADDING = CHUNK_SIZE; // COMPRESSIBLE_THRESHOLD;

	Block block;
	block.source = static_cast<const uint8_t *>(_ptr) + offset;
	block.chunk_padding = CHUNK_PADDING;
	block.offset = offset;

	uint8_t *chunk_buffer = block.allocate_buffer(
		_impl842.preferred_alignment, CHUNK_PADDING);
	std::array<int, NUM_CHUNKS_PER_BLOCK> chunk_rets;
	std::array<size_t, NUM_CHUNKS_PER_BLOCK> input_chunk_sizes, output_chunk_sizes;
	input_chunk_sizes.fill(CHUNK_SIZE);
	output_chunk_sizes.fill(CHUNK_PADDING);

#ifdef LIB842_STREAM_INDEPTH_TRACE
	auto stat_compress_start_time = std::chrono::steady_clock::now();
#endif
	int ret = _impl842.compress_chunked(NUM_CHUNKS_PER_BLOCK, chunk_rets.data(),
		static_cast<const uint8_t *>(_ptr) + offset, CHUNK_SIZE, input_chunk_sizes.data(),
		chunk_buffer, CHUNK_PADDING, output_chunk_sizes.data());
#ifdef LIB842_STREAM_INDEPTH_TRACE
	stats.compress_duration += std::chrono::steady_clock::now() - stat_compress_start_time;
#endif
	if (ret != 0) {
		block.offset = SIZE_MAX; // Indicates error to the user
		block.release_buffer();
		return block;
	}

	bool any_compressible = false;
	for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
		auto compressible = chunk_rets[i] == 0 && output_chunk_sizes[i] <= COMPRESSIBLE_THRESHOLD;
		block.sizes[i] = compressible ? output_chunk_sizes[i] : CHUNK_SIZE;
		any_compressible |= compressible;
	}

	// If no chunk is compressible, release the unused compression buffer now
	if (!any_compressible)
		block.release_buffer();

	return block;
}

} // namespace stream

} // namespace lib842
