// TODOXXX: Should add the code to spread the threads among NUMA zones for HW? (From lib842 sample)

#include <lib842/stream/decomp.h>

#include <cassert>

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

namespace lib842 {

namespace stream {

DataDecompressionStream::DataDecompressionStream(
    lib842_decompress_func decompress842_func,
    unsigned int num_threads,
    std::function<std::ostream&(void)> error_logger,
    std::function<std::ostream&(void)> debug_logger) :
    _decompress842_func(decompress842_func),
    _error_logger(std::move(error_logger)),
    _debug_logger(std::move(debug_logger)),
    _state(decompress_state::processing),
    _working_thread_count(0),
    _finish_barrier(num_threads),
    _report_error(false) {
    _threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++)
        _threads.emplace_back(&DataDecompressionStream::loop_decompress_thread, this, i);
}

DataDecompressionStream::~DataDecompressionStream() {
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _state = decompress_state::quitting;
        _queue_available.notify_all();
    }
    for (auto &t : _threads)
        t.join();
}

void DataDecompressionStream::start() {
    _working_thread_count = 0;
}

bool DataDecompressionStream::push_block(DataDecompressionStream::decompress_block &&dm) {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    if (_report_error) {
        _report_error = false;
        return false;
    }

    _queue.push(std::move(dm));
    _queue_available.notify_one();
    return true;
}

void DataDecompressionStream::finalize(bool cancel, const std::function<void(bool)> &finalize_callback) {
    bool done, report_error;
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        done = _state == decompress_state::processing &&
               _working_thread_count == 0 && _queue.empty();
        // If there are still decompression operations active, we need to wait
        // until they finish to finalize the entire operation
        // In this case, transfer the responsibility of finalizing to the decompression threads
        if (!done) {
            _state = cancel ? decompress_state::cancelling : decompress_state::finalizing;
            _finalize_callback = finalize_callback;
            _queue_available.notify_all();
        }

        report_error = _report_error;
        _report_error = false;
    }

    // Otherwise, if all decompression threads finished as well,
    // finalize the entire operation as soon as possible here
    if (done) {
        finalize_callback(!report_error);
    }
}

void DataDecompressionStream::loop_decompress_thread(size_t thread_id) {
#ifdef INDEPTH_TRACE
    _debug_logger()
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start decompression thread with id " << thread_id
        << std::endl;
    size_t stat_handled_blocks = 0;
#endif

    while (true) {
        // (Blocking) pop from the chunk queue
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _queue_available.wait(lock, [this] {
            return !_queue.empty() || _state != decompress_state::processing;
        });
        if (_state == decompress_state::handling_error || _state == decompress_state::cancelling) {
            lock.unlock();

            // Wait until all threads have got the "error" message
            _finish_barrier.wait();

            // "Leader" thread clears the queue
            if (thread_id == 0) {
                lock.lock();
                _queue = {};
                _state = decompress_state::processing;
                _report_error = _state == decompress_state::handling_error;
                lock.unlock();
            }

            // Once write is finalized, wait again
            _finish_barrier.wait();
        } else if (_state == decompress_state::finalizing && _queue.empty()) {
            lock.unlock();

            // Wait until all threads have got the "finalize" message
            _finish_barrier.wait();

            // "Leader" thread finalizes the write
            if (thread_id == 0) {
                lock.lock();
                auto report_error = _report_error;
                _report_error = false;
                _state = decompress_state::processing;
                lock.unlock();
                _finalize_callback(!report_error);
            }

            // Once write is finalized, wait again
            _finish_barrier.wait();
        } else if (_state == decompress_state::quitting) {
            break;
        } else {
            auto block = std::move(_queue.front());
            _queue.pop();
            _working_thread_count++;

            lock.unlock();
#ifdef INDEPTH_TRACE
            stat_handled_blocks++;
#endif
            for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                const auto &chunk = block.chunks[i];
                if (chunk.compressed_data == nullptr && chunk.compressed_length == 0 &&
                    chunk.destination == nullptr) {
                    // Chunk was transferred uncompressed, nothing to do
                    continue;
                }


                auto destination = static_cast<uint8_t *>(chunk.destination);

                assert(chunk.compressed_length > 0 &&
                       chunk.compressed_length <= COMPRESSIBLE_THRESHOLD);

                size_t uncompressed_size = COMPR842_CHUNK_SIZE;
                int ret = _decompress842_func(chunk.compressed_data,
                                              chunk.compressed_length,
                                              destination, &uncompressed_size);
                if (ret != 0) {
                    _error_logger()
                            << "Data decompression failed, aborting operation"
                            << std::endl;

                    lock.lock();
                    _state = decompress_state::handling_error;
                    _queue_available.notify_all();
                    lock.unlock();
                    break;
                }

                assert(uncompressed_size == COMPR842_CHUNK_SIZE);
            }

            lock.lock();
            _working_thread_count--;
            lock.unlock();
        }
    }

#ifdef INDEPTH_TRACE
    _debug_logger()
        << "(DataStream to " << _remote_endpoint << ") "
        << "End decompression thread with id " << thread_id << " (stat_handled_blocks=" << stat_handled_blocks << ")"
        << std::endl;
#endif
}

} // namespace stream

} // namespace lib842
