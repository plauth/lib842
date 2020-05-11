#ifndef LIB842_DETAIL_BARRIER_H
#define LIB842_DETAIL_BARRIER_H

// A standalone implementation of a barrier for thread coordination,
// similar to C++20's std::barrier

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <mutex>
#include <condition_variable>
#include <cstddef>

namespace lib842 {

namespace detail {

class barrier {
public:
	explicit barrier(std::ptrdiff_t num_threads) :
		_num_threads(num_threads),
		_count(0),
		_generation(0)
	{}


	void arrive_and_wait() {
		std::unique_lock<std::mutex> lock(_mutex);
		if (++_count == _num_threads) {
			_generation++;
			_count = 0;
			_cv.notify_all();
		} else {
			auto current_generation = _generation;
			_cv.wait(lock, [this, current_generation] {
				return _generation != current_generation;
			});
		}
	}

private:
	std::mutex _mutex;
	std::condition_variable _cv;
	std::ptrdiff_t _num_threads;
	std::ptrdiff_t _count;
	unsigned int _generation;
};

} // namespace detail

} // namespace lib842

#endif // LIB842_DETAIL_BARRIER_H
