#ifndef LIB842_DETAIL_LATCH_H
#define LIB842_DETAIL_LATCH_H

// A standalone implementation of a latch for thread coordination,
// similar to C++20's std::latch

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <mutex>
#include <condition_variable>
#include <cstddef>
#include <cassert>

namespace lib842 {

namespace detail {

class latch {
public:
	explicit latch(std::ptrdiff_t value) :
		_value(value)
	{}


	void count_down() {
		std::lock_guard<std::mutex> lock(_mutex);
		assert(_value > 0);
		_value--;
		_cv.notify_all();
	}

	void wait() {
		std::unique_lock<std::mutex> lock(_mutex);
		_cv.wait(lock, [this] { return _value == 0; });
	}

private:
	std::mutex _mutex;
	std::condition_variable _cv;
	std::ptrdiff_t _value;
};

} // namespace detail

} // namespace lib842

#endif // LIB842_DETAIL_LATCH_H
