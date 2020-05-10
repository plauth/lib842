#ifndef LIB842_DETAIL_BARRIER_H
#define LIB842_DETAIL_BARRIER_H

// A standalone implementation of a barrier for thread coordination,
// similar to Boost's boost::barrier or C++20's std::barrier

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <mutex>
#include <condition_variable>

namespace lib842 {

class barrier {
public:
    barrier(unsigned int num_threads) :
        _num_threads(num_threads),
        _count(0),
        _generation(0)
    {}


    void wait() {
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
    unsigned int _num_threads;
    unsigned int _count;
    unsigned int _generation;
};

} // namespace lib842

#endif // LIB842_DETAIL_BARRIER_H
