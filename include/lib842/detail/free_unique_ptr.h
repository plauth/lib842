#ifndef LIB842_DETAIL_FREE_UNIQUE_PTR_H
#define LIB842_DETAIL_FREE_UNIQUE_PTR_H

// An implementation of std::unique_ptr which frees memory with C's free
// (for memory from C APIs such as malloc, aligned_alloc, posix_memalign, etc.)
// instead of with delete or delete[]

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <type_traits>
#include <memory>
#include <cstdlib>

namespace lib842 {

namespace detail {


struct free_c_pointer {
	template<typename T>
	void operator()(T *ptr) const {
		std::free(const_cast<typename std::remove_const<T>::type *>(ptr));
	}
};

template<typename T>
using free_unique_ptr = std::unique_ptr<T, free_c_pointer>;

} // namespace detail

} // namespace lib842

#endif // LIB842_DETAIL_FREE_UNIQUE_PTR_H
