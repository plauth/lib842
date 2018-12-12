#ifndef __HASH_HPP__
#define __HASH_HPP__

#include <stdint.h>
#include <stdio.h>
//#include "memaccess.h"

//#define DEBUG           (1)

#define BUFFER16_BITS   (8)
#define BUFFER32_BITS   (9)
#define BUFFER64_BITS   (8)

#define DICT16_BITS     (12)
#define DICT32_BITS     (13)
#define DICT64_BITS     (12)

#define PRIME16		(65521)
#define PRIME32		(2654435761U)
#define PRIME64		(11400714785074694791ULL)

#define NO_ENTRY        (-1)

template<typename V, typename H, uint8_t dictBits> static inline H hash(V value) {
	uint64_t result;
        switch(sizeof(V)) {
                case 2:
                        result = PRIME64 * value;
                        result >>= (64 - dictBits);
                        break;
                case 4:
                        result = PRIME64 * value;
                        result >>= (64 - dictBits);
                        break;
                case 8:
                        result = PRIME64 * value;
                        result >>= (64 - dictBits);
                        break;
                default:
                        fprintf(stderr, "Invalid template parameter V for function hash(V value)\n");
        }

	return (H) result;
}

static inline void hashVec(uint64_t* values, uint64_t* results) {
        results[0] = (PRIME64 * values[0]) >> (64 - DICT16_BITS);   // 2
        results[1] = (PRIME64 * values[1]) >> (64 - DICT16_BITS);   // 2
        results[2] = (PRIME64 * values[2]) >> (64 - DICT16_BITS);   // 2
        results[3] = (PRIME64 * values[3]) >> (64 - DICT16_BITS);   // 2
        results[4] = (PRIME64 * values[4]) >> (64 - DICT32_BITS);   // 4
        results[5] = (PRIME64 * values[5]) >> (64 - DICT32_BITS);   // 4
        results[6] = (PRIME64 * values[6]) >> (64 - DICT64_BITS);   // 8
}

#endif
