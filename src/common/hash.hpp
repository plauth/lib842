#ifndef __HASH_HPP__
#define __HASH_HPP__

#include <stdint.h>
#include <stdio.h>

#define DICT16_BITS     (10)
#define DICT32_BITS     (11)
#define DICT64_BITS     (10)

#define PRIME16		(65521)
#define PRIME32		(2654435761U)
#define PRIME64		(11400714785074694791ULL)

#define NO_ENTRY        (-1)

static inline void hash(uint64_t* values, uint64_t* results) {
        results[0] = (PRIME64 * values[0]) >> (64 - DICT16_BITS);   // 2
        results[1] = (PRIME64 * values[1]) >> (64 - DICT16_BITS);   // 2
        results[2] = (PRIME64 * values[2]) >> (64 - DICT16_BITS);   // 2
        results[3] = (PRIME64 * values[3]) >> (64 - DICT16_BITS);   // 2
        results[4] = (PRIME64 * values[4]) >> (64 - DICT32_BITS);   // 4
        results[5] = (PRIME64 * values[5]) >> (64 - DICT32_BITS);   // 4
        results[6] = (PRIME64 * values[6]) >> (64 - DICT64_BITS);   // 8
}

#endif
