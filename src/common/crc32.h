#ifndef LIB842_SRC_COMMON_CRC32_H
#define LIB842_SRC_COMMON_CRC32_H

#include <stdint.h>
#include <stdlib.h>
#include "crc32table.h"
#include "endianness.h"

/* implements slicing-by-4 or slicing-by-8 algorithm */
static inline uint32_t crc32_be(uint32_t crc, unsigned char const *buf, size_t len)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define DO_CRC(x) crc = t0[(crc ^ (x)) & 255] ^ (crc >> 8)
#  define DO_CRC4 (t3[(q) & 255] ^ t2[(q >> 8) & 255] ^ \
		   t1[(q >> 16) & 255] ^ t0[(q >> 24) & 255])
#  define DO_CRC8 (t7[(q) & 255] ^ t6[(q >> 8) & 255] ^ \
		   t5[(q >> 16) & 255] ^ t4[(q >> 24) & 255])
# else
#  define DO_CRC(x) crc = t0[((crc >> 24) ^ (x)) & 255] ^ (crc << 8)
#  define DO_CRC4 (t0[(q) & 255] ^ t1[(q >> 8) & 255] ^ \
		   t2[(q >> 16) & 255] ^ t3[(q >> 24) & 255])
#  define DO_CRC8 (t4[(q) & 255] ^ t5[(q >> 8) & 255] ^ \
		   t6[(q >> 16) & 255] ^ t7[(q >> 24) & 255])
# endif

	const uint32_t *b;
	size_t rem_len;
	size_t i;

	const uint32_t *t0 = crc32table_be[0], *t1 = crc32table_be[1], *t2 = crc32table_be[2], *t3 = crc32table_be[3];
	const uint32_t *t4 = crc32table_be[4], *t5 = crc32table_be[5], *t6 = crc32table_be[6], *t7 = crc32table_be[7];

	uint32_t q;

	/* Align it */
	if ((long)buf & 3 && len) {
		do {
			DO_CRC(*buf++);
		} while ((--len) && ((long)buf) & 3);
	}

	rem_len = len & 7;
	len = len >> 3;

	b = (const uint32_t *)buf;
	--b;

	for (i = 0; i < len; i++) {
		q = crc ^ *++b; /* use pre increment for speed */
		crc = DO_CRC8;
		q = *++b;
		crc ^= DO_CRC4;
	}

	len = rem_len;

	/* And the last few bytes */
	if (len) {
		uint8_t *p = (uint8_t *)(b + 1) - 1;
		for (i = 0; i < len; i++)
			DO_CRC(*++p); /* use pre increment for speed */
	}
	crc = swap_be_to_native32(crc);
	return crc;
#undef DO_CRC
#undef DO_CRC4
#undef DO_CRC8
}

#endif // LIB842_SRC_COMMON_CRC32_H
