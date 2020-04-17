/*
 * 842 Software Compression
 *
 * Copyright (C) 2015 Dan Streetman, IBM Corp
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * See 842.h for details of the 842 compressed format.
 */
#include "842-internal.h"
#include "bitstream.h"
#include "../../include/sw842.h"
#include "../common/opcodes.h"
#include "../common/endianness.h"
#include "../common/memaccess.h"
#include "../common/crc32.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

#define PRIME64 (11400714785074694791ULL)

#define NO_ENTRY (-1)

static inline void hash(uint64_t *values, uint64_t *results)
{
	results[0] = (PRIME64 * values[0]) >> (64 - DICT16_BITS); // 2
	results[1] = (PRIME64 * values[1]) >> (64 - DICT16_BITS); // 2
	results[2] = (PRIME64 * values[2]) >> (64 - DICT16_BITS); // 2
	results[3] = (PRIME64 * values[3]) >> (64 - DICT16_BITS); // 2
	results[4] = (PRIME64 * values[4]) >> (64 - DICT32_BITS); // 4
	results[5] = (PRIME64 * values[5]) >> (64 - DICT32_BITS); // 4
	results[6] = (PRIME64 * values[6]) >> (64 - DICT64_BITS); // 8
}

static inline void find_index(struct sw842_param *p)
{
	int16_t index[7];
	uint16_t isIndexValid[7];
	uint16_t isDataValid[7];

	index[0] = p->hashTable16[p->hashes[0]];
	index[1] = p->hashTable16[p->hashes[1]];
	index[2] = p->hashTable16[p->hashes[2]];
	index[3] = p->hashTable16[p->hashes[3]];
	index[4] = p->hashTable32[p->hashes[4]];
	index[5] = p->hashTable32[p->hashes[5]];
	index[6] = p->hashTable64[p->hashes[6]];

	isIndexValid[0] = (index[0] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[1] = (index[1] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[2] = (index[2] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[3] = (index[3] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[4] = (index[4] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[5] = (index[5] >= 0) ? 0xFFFF : 0x0000;
	isIndexValid[6] = (index[6] >= 0) ? 0xFFFF : 0x0000;

#ifdef ONLY_WELL_DEFINED_BEHAVIOUR
	isDataValid[0] = (index[0] >= 0 && p->rollingFifo16[index[0]] == p->dataAndIndices[0]) ? 0xFFFF : 0x0000;
	isDataValid[1] = (index[1] >= 0 && p->rollingFifo16[index[1]] == p->dataAndIndices[1]) ? 0xFFFF : 0x0000;
	isDataValid[2] = (index[2] >= 0 && p->rollingFifo16[index[2]] == p->dataAndIndices[2]) ? 0xFFFF : 0x0000;
	isDataValid[3] = (index[3] >= 0 && p->rollingFifo16[index[3]] == p->dataAndIndices[3]) ? 0xFFFF : 0x0000;
	isDataValid[4] = (index[4] >= 0 && p->rollingFifo32[index[4]] == p->dataAndIndices[4]) ? 0xFFFF : 0x0000;
	isDataValid[5] = (index[5] >= 0 && p->rollingFifo32[index[5]] == p->dataAndIndices[5]) ? 0xFFFF : 0x0000;
	isDataValid[6] = (index[6] >= 0 && p->rollingFifo64[index[6]] == p->dataAndIndices[6]) ? 0xFFFF : 0x0000;
#else
	// Causes (generally inocuous) out-of-bounds access when index[i] is negative
	isDataValid[0] = (p->rollingFifo16[index[0]] == p->dataAndIndices[0]) ? 0xFFFF : 0x0000;
	isDataValid[1] = (p->rollingFifo16[index[1]] == p->dataAndIndices[1]) ? 0xFFFF : 0x0000;
	isDataValid[2] = (p->rollingFifo16[index[2]] == p->dataAndIndices[2]) ? 0xFFFF : 0x0000;
	isDataValid[3] = (p->rollingFifo16[index[3]] == p->dataAndIndices[3]) ? 0xFFFF : 0x0000;
	isDataValid[4] = (p->rollingFifo32[index[4]] == p->dataAndIndices[4]) ? 0xFFFF : 0x0000;
	isDataValid[5] = (p->rollingFifo32[index[5]] == p->dataAndIndices[5]) ? 0xFFFF : 0x0000;
	isDataValid[6] = (p->rollingFifo64[index[6]] == p->dataAndIndices[6]) ? 0xFFFF : 0x0000;
#endif

	p->validity[0] = isIndexValid[0] & isDataValid[0];
	p->validity[1] = isIndexValid[1] & isDataValid[1];
	p->validity[2] = isIndexValid[2] & isDataValid[2];
	p->validity[3] = isIndexValid[3] & isDataValid[3];
	p->validity[4] = isIndexValid[4] & isDataValid[4];
	p->validity[5] = isIndexValid[5] & isDataValid[5];
	p->validity[6] = isIndexValid[6] & isDataValid[6];

	p->dataAndIndices[7] = p->validity[0] & index[0];
	p->dataAndIndices[8] = p->validity[1] & index[1];
	p->dataAndIndices[9] = p->validity[2] & index[2];
	p->dataAndIndices[10] = p->validity[3] & index[3];
	p->dataAndIndices[11] = p->validity[4] & index[4];
	p->dataAndIndices[12] = p->validity[5] & index[5];
	p->dataAndIndices[13] = p->validity[6] & index[6];

	p->templateKeys[0] = (13 * 3) & p->validity[0];
	p->templateKeys[1] = (13 * 5) & p->validity[1];
	p->templateKeys[2] = (13 * 7) & p->validity[2];
	p->templateKeys[3] = (13 * 11) & p->validity[3];
	p->templateKeys[4] = (53 * 3) & p->validity[4];
	p->templateKeys[5] = (53 * 5) & p->validity[5];
	p->templateKeys[6] = (149 * 3) & p->validity[6];
}

static inline uint16_t max(uint16_t a, uint16_t b)
{
	return (a > b) ? a : b;
}

static inline uint8_t get_template(struct sw842_param *p)
{
	uint16_t template_key = 0;

	uint16_t former = max(p->templateKeys[4],
			      p->templateKeys[0] + p->templateKeys[1]);
	uint16_t latter = max(p->templateKeys[5],
			      p->templateKeys[2] + p->templateKeys[3]);
	template_key = max(p->templateKeys[6], former + latter);

	template_key >>= 1;

	return ops_dict[template_key];
}

template <uint8_t TEMPLATE_KEY>
static inline void add_template(struct sw842_param *p)
{
	uint64_t out = 0;

	switch (TEMPLATE_KEY) {
	case 0x00: // { D8, N0, N0, N0 }, 64 bits
		stream_write_bits(p->stream, TEMPLATE_KEY, OP_BITS);
		stream_write_bits(p->stream, p->dataAndIndices[6], D8_BITS);
		stream_write_bits(p->stream, p->dataAndIndices[15], 0);
		stream_write_bits(p->stream, p->dataAndIndices[15], 0);
		stream_write_bits(p->stream, p->dataAndIndices[15], 0);
		break;
	case 0x01: // { D4, D2, I2, N0 }, 56 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D4_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[4]) << (D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[2]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D4_BITS + D2_BITS + I2_BITS);
		break;
	case 0x02: // { D4, I2, D2, N0 }, 56 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D4_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[4]) << (I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[3]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D4_BITS + I2_BITS + D2_BITS);
		break;
	case 0x03: // { D4, I2, I2, N0 }, 48 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D4_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[4]) << (I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D4_BITS + I2_BITS + I2_BITS);
		break;
	case 0x04: // { D4, I4, N0, N0 }, 41 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D4_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[4]) << (I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[12]));
		stream_write_bits(p->stream, out, OP_BITS + D4_BITS + I4_BITS);
		break;
	case 0x05: // { D2, I2, D4, N0 }, 56 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D2_BITS + I2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[0]) << (I2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[5]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D2_BITS + I2_BITS + D4_BITS);
		break;
	case 0x06: // { D2, I2, D2, I2 }, 48 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D2_BITS + I2_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[0]) << (I2_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[2]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D2_BITS + I2_BITS + D2_BITS +
					  I2_BITS);
		break;
	case 0x07: // { D2, I2, I2, D2 }, 48 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D2_BITS + I2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[0]) << (I2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[3]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D2_BITS + I2_BITS + I2_BITS +
					  D2_BITS);
		break;
	case 0x08: // { D2, I2, I2, I2 }, 40 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D2_BITS + I2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[0]) << (I2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D2_BITS + I2_BITS + I2_BITS +
					  I2_BITS);
		break;
	case 0x09: // { D2, I2, I4, N0 }, 33 bits
		out = (((uint64_t)TEMPLATE_KEY) << (D2_BITS + I2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[0]) << (I2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[12]));
		stream_write_bits(p->stream, out,
				  OP_BITS + D2_BITS + I2_BITS + I4_BITS);
		break;
	case 0x0a: // { I2, D2, D4, N0 }, 56 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + D2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (D2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[1]) << (D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[5]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + D2_BITS + D4_BITS);
		break;
	case 0x0b: // { I2, D4, I2, N0 }, 48 bits
		//printf("template 0x0b!\n")
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + D4_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (D4_BITS + I2_BITS)) |
		      (((uint64_t)swap_be_to_native32(read32(p->in + 2)))) << (I2_BITS) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + D4_BITS + I2_BITS);
		break;
	case 0x0c: // { I2, D2, I2, D2 }, 48 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + D2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (D2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[1]) << (I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[3]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + D2_BITS + I2_BITS +
					  D2_BITS);
		break;
	case 0x0d: // { I2, D2, I2, I2 }, 40 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + D2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (D2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[1]) << (I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + D2_BITS + I2_BITS +
					  I2_BITS);
		break;
	case 0x0e: // { I2, D2, I4, N0 }, 33 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + D2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (D2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[1]) << (I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[12]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + D2_BITS + I4_BITS);
		break;
	case 0x0f: // { I2, I2, D4, N0 }, 48 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + I2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (I2_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[5]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + I2_BITS + D4_BITS);
		break;
	case 0x10: // { I2, I2, D2, I2 }, 40 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + I2_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (I2_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[2]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + I2_BITS + D2_BITS +
					  I2_BITS);
		break;
	case 0x11: // { I2, I2, I2, D2 }, 40 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + I2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (I2_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[3]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + I2_BITS + I2_BITS +
					  D2_BITS);
		break;
	case 0x12: // { I2, I2, I2, I2 }, 32 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + I2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (I2_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + I2_BITS + I2_BITS +
					  I2_BITS);
		break;
	case 0x13: // { I2, I2, I4, N0 }, 25 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I2_BITS + I2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[7]) << (I2_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[8]) << (I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[12]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I2_BITS + I2_BITS + I4_BITS);
		break;
	case 0x14: // { I4, D4, N0, N0 }, 41 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I4_BITS + D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[11]) << (D4_BITS)) |
		      (((uint64_t)p->dataAndIndices[5]));
		stream_write_bits(p->stream, out, OP_BITS + I4_BITS + D4_BITS);
		break;
	case 0x15: // { I4, D2, I2, N0 }, 33 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I4_BITS + D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[11]) << (D2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[2]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I4_BITS + D2_BITS + I2_BITS);
		break;
	case 0x16: // { I4, I2, D2, N0 }, 33 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I4_BITS + I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[11]) << (I2_BITS + D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (D2_BITS)) |
		      (((uint64_t)p->dataAndIndices[3]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I4_BITS + D2_BITS + I2_BITS);
		break;
	case 0x17: // { I4, I2, I2, N0 }, 25 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I4_BITS + I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[11]) << (I2_BITS + I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[9]) << (I2_BITS)) |
		      (((uint64_t)p->dataAndIndices[10]));
		stream_write_bits(p->stream, out,
				  OP_BITS + I4_BITS + I2_BITS + I2_BITS);
		break;
	case 0x18: // { I4, I4, N0, N0 }, 18 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I4_BITS + I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[11]) << (I4_BITS)) |
		      (((uint64_t)p->dataAndIndices[12]));
		stream_write_bits(p->stream, out, OP_BITS + I4_BITS + I4_BITS);
		break;
	case 0x19: // { I8, N0, N0, N0 }, 8 bits
		out = (((uint64_t)TEMPLATE_KEY) << (I8_BITS)) |
		      (((uint64_t)p->dataAndIndices[13]));
		stream_write_bits(p->stream, out, OP_BITS + I8_BITS);
		break;
	default:
		fprintf(stderr, "Invalid template: %x\n", TEMPLATE_KEY);
	}
}

static inline void add_repeat_template(struct sw842_param *p, uint8_t r)
{
	uint64_t out =
		(((uint64_t)OP_REPEAT) << (REPEAT_BITS)) | (((uint64_t)--r));

	stream_write_bits(p->stream, out, OP_BITS + REPEAT_BITS);
}

static inline void add_zeros_template(struct sw842_param *p)
{
	stream_write_bits(p->stream, OP_ZEROS, OP_BITS);
}

static inline void add_end_template(struct sw842_param *p)
{
	stream_write_bits(p->stream, OP_END, OP_BITS);
}

static inline void get_next_data(struct sw842_param *p)
{
	p->dataAndIndices[0] = swap_be_to_native16(read16(p->in));
	p->dataAndIndices[1] = swap_be_to_native16(read16(p->in + 2));
	p->dataAndIndices[2] = swap_be_to_native16(read16(p->in + 4));
	p->dataAndIndices[3] = swap_be_to_native16(read16(p->in + 6));
	p->dataAndIndices[4] = swap_be_to_native32(read32(p->in));
	p->dataAndIndices[5] = swap_be_to_native32(read32(p->in + 4));
	p->dataAndIndices[6] = swap_be_to_native64(read64(p->in));
#if defined(BRANCH_FREE) && BRANCH_FREE == 1
	p->dataAndIndices[14] = swap_be_to_native32(read32(p->in + 2));
	p->dataAndIndices[15] = 0x0000000000000000;
#endif
}

/* update the hashtable entries.
 * only call this after finding/adding the current template
 */
static inline void update_hashtables(struct sw842_param *p)
{
	uint64_t pos = p->in - p->instart;
	uint16_t i64 = (pos >> 3) % (1 << I8_BITS);
	uint16_t i32 = (pos >> 2) % (1 << I4_BITS);
	uint16_t i16 = (pos >> 1) % (1 << I2_BITS);

	p->rollingFifo16[i16] = p->dataAndIndices[0];
	p->rollingFifo16[i16 + 1] = p->dataAndIndices[1];
	p->rollingFifo16[i16 + 2] = p->dataAndIndices[2];
	p->rollingFifo16[i16 + 3] = p->dataAndIndices[3];
	p->rollingFifo32[i32] = p->dataAndIndices[4];
	p->rollingFifo32[i32 + 1] = p->dataAndIndices[5];
	p->rollingFifo64[i64] = p->dataAndIndices[6];

	p->hashTable16[p->hashes[0]] = i16;
	p->hashTable16[p->hashes[1]] = i16 + 1;
	p->hashTable16[p->hashes[2]] = i16 + 2;
	p->hashTable16[p->hashes[3]] = i16 + 3;
	p->hashTable32[p->hashes[4]] = i32;
	p->hashTable32[p->hashes[5]] = i32 + 1;
	p->hashTable64[p->hashes[6]] = i64;
}

/* find the next template to use, and add it
 * the p->dataN fields must already be set for the current 8 byte block
 */
static inline void process_next(struct sw842_param *p)
{
	uint8_t templateKey;

	p->validity[0] = false;
	p->validity[1] = false;
	p->validity[2] = false;
	p->validity[3] = false;
	p->validity[4] = false;
	p->validity[5] = false;
	p->validity[6] = false;

	p->templateKeys[0] = 0;
	p->templateKeys[1] = 0;
	p->templateKeys[2] = 0;
	p->templateKeys[3] = 0;
	p->templateKeys[4] = 0;
	p->templateKeys[5] = 0;
	p->templateKeys[6] = 0;

	hash(p->dataAndIndices, p->hashes);

	find_index(p);

	templateKey = get_template(p);

#if defined(BRANCH_FREE) && BRANCH_FREE == 1
	stream_write_bits(p->stream, templateKey, OP_BITS);
	for (int opnum = 0; opnum < 4; opnum++) {
		stream_write_bits(
			p->stream,
			p->dataAndIndices[templates[templateKey][opnum][0]],
			templates[templateKey][opnum][1]);
	}
#else
	switch (templateKey) {
	case 0x00: // { D8, N0, N0, N0 }, 64 bits
		add_template<0x00>(p);
		break;
	case 0x01: // { D4, D2, I2, N0 }, 56 bits
		add_template<0x01>(p);
		break;
	case 0x02: // { D4, I2, D2, N0 }, 56 bits
		add_template<0x02>(p);
		break;
	case 0x03: // { D4, I2, I2, N0 }, 48 bits
		add_template<0x03>(p);
		break;
	case 0x04: // { D4, I4, N0, N0 }, 41 bits
		add_template<0x04>(p);
		break;
	case 0x05: // { D2, I2, D4, N0 }, 56 bits
		add_template<0x05>(p);
		break;
	case 0x06: // { D2, I2, D2, I2 }, 48 bits
		add_template<0x06>(p);
		break;
	case 0x07: // { D2, I2, I2, D2 }, 48 bits
		add_template<0x07>(p);
		break;
	case 0x08: // { D2, I2, I2, I2 }, 40 bits
		add_template<0x08>(p);
		break;
	case 0x09: // { D2, I2, I4, N0 }, 33 bits
		add_template<0x09>(p);
		break;
	case 0x0a: // { I2, D2, D4, N0 }, 56 bits
		add_template<0x0a>(p);
		break;
	case 0x0b: // { I2, D4, I2, N0 }, 48 bits
		add_template<0x0b>(p);
		break;
	case 0x0c: // { I2, D2, I2, D2 }, 48 bits
		add_template<0x0c>(p);
		break;
	case 0x0d: // { I2, D2, I2, I2 }, 40 bits
		add_template<0x0d>(p);
		break;
	case 0x0e: // { I2, D2, I4, N0 }, 33 bits
		add_template<0x0e>(p);
		break;
	case 0x0f: // { I2, I2, D4, N0 }, 48 bits
		add_template<0x0f>(p);
		break;
	case 0x10: // { I2, I2, D2, I2 }, 40 bits
		add_template<0x10>(p);
		break;
	case 0x11: // { I2, I2, I2, D2 }, 40 bits
		add_template<0x11>(p);
		break;
	case 0x12: // { I2, I2, I2, I2 }, 32 bits
		add_template<0x12>(p);
		break;
	case 0x13: // { I2, I2, I4, N0 }, 25 bits
		add_template<0x13>(p);
		break;
	case 0x14: // { I4, D4, N0, N0 }, 41 bits
		add_template<0x14>(p);
		break;
	case 0x15: // { I4, D2, I2, N0 }, 33 bits
		add_template<0x15>(p);
		break;
	case 0x16: // { I4, I2, D2, N0 }, 33 bits
		add_template<0x16>(p);
		break;
	case 0x17: // { I4, I2, I2, N0 }, 25 bits
		add_template<0x17>(p);
		break;
	case 0x18: // { I4, I4, N0, N0 }, 18 bits
		add_template<0x18>(p);
		break;
	case 0x19: // { I8, N0, N0, N0 }, 8 bits
		add_template<0x19>(p);
		break;
	default:
		fprintf(stderr, "Invalid template: %x\n", templateKey);
	}
#endif
}

/**
 * sw842_compress
 *
 * Compress the uncompressed buffer of length @ilen at @in to the output buffer
 * @out, using no more than @olen bytes, using the 842 compression format.
 *
 * Returns: 0 on success, error on failure.  The @olen parameter
 * will contain the number of output bytes written on success, or
 * 0 on error.
 */
int optsw842_compress(const uint8_t *in, size_t ilen, uint8_t *out,
		      size_t *olen)
{
	/* if using strict mode, we can only compress a multiple of 8 */
	if (ilen % 8) {
		fprintf(stderr,
			"Can only compress multiples of 8 bytes, but len is len %zu (%% 8 = %zu)\n",
			ilen, ilen % 8);
		return -EINVAL;
	}

	struct sw842_param *p = (struct sw842_param *)malloc(
		sizeof(struct sw842_param));

	memset(&p->hashes, 0, sizeof(p->hashes));

	uint64_t last, next;
	uint8_t repeat_count = 0;

	for (uint16_t i = 0; i < (1 << DICT16_BITS); i++) {
		p->hashTable16[i] = NO_ENTRY;
	}

	for (uint16_t i = 0; i < (1 << DICT32_BITS); i++) {
		p->hashTable32[i] = NO_ENTRY;
	}

	for (uint16_t i = 0; i < (1 << DICT64_BITS); i++) {
		p->hashTable64[i] = NO_ENTRY;
	}

	p->in = in;
	p->instart = in;
	p->ilen = ilen;

	p->stream = stream_open(out, *olen);

	p->olen = *olen;

	*olen = 0;

	/* make initial 'last' different so we don't match the first time */
	last = ~read64(p->in);

	while (p->ilen > 7) {
		next = read64(p->in);

		/* must get the next data, as we need to update the hashtable
		 * entries with the new data every time
		 */
		get_next_data(p);

#ifdef CUDA842_STRICT
		/* we don't care about endianness in last or next;
		 * we're just comparing 8 bytes to another 8 bytes,
		 * they're both the same endianness
		 */
		if (next == last) {
			/* repeat count bits are 0-based, so we stop at +1 */
			if (++repeat_count <= REPEAT_BITS_MAX)
				goto repeat;
		}
		if (repeat_count) {
			add_repeat_template(p, repeat_count);
			repeat_count = 0;
			if (next == last) /* reached max repeat bits */
				goto repeat;
		}

		if (next == 0)
			add_zeros_template(p);
		else
			process_next(p);

	repeat:
		last = next;
#else
		process_next(p);
#endif
		update_hashtables(p);
		p->in += 8;
		p->ilen -= 8;

#ifdef ENABLE_ERROR_HANDLING
		if (stream_is_overfull(p->stream))
			break;
#endif
	}

	if (repeat_count)
		add_repeat_template(p, repeat_count);

	add_end_template(p);

	/*
	 * crc(0:31) is appended to target data starting with the next
	 * bit after End of stream template.
	 * nx842 calculates CRC for data in big-endian format. So doing
	 * same here so that sw842 decompression can be used for both
	 * compressed data.
	 */
#ifndef DISABLE_CRC
	uint32_t crc = crc32_be(0, in, ilen);

	stream_write_bits(p->stream, crc, CRC_BITS);
#endif

	stream_flush(p->stream);

#ifdef ENABLE_ERROR_HANDLING
	bool overfull = stream_is_overfull(p->stream);
#endif

	*olen = stream_size(p->stream);

	stream_close(p->stream);
	free(p);


#ifdef ENABLE_ERROR_HANDLING
	if (overfull)
		return -ENOSPC;
#endif

	return 0;
}
