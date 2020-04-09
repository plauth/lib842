
#ifndef __842_INTERNAL_H__
#define __842_INTERNAL_H__

#include "../common/842.h"
#include "../../include/cuda842.h"

#define BRANCH_FREE (1)
//#define DEBUG 1

#ifndef CUDA842_STRICT
#undef I2_BITS
#undef I4_BITS
#undef I8_BITS
#define STATIC_LOG2_ARG CUDA842_CHUNK_SIZE
#include "../common/static_log2.h"
#define I2_BITS		(STATIC_LOG2_VALUE-1)
#define I4_BITS		(STATIC_LOG2_VALUE-2)
#define I8_BITS		(STATIC_LOG2_VALUE-3)
#endif

#define D2_BITS 	(16)
#define D4_BITS 	(32)
#define D8_BITS 	(64)
#define N0_BITS		(0)

#define DICT16_BITS     (10)
#define DICT32_BITS     (11)
#define DICT64_BITS     (10)

#define I2N (13)
#define I4N (53)
#define I8N (149)

//1st value: position of payload in dataAndIndices
//2nd value: number of bits
#define D20_OP	{0,  D2_BITS}
#define D21_OP	{1,  D2_BITS}
#define D22_OP	{2,  D2_BITS}
#define D23_OP	{3,  D2_BITS}
#define D40_OP	{4,  D4_BITS}
#define D41_OP  {5,  D4_BITS}
#define D80_OP	{6,  D8_BITS}
#define I20_OP	{7,  I2_BITS}
#define I21_OP	{8,  I2_BITS}
#define I22_OP	{9,  I2_BITS}
#define I23_OP	{10, I2_BITS}
#define I40_OP	{11, I4_BITS}
#define I41_OP	{12, I4_BITS}
#define I80_OP	{13, I8_BITS}
#define D4S_OP  {14, D4_BITS}
#define N0_OP	{15, 0}

#define OP_DEC_NOOP  (0x00)
#define OP_DEC_DATA	 (0x00)
#define OP_DEC_INDEX (0x80)

#define OP_DEC_N0	{(N0_BITS | OP_DEC_NOOP),  0}
#define OP_DEC_D2	{(D2_BITS | OP_DEC_DATA),  2}
#define OP_DEC_D4	{(D4_BITS | OP_DEC_DATA),  4}
#define OP_DEC_D8	{(D8_BITS | OP_DEC_DATA),  8}
#define OP_DEC_I2	{(I2_BITS | OP_DEC_INDEX), 2}
#define OP_DEC_I4	{(I4_BITS | OP_DEC_INDEX), 4}
#define OP_DEC_I8	{(I8_BITS | OP_DEC_INDEX), 8}

struct sw842_param {
	struct bitstream* stream;

	const uint8_t *in;
	const uint8_t *instart;
	uint64_t ilen;
	uint8_t *out;
	uint64_t olen;

	// 0-6: data; 7-13: indices; 14: 0
	uint64_t dataAndIndices[16];
	uint64_t hashes[7];
	uint16_t validity[7];
	uint16_t templateKeys[7];

	// L1D cache consumption: ~12.5 KiB
	int16_t hashTable16[1 << DICT16_BITS]; // 1024 * 2 bytes =   2 KiB
	int16_t hashTable32[1 << DICT32_BITS]; // 2048 * 2 bytes =   4 KiB
	int16_t hashTable64[1 << DICT64_BITS]; // 1024 * 2 bytes =   2 KiB
	uint16_t rollingFifo16[1 << I2_BITS];   // 256  * 2 bytes = 0.5 KiB
	uint32_t rollingFifo32[1 << I4_BITS];   // 512  * 4 bytes =   2 KiB
	uint64_t rollingFifo64[1 << I8_BITS];   // 256  * 8 bytes =   2 KiB
};

 struct sw842_param_decomp {
 	uint64_t *out;
 	const uint64_t* ostart;
 	const uint64_t *in;
 	uint32_t bits;
 	uint64_t buffer;
 };

#endif
