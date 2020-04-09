
#ifndef __842_INTERNAL_H__
#define __842_INTERNAL_H__

/* The 842 compressed format is made up of multiple blocks, each of
 * which have the format:
 *
 * <template>[arg1][arg2][arg3][arg4]
 *
 * where there are between 0 and 4 template args, depending on the specific
 * template operation.  For normal operations, each arg is either a specific
 * number of data bytes to add to the output buffer, or an index pointing
 * to a previously-written number of data bytes to copy to the output buffer.
 *
 * The template code is a 5-bit value.  This code indicates what to do with
 * the following data.  Template codes from 0 to 0x19 should use the template
 * table, the static "decomp_ops" table used in decompress.  For each template
 * (table row), there are between 1 and 4 actions; each action corresponds to
 * an arg following the template code bits.  Each action is either a "data"
 * type action, or a "index" type action, and each action results in 2, 4, or 8
 * bytes being written to the output buffer.  Each template (i.e. all actions
 * in the table row) will add up to 8 bytes being written to the output buffer.
 * Any row with less than 4 actions is padded with noop actions, indicated by
 * N0 (for which there is no corresponding arg in the compressed data buffer).
 *
 * "Data" actions, indicated in the table by D2, D4, and D8, mean that the
 * corresponding arg is 2, 4, or 8 bytes, respectively, in the compressed data
 * buffer should be copied directly to the output buffer.
 *
 * "Index" actions, indicated in the table by I2, I4, and I8, mean the
 * corresponding arg is an index parameter that points to, respectively, a 2,
 * 4, or 8 byte value already in the output buffer, that should be copied to
 * the end of the output buffer.  Essentially, the index points to a position
 * in a ring buffer that contains the last N bytes of output buffer data.
 * The number of bits for each index's arg are: 8 bits for I2, 9 bits for I4,
 * and 8 bits for I8.  Since each index points to a 2, 4, or 8 byte section,
 * this means that I2 can reference 512 bytes ((2^8 bits = 256) * 2 bytes), I4
 * can reference 2048 bytes ((2^9 = 512) * 4 bytes), and I8 can reference 2048
 * bytes ((2^8 = 256) * 8 bytes).  Think of it as a kind-of ring buffer for
 * each of I2, I4, and I8 that are updated for each byte written to the output
 * buffer.  In this implementation, the output buffer is directly used for each
 * index; there is no additional memory required.  Note that the index is into
 * a ring buffer, not a sliding window; for example, if there have been 260
 * bytes written to the output buffer, an I2 index of 0 would index to byte 256
 * in the output buffer, while an I2 index of 16 would index to byte 16 in the
 * output buffer.
 *
 * There are also 3 special template codes; 0x1b for "repeat", 0x1c for
 * "zeros", and 0x1e for "end".  The "repeat" operation is followed by a 6 bit
 * arg N indicating how many times to repeat.  The last 8 bytes written to the
 * output buffer are written again to the output buffer, N + 1 times.  The
 * "zeros" operation, which has no arg bits, writes 8 zeros to the output
 * buffer.  The "end" operation, which also has no arg bits, signals the end
 * of the compressed data.  There may be some number of padding (don't care,
 * but usually 0) bits after the "end" operation bits, to fill the buffer
 * length to a specific byte multiple (usually a multiple of 8, 16, or 32
 * bytes).
 *
 * This software implementation also uses one of the undefined template values,
 * 0x1d as a special "short data" template code, to represent less than 8 bytes
 * of uncompressed data.  It is followed by a 3 bit arg N indicating how many
 * data bytes will follow, and then N bytes of data, which should be copied to
 * the output buffer.  This allows the software 842 compressor to accept input
 * buffers that are not an exact multiple of 8 bytes long.  However, those
 * compressed buffers containing this sw-only template will be rejected by
 * the 842 hardware decompressor, and must be decompressed with this software
 * library.  The 842 software compression module includes a parameter to
 * disable using this sw-only "short data" template, and instead simply
 * reject any input buffer that is not a multiple of 8 bytes long.
 *
 * After all actions for each operation code are processed, another template
 * code is in the next 5 bits.  The decompression ends once the "end" template
 * code is detected.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <errno.h>
#include <string.h>

#include "../../include/cuda842.h"

#include "../common/memaccess.h"
#include "../common/endianness.h"
#include "../common/crc32.h"

#define BRANCH_FREE (1)
//#define DEBUG 1

/* special templates */
#define OP_REPEAT	(0x1B)
#define OP_ZEROS	(0x1C)
#define OP_END		(0x1E)

/* additional bits of each op param */
#define OP_BITS		(5)
#define REPEAT_BITS	(6)
#ifdef CUDA842_STRICT
#define I2_BITS		(8)
#define I4_BITS		(9)
#define I8_BITS		(8)
#else
#define STATIC_LOG2_ARG CUDA842_CHUNK_SIZE
#include "../common/static_log2.h"
#define I2_BITS		(STATIC_LOG2_VALUE-1)
#define I4_BITS		(STATIC_LOG2_VALUE-2)
#define I8_BITS		(STATIC_LOG2_VALUE-3)
#endif
#define D2_BITS 	(16)
#define D4_BITS 	(32)
#define D8_BITS 	(64)
#define CRC_BITS	(32)
#define N0_BITS		(0)

#define REPEAT_BITS_MAX		(0x3f)

/* Arbitrary values used to indicate action */
#define OP_ACTION	(0x70)
#define OP_ACTION_INDEX	(0x10)
#define OP_ACTION_DATA	(0x20)
#define OP_ACTION_NOOP	(0x40)
#define OP_AMOUNT	(0x0f)
#define OP_AMOUNT_0	(0x00)
#define OP_AMOUNT_2	(0x02)
#define OP_AMOUNT_4	(0x04)
#define OP_AMOUNT_8	(0x08)

#define D2		(OP_ACTION_DATA  | OP_AMOUNT_2)
#define D4		(OP_ACTION_DATA  | OP_AMOUNT_4)
#define D8		(OP_ACTION_DATA  | OP_AMOUNT_8)
#define I2		(OP_ACTION_INDEX | OP_AMOUNT_2)
#define I4		(OP_ACTION_INDEX | OP_AMOUNT_4)
#define I8		(OP_ACTION_INDEX | OP_AMOUNT_8)
#define N0		(OP_ACTION_NOOP  | OP_AMOUNT_0)

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
