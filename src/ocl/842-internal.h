
#ifndef __842_INTERNAL_H__
#define __842_INTERNAL_H__


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <errno.h>
#include <string.h>


#include "../../include/sw842.h"

#include "../common/endianness.h"
#include "../common/crc32.h"
#include "kerneldeps.h"

//#define DEBUG 1

/* special templates */
#define OP_REPEAT	(0x1B)
#define OP_ZEROS	(0x1C)
#define OP_END		(0x1E)

/* sw only template - this is not in the hw design; it's used only by this
 * software compressor and decompressor, to allow input buffers that aren't
 * a multiple of 8.
 */
#define OP_SHORT_DATA	(0x1D)

/* additional bits of each op param */
#define OP_BITS		(5)
#define REPEAT_BITS	(6)
#define SHORT_DATA_BITS	(3)
#define I2_BITS		(8)
#define I4_BITS		(9)
#define I8_BITS		(8)
#define CRC_BITS	(32)

#define REPEAT_BITS_MAX		(0x3f)
#define SHORT_DATA_BITS_MAX	(0x7)

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

/* the max of the regular templates - not including the special templates */
#define OPS_MAX		(0x1a)

struct sw842_param_decomp {
	uint8_t *in;
	uint8_t bit;
	uint64_t ilen;
	uint8_t *out;
	uint8_t *ostart;
	uint64_t olen;
};

#endif
