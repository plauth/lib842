#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "../common/endianness.h"

#define CRCPOLY_BE 0x04c11db7

static uint32_t crc32table_be[8][256];

static void crc32init_be(void)
{
	unsigned i, j;
	uint32_t crc = 0x80000000;

	crc32table_be[0][0] = 0;

	for (i = 1; i < 256; i <<= 1) {
		crc = (crc << 1) ^ ((crc & 0x80000000) ? CRCPOLY_BE : 0);
		for (j = 0; j < i; j++)
			crc32table_be[0][i + j] = crc ^ crc32table_be[0][j];
	}
	for (i = 0; i < 256; i++) {
		crc = crc32table_be[0][i];
		for (j = 1; j < 8; j++) {
			crc = crc32table_be[0][(crc >> 24) & 0xff] ^ (crc << 8);
			crc32table_be[j][i] = crc;
		}
	}
}

static void output_table(int rows, int len, char *trans)
{
	int i, j;

	for (j = 0; j < rows; j++) {
		printf("{");
		for (i = 0; i < len - 1; i++) {
			if (i % 4 == 0)
				printf("\n");
			printf("%s(0x%8.8xL), ", trans, crc32table_be[j][i]);
		}
		printf("%s(0x%8.8xL)},\n", trans, crc32table_be[j][len - 1]);
	}
}

int main(int argc, char **argv)
{
	printf("/* this file is generated - do not edit */\n");
	printf("#include <stdint.h>\n\n");

	printf("#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__\n");
	printf("#define tobe(x) (((x & 0x000000FF) << 24) | ((x & 0x0000FF00) <<  8) |\\\n");
	printf("                 ((x & 0x00FF0000) >>  8) | ((x & 0xFF000000) >> 24))\n");
	printf("#else\n");
	printf("#define tobe(x) x\n");
	printf("#endif\n");

	crc32init_be();
	printf("static const uint32_t crc32table_be[%d][%d] = {", 8, 256);
	output_table(8, 256, "tobe");
	printf("};\n");

	printf("#undef tobe\n");

	return 0;
}
