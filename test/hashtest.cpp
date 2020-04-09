#include "hash.hpp"

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

static size_t nextMultipleOfEight(size_t input)
{
	return (input + 7) & ~7;
}

int main(int argc, char **argv)
{
	if (argc <= 1) {
		printf("Wrong usage!\n");
	} else {
		printf("Memory consumption by hash tables: %lu KiB\n\n",
		       ((1 << DICT16_BITS) * sizeof(int16_t) +
			(1 << DICT32_BITS) * sizeof(int16_t) +
			(1 << DICT64_BITS) * sizeof(int16_t)) /
			       1024);

		for (int i_args = 1; i_args < argc; i_args++) {
			FILE *fp;
			fp = fopen(argv[i_args], "r");
			fseek(fp, 0, SEEK_END);
			size_t flen = (size_t)ftell(fp);
			fseek(fp, 0, SEEK_SET);
			size_t blen = nextMultipleOfEight(flen);
			printf("Computing all hash values for file '%s' (padded length: %d)\n",
			       argv[i_args], blen);
			uint8_t *buffer = (uint8_t *)malloc(blen);
			memset(buffer, 0, blen);
			if (!fread(buffer, flen, 1, fp)) {
				fprintf(stderr,
					"FAIL: Reading file content to memory failed.\n");
			}
			fclose(fp);

			int16_t hashTable16[1 << DICT16_BITS],
				hashTable32[1 << DICT32_BITS],
				hashTable64[1 << DICT64_BITS];
			uint16_t ringBuffer16[1 << BUFFER16_BITS];
			for (uint16_t i = 0; i < (1 << DICT16_BITS); i++) {
				hashTable16[i] = NO_ENTRY;
			}
			memset(ringBuffer16, 0,
			       (1 << BUFFER16_BITS) * sizeof(uint16_t));
			uint64_t collisions16 = 0;

			//-----

			uint32_t ringBuffer32[1 << BUFFER32_BITS];
			for (uint16_t i = 0; i < (1 << DICT32_BITS); i++) {
				hashTable32[i] = NO_ENTRY;
			}
			memset(ringBuffer32, 0,
			       (1 << BUFFER32_BITS) * sizeof(uint32_t));
			uint64_t collisions32 = 0;

			//-----

			uint64_t ringBuffer64[1 << BUFFER64_BITS];
			for (uint16_t i = 0; i < (1 << DICT64_BITS); i++) {
				hashTable64[i] = NO_ENTRY;
			}
			memset(ringBuffer64, 0,
			       (1 << BUFFER64_BITS) * sizeof(uint32_t));
			uint64_t collisions64 = 0;

			uint8_t *p = buffer;

			while (p < buffer + blen) {
				uint64_t pos = p - buffer;
				uint16_t i64 =
					(pos >> 3) % (1 << BUFFER64_BITS);
				uint16_t i32 =
					(pos >> 2) % (1 << BUFFER32_BITS);
				uint16_t i16 =
					(pos >> 1) % (1 << BUFFER16_BITS);

				replace_hash<uint16_t>(hashTable16,
						       ringBuffer16, i16,
						       read16(p),
						       &collisions16);
				replace_hash<uint16_t>(hashTable16,
						       ringBuffer16, i16 + 1,
						       read16(p + 2),
						       &collisions16);
				replace_hash<uint16_t>(hashTable16,
						       ringBuffer16, i16 + 2,
						       read16(p + 4),
						       &collisions16);
				replace_hash<uint16_t>(hashTable16,
						       ringBuffer16, i16 + 3,
						       read16(p + 6),
						       &collisions16);

				replace_hash<uint32_t>(hashTable32,
						       ringBuffer32, i32,
						       read32(p),
						       &collisions32);
				replace_hash<uint32_t>(hashTable32,
						       ringBuffer32, i32 + 1,
						       read32(p + 4),
						       &collisions32);

				replace_hash<uint64_t>(hashTable64,
						       ringBuffer64, i64,
						       read64(p),
						       &collisions64);

				p += 8;
			}

#ifdef DEBUG
			printf("Collisions for 2 byte values: %" PRIu64
			       " (%f%%)\n",
			       collisions16,
			       ((float)collisions16 / (float)(blen >> 1) *
				100));
			printf("Collisions for 4 byte values: %" PRIu64
			       " (%f%%)\n",
			       collisions32,
			       ((float)collisions32 / (float)(blen >> 2) *
				100));
			printf("Collisions for 8 byte values: %" PRIu64
			       " (%f%%)\n",
			       collisions64,
			       ((float)collisions64 / (float)(blen >> 3) *
				100));
			printf("\n\n");
#endif

			free(buffer);
		}
	}
}
