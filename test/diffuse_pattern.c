#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#ifdef USEHW
#include "hw842.h"
#define lib842_compress hw842_compress
#define lib842_decompress hw842_decompress
#else
#include "sw842.h"
#define lib842_compress sw842_compress
#define lib842_decompress sw842_decompress
#endif

#define BUFFER_SIZE 65536

static unsigned xorshift_seed;
unsigned xorshift_next()
{
    xorshift_seed ^= xorshift_seed << 13;
    xorshift_seed ^= xorshift_seed >> 17;
    xorshift_seed ^= xorshift_seed << 5;
    return xorshift_seed;
}

int main(void)
{
    xorshift_seed = 1234u;

    // Initialize data_buffer with a random pattern
    uint8_t *data_buffer = (uint8_t *)malloc(BUFFER_SIZE);
    for (size_t i = 0; i < BUFFER_SIZE; i++) {
        data_buffer[i] = (uint8_t)xorshift_next();
    }

    for (int it = 0; it < 13; it++) {
        // "Diffuse" each value in data_buffer with its neighbors.
        // This progressively increases the compressibility of the data,
        // and eventually converges to all zeroes
        uint8_t *tmpbuf = (uint8_t *)malloc(BUFFER_SIZE);
        for (size_t i = 0; i < BUFFER_SIZE; i++) {
            tmpbuf[i] = (uint8_t)((data_buffer[(i+BUFFER_SIZE-1) % BUFFER_SIZE] +
                                   data_buffer[(i+1) % BUFFER_SIZE]) / 3);
        }
        memcpy(data_buffer, tmpbuf, BUFFER_SIZE);
        free(tmpbuf);

        // Try to copress the data, then uncompress it, and check the result is correct
        uint8_t *comp_data = (uint8_t *)malloc(2*BUFFER_SIZE);
        size_t comp_size = 2*BUFFER_SIZE;
        int ret_compress = lib842_compress(data_buffer, BUFFER_SIZE, comp_data, &comp_size);
        if (ret_compress == 0) {
            uint8_t *uncomp_data = (uint8_t *)malloc(BUFFER_SIZE);
            size_t uncomp_size = BUFFER_SIZE;
            int ret_decompress = lib842_decompress(comp_data, comp_size, uncomp_data, &uncomp_size);
            if (ret_decompress == 0) {
                if (memcmp(data_buffer, uncomp_data, BUFFER_SIZE) == 0) {
                    printf("it=%i: Pattern compressed to %zu bytes and decompressed OK\n", it, comp_size);
                } else {
                    printf("it=%i: Pattern compressed to %zu bytes but decompression INCORRECT\n", it, comp_size);
                }
            } else {
                printf("it=%i: Pattern compressed to %zu bytes but decompression FAILED (ret=%d)\n", it, comp_size, ret_decompress);
            }

            free(uncomp_data);
        } else {
            printf("it=%i: sw842_compress FAILED (ret=%d)\n", it, ret_compress);
        }

        free(comp_data);
    }

    free(data_buffer);

    return 0;
}