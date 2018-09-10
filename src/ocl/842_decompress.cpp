#include "clutil.hpp"
#include "cl842kernels.hpp"
#include <iostream>

using namespace std;


int main(int argc, char *argv[]) {
    CL842Kernels kernels;


    uint8_t in[]  = {0x01,0x81,0x81,0x89,0x89,0x91,0x91,0x99,0x98,0x0d,0x0d,0x0d,0x4d,0x4d,0x8d,0x8d,0xcd,0xc4,0x70,0x70,0x72,0x72,0x00,0x82,0x82,0x04,0x24,0x24,0x34,0x34,0x44,0x44,0x54,0x5f,0x1d,0xc0,0x3f,0x26,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    uint8_t out[] = "0000000000000000000000000000000000000000000000000000000000000000";
    uint8_t reference[] = "0011223344556677889900AABBCCDDEE";

    uint64_t ilen = 64;
    uint64_t olen = 64;

    cl::Buffer inBuffer     = kernels.allocateBuffer(64, CL_MEM_READ_ONLY);
    cl::Buffer outBuffer    = kernels.allocateBuffer(64, CL_MEM_READ_WRITE);
    cl::Buffer olenBuffer   = kernels.allocateBuffer(8 , CL_MEM_READ_WRITE);

    kernels.writeBuffer(inBuffer, (const void*) in, 64);
    kernels.fillBuffer(outBuffer, 0, 0, 64);
    kernels.writeBuffer(olenBuffer, (const void*) &olen, 8);
    kernels.decompress(inBuffer, 64, outBuffer, olenBuffer);
    kernels.readBuffer(outBuffer, (void*) out, 64);
    kernels.readBuffer(olenBuffer, (void*) &olen, 8);

    printf("Input: %ld bytes\n", ilen);
    printf("Output: %ld bytes\n", olen);
    

    for (int i = 0; i < 64; i++) {
        printf("%02x:", in[i]);
    }

    printf("\n\n");

    for (int i = 0; i < 64; i++) {
        printf("%02x:", out[i]);
    }

    printf("\n\n");
    
    
    if (memcmp(out, reference, 32) == 0) {
        printf("Compression- and decompression was successful!\n");
    } else {
        fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
        return -1;
    }

    return 0;
    
}
