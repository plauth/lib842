#include <iostream>
#include <fstream>

#include "../../include/sw842.h"
#include "cl842decompress.hpp"

using namespace std;

#define STRLEN 32


int main(int argc, char *argv[]) {


    char *compressIn;
    uint8_t *compressOut, *decompressIn, *decompressOut;
    size_t flen, ilen, olen, dlen;

    flen = ilen = olen = dlen = 0;

    size_t num_chunks = 0;

    if(argc <= 1) {
        ilen = STRLEN;
    } else if (argc == 2) {
        std::ifstream is (argv[1], std::ifstream::binary);
        if(!is)
            exit(-1);
        is.seekg (0, is.end);
        flen = (size_t)is.tellg();
        is.seekg (0, is.beg);
        is.close();
        
        ilen = CL842Decompress::paddedSize(flen);
        
        printf("original file length: %zu\n", flen);
        printf("original file length (padded): %zu\n", ilen);
    }

    olen = ilen * 2;
    dlen = ilen;

    compressIn = new char [ilen];
    compressOut = (uint8_t *) calloc(olen, sizeof(uint8_t));
    decompressIn = (uint8_t *) calloc(olen, sizeof(uint8_t));
    decompressOut = (uint8_t *) calloc(dlen, sizeof(uint8_t));

    if(argc <= 1) {
        uint8_t tmp[] = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45};//"0011223344556677889900AABBCCDDEE";
        memcpy(compressIn, tmp, STRLEN);
    }  else if (argc == 2) {
        num_chunks = ilen / CHUNK_SIZE;

        std::ifstream is (argv[1], std::ifstream::binary);
        if(!is)
            exit(-1);
        is.seekg (0, is.beg);
        is.read (compressIn, flen);

        if (is)
            std::cerr << "successfully read all " << flen << " bytes." << std::endl;
        else
            std::cerr << "error: only " << is.gcount() << "bytes could be read"  << std::endl;
        is.close();
    }

    if(num_chunks > 1) {
        printf("Using %zu chunks of %d bytes\n", num_chunks, CHUNK_SIZE);
    
        for(size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) { 
            size_t chunk_olen = CHUNK_SIZE * 2;
            uint8_t* chunk_in = ((uint8_t*) compressIn) + (CHUNK_SIZE * chunk_num);
            uint8_t* chunk_out = compressOut + ((CHUNK_SIZE * 2) * chunk_num);
            
            sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
        }

    } else {
        sw842_compress((uint8_t*) compressIn, ilen, compressOut, &olen);
    }

    memcpy(decompressIn, compressOut, olen);

    CL842Decompress clDecompress(decompressIn, olen, decompressOut, dlen);
    clDecompress.decompress();


    if (memcmp(compressIn, decompressOut, ilen) == 0) {
        //printf("Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));
        //printf("Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (ilen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));

        printf("Compression- and decompression was successful!\n");
    } else {

        /*
        for (size_t i = 0; i < ilen; i++) {
            printf("%02x:", compressIn[i]);
        }

        printf("\n\n");

        for (size_t i = 0; i < olen; i++) {
            printf("%02x:", compressOut[i]);
        }

        printf("\n\n");

        for (size_t i = 0; i < dlen; i++) {
            printf("%02x:", decompressOut[i]);
        }
        */

        printf("\n\n");

        fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
    }

    return 0;
    
}
