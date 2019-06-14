#include <iostream>
#include <cstring>
#include <sys/time.h>

#include "../../include/sw842.h"

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 1024
#define STRLEN 32

using namespace std;

long long timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    long long ms = te.tv_sec * 1000LL + te.tv_usec/1000;
    return ms;
}

int nextMultipleOfChunkSize(unsigned int input) {
    unsigned int size = CHUNK_SIZE * THREADS_PER_BLOCK;
    return (input + (size-1)) & ~(size-1);
} 

int main(int argc, char *argv[]) {
    uint8_t *inH, *compressedH;
    uint32_t ilen, olen, num_chunks;
    ilen = olen = 0;
    num_chunks = 1;

    long long timestart_comp, timeend_comp;


    if(argc <= 1) {
        ilen = STRLEN;
        olen = ilen * 2;

        inH = (uint8_t*) malloc(ilen);
        compressedH = (uint8_t*) malloc(olen);
        memset(inH, 0, ilen);
        memset(compressedH, 0, olen);

        uint8_t tmp[] = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45};//"0011223344556677889900AABBCCDDEE";
        strncpy((char *) inH, (const char *) tmp, STRLEN);
    }  else if (argc == 2) {
        FILE *fp;
        fp=fopen(argv[1], "r");
        fseek(fp, 0, SEEK_END);
        unsigned int flen = ftell(fp);
        ilen = flen;
        fprintf(stderr, "original file length: %d\n", ilen);
        ilen = nextMultipleOfChunkSize(ilen);
        fprintf(stderr, "original file length (padded): %d\n", ilen);
        olen = ilen * 2;
        fseek(fp, 0, SEEK_SET);

        inH = (uint8_t*) malloc(ilen);
        compressedH = (uint8_t*) malloc(olen);
        memset(inH, 0, ilen);
        memset(compressedH, 0, olen);

        num_chunks = ilen / CHUNK_SIZE;

        if(!fread(inH, flen, 1, fp)) {
            fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
            exit(-1);
        }
        fclose(fp);
    }

    if(ilen > CHUNK_SIZE) {
        fprintf(stderr, "Using chunks of %d bytes\n", CHUNK_SIZE);
    
        timestart_comp = timestamp();
        #pragma omp parallel for
        for(uint32_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) { 
            uint32_t chunk_olen = CHUNK_SIZE * 2;
            uint8_t* chunk_in = inH + (CHUNK_SIZE * chunk_num);
            uint8_t* chunk_out = compressedH + ((CHUNK_SIZE * 2) * chunk_num);
            
            sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
        }
        timeend_comp = timestamp();

    } else {
        sw842_compress(inH, ilen, compressedH, &olen);
    }

    fprintf(stderr, "Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));

    fwrite(&olen, sizeof(uint32_t), 1, stdout);
    fwrite(&ilen, sizeof(uint32_t), 1, stdout);
    fwrite(&num_chunks, sizeof(uint32_t), 1, stdout);
    fwrite(compressedH, sizeof(uint8_t), olen, stdout);




    return 0;
    
}
