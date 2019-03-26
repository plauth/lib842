#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#include "sw842.h"

//#define CHUNK_SIZE 32768
#define CHUNK_SIZE 1024
#define THREADS_PER_BLOCK 32
#define STRLEN 32

__global__ void cuda842_decompress(uint64_t *in, unsigned int ilen, uint64_t *out);

#define CHECK_ERROR( err ) \
  if( err != cudaSuccess ) { \
    printf("Error: %s\n", cudaGetErrorString(err)); \
    exit( -1 ); \
  }

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

int main( int argc, const char* argv[])
{
	uint8_t *inH, *compressedH, *decompressedH, *transposedH;
	uint64_t *inD, *compressedD, *decompressedD;
	inH = compressedH = decompressedH = NULL;
	unsigned int ilen, olen, dlen;
	ilen = olen = dlen = 0;
	long long timestart_comp, timeend_comp;
	long long timestart_decomp, timeend_decomp;
	cudaError_t cuda_error;
	int count = 0;
	cudaGetDeviceCount(&count);
  	printf(" %d CUDA devices found\n", count);
  	if(!count)
    		::exit(EXIT_FAILURE);


	if(argc <= 1) {
		ilen = STRLEN;
		olen = ilen * 2;
		dlen = ilen;
		inH = (uint8_t*) malloc(ilen);
		cudaMalloc((void**) &inD, ilen);
		compressedH = (uint8_t*) malloc(olen);
		cudaMalloc((void**) &compressedD, olen);
		decompressedH = (uint8_t*) malloc(dlen);
		cudaMalloc((void**) &decompressedD, dlen);
		memset(inH, 0, ilen);
		cudaMemset(inD, 0, ilen);
		memset(compressedH, 0, olen);
		cudaMemset(compressedD, 0, olen);
		memset(decompressedH, 0, dlen);
		cudaMemset(decompressedD, 0, dlen);

		uint8_t tmp[] = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45};//"0011223344556677889900AABBCCDDEE";
		strncpy((char *) inH, (const char *) tmp, STRLEN);

	} else if (argc == 2) {
		FILE *fp;
		fp=fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		unsigned int flen = ftell(fp);
		ilen = flen;
		printf("original file length: %d\n", ilen);
		ilen = nextMultipleOfChunkSize(ilen);
		printf("original file length (padded): %d\n", ilen);
		olen = ilen * 2;
		dlen = ilen;
		fseek(fp, 0, SEEK_SET);

		inH = (uint8_t*) malloc(ilen);
		cudaMalloc((void**) &inD, ilen);
		compressedH = (uint8_t*) malloc(olen);
		transposedH = (uint8_t*) malloc(olen);
		cudaMalloc((void**) &compressedD, olen);
		decompressedH = (uint8_t*) malloc(dlen);
		cudaMalloc((void**) &decompressedD, dlen);
		memset(inH, 0, ilen);
		cudaMemset(inD, 0, ilen);
		memset(compressedH, 0, olen);
		cudaMemset(compressedD, 0, olen);
		memset(decompressedH, 0, dlen);
		cudaMemset(decompressedD, 0, dlen);


		if(!fread(inH, flen, 1, fp)) {
			fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
		}
		fclose(fp);
	}

	if(ilen > CHUNK_SIZE) {
		printf("Using chunks of %d bytes\n", CHUNK_SIZE);
	
		timestart_comp = timestamp();
		#pragma omp parallel for
		for(int chunk_num = 0; chunk_num < ilen / CHUNK_SIZE; chunk_num++) {
			
			unsigned int chunk_olen = CHUNK_SIZE * 2;
			uint8_t* chunk_in = inH + (CHUNK_SIZE * chunk_num);
			uint8_t* chunk_out = compressedH + ((CHUNK_SIZE * 2) * chunk_num);
			
			sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);

			for(int i = 0; i < 256; i++){
				memcpy(transposedH + (i*(CHUNK_SIZE*2)) + (chunk_num*8), chunk_out + (i*8), 8);
			}
		}
		timeend_comp = timestamp();




		cuda_error = cudaMemcpy(compressedD, compressedH, olen, cudaMemcpyHostToDevice);
		CHECK_ERROR(cuda_error);

		timestart_decomp = timestamp();

		printf("Threads per Block: %d\n", THREADS_PER_BLOCK );

		cuda842_decompress<<<(ilen / CHUNK_SIZE) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(compressedD, ilen, decompressedD);
		cudaDeviceSynchronize();
		cuda_error = cudaGetLastError();
		CHECK_ERROR(cuda_error);

		timeend_decomp = timestamp();

		cuda_error = cudaMemcpy(decompressedH, decompressedD, dlen, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
        CHECK_ERROR(cuda_error);

		printf("Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));
		printf("Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (ilen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));


	} else {

		sw842_compress(inH, ilen, compressedH, &olen);
		printf("copying compressed data to device\n");
		cuda_error = cudaMemcpy(compressedD, compressedH, olen, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
        CHECK_ERROR(cuda_error);
        printf("starting with device-based decompression\n");
        cuda842_decompress<<<1,1>>>(compressedD, olen, decompressedD);
        printf("copying decompressed data back to the host\n");
		cuda_error = cudaMemcpy(decompressedH, decompressedD, dlen, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
        CHECK_ERROR(cuda_error);

	}
	
	if (memcmp(inH, decompressedH, ilen) == 0) {
		printf("Compression- and decompression was successful!\n");
	} else {
		fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
		FILE *fpIn, *fpOut;
		fpIn=fopen("original.bin", "w");
		fwrite(inH, ilen, 1, fpIn);
		fclose(fpIn);
		fpOut=fopen("decompressed.bin", "w");
		fwrite(decompressedH, dlen, 1, fpOut);
		fclose(fpOut);
		free(inH);
		free(compressedH);
		free(decompressedH);
		cudaFree(inD);
		cudaFree(compressedD);
		cudaFree(decompressedD);
		return 0;
	}
	free(inH);
	free(compressedH);
	free(decompressedH);
	cudaFree(inD);
	cudaFree(compressedD);
	cudaFree(decompressedD);

	printf("\n\n");
	return 0;
}
