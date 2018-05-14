#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <sys/time.h>

#ifdef USEHW
#include "hw842.h"
#else
#include "sw842.h"
#endif

//#define CHUNK_SIZE 32768
#define CHUNK_SIZE 4096

#define ERRORCHECK() cErrorCheck(__FILE__, __LINE__)

#define CHECK_ERROR( err ) \
  if( err != cudaSuccess ) { \
    printf("Error: %s\n", cudaGetErrorString(err)); \
    exit( -1 ); \
  }

#define TIMER_CREATE(t)                      \
  cudaEvent_t t##_start, t##_end;            \
  cudaEventCreate(&t##_start);               \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                          \
  cudaEventRecord(t##_start);                   \
  cudaEventSynchronize(t##_start);              \
 
 
#define TIMER_END(t)                                          \
  cudaEventRecord(t##_start);                                 \
  cudaEventSynchronize(t##_start);                            \
  cudaEventRecord(t##_end);                                   \
  cudaEventSynchronize(t##_end);                              \
  cudaEventElapsedTime(&t, t##_start, t##_end);               

long long timestamp() {
	struct timeval te;
	gettimeofday(&te, NULL);
	long long ms = te.tv_sec * 1000LL + te.tv_usec/1000;
	return ms;
}

int nextMultipleOfChunkSize(unsigned int input) {
	return (input + (CHUNK_SIZE-1)) & ~(CHUNK_SIZE-1);
}

inline void cErrorCheck(const char *file, int line) {
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    printf(" @ %s: %d\n", file, line);
    exit(-1);
  }
} 

int main( int argc, const char* argv[])
{
	cudaError_t cuda_error;
	int count = 0;
	cudaGetDeviceCount(&count);
  	printf(" %d CUDA devices found\n", count);
  	if(!count)
    		::exit(EXIT_FAILURE);
  	
	uint8_t *cuda_uncompressed, *cuda_compressed;
	uint8_t *in, *out, *decompressed;
	in = out = decompressed = NULL;
	unsigned int ilen, olen, dlen;
	ilen = olen = dlen = 0;
	long long timestart_comp, timeend_comp;
	long long timestart_decomp, timeend_decomp;
	long long timestart_condense, timeend_condense;

	if(argc <= 1) {
		ilen = 32;
		olen = ilen * 2;
		#ifdef USEHW
		dlen = ilen * 2;
		#else
		dlen = ilen;
		#endif
		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);

		uint8_t tmp[] = "0011223344556677889900AABBCCDDEE";

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		strncpy((char *) in, (const char *) tmp, 32);

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
		#ifdef USEHW
		dlen = ilen * 2;
		#else
		dlen = ilen;
		#endif
		fseek(fp, 0, SEEK_SET);

		in = (uint8_t*) malloc(ilen);
		cudaMalloc((void**) &cuda_uncompressed, ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);
		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		if(!fread(in, flen, 1, fp)) {
			fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
		}
		fclose(fp);
	}

	if(ilen > CHUNK_SIZE) {
		printf("Using chunks of %d bytes\n", CHUNK_SIZE);
	
		int num_chunks = ilen / CHUNK_SIZE;
		uint64_t *compressedChunkPositions = (uint64_t*) malloc(sizeof(uint64_t) * num_chunks);
		uint32_t *compressedChunkSizes = (uint32_t*) malloc(sizeof(uint32_t) * num_chunks);

		timestart_comp = timestamp();
		#pragma omp parallel for
		for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			
			unsigned int chunk_olen = CHUNK_SIZE * 2;
			uint8_t* chunk_in = in + (CHUNK_SIZE * chunk_num);
			uint8_t* chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
			
			#ifdef USEHW
			hw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
			#else
			sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
			#endif
			compressedChunkSizes[chunk_num] = chunk_olen;
		}
		timeend_comp = timestamp();
		timestart_condense = timeend_comp;

		uint64_t currentChunkPos = 0;
		for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			compressedChunkPositions[chunk_num] = currentChunkPos;
			currentChunkPos += compressedChunkSizes[chunk_num];
		}

		uint8_t *out_condensed = (uint8_t *) malloc(currentChunkPos);

		#pragma omp parallel for
		for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			uint8_t * chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
			uint8_t * chunk_condensed = out_condensed + compressedChunkPositions[chunk_num];
			memcpy(chunk_condensed, chunk_out, compressedChunkSizes[chunk_num]);
		}
		timeend_condense = timestamp();
		
		cudaMalloc((void**) &cuda_compressed, ilen);

		timestart_decomp = timestamp();
		#pragma omp parallel for
		for(int chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			unsigned int chunk_dlen = CHUNK_SIZE;

			uint8_t* chunk_in = in + (CHUNK_SIZE * chunk_num);
			uint8_t* chunk_condensed = out_condensed + compressedChunkPositions[chunk_num];
			uint8_t* chunk_decomp = in + (CHUNK_SIZE * chunk_num);
			
			
			#ifdef USEHW
			hw842_decompress(chunk_condensed, compressedChunkSizes[chunk_num], chunk_decomp, &chunk_dlen);
			#else
			sw842_decompress(chunk_condensed, compressedChunkSizes[chunk_num], chunk_decomp, &chunk_dlen);
			#endif

			if (!(memcmp(chunk_in, chunk_decomp, CHUNK_SIZE) == 0)) {
				fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
				//return -1;
			}
		}
		timeend_decomp = timestamp();

		float timer_uncompressed, timer_compressed;
		TIMER_CREATE(timer_uncompressed);
		TIMER_CREATE(timer_compressed);

		TIMER_START(timer_uncompressed);
		cuda_error = cudaMemcpy(cuda_uncompressed, in, ilen, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		TIMER_END(timer_uncompressed);
		CHECK_ERROR(cuda_error);
		ERRORCHECK();

                TIMER_START(timer_compressed);
                cuda_error = cudaMemcpy(cuda_compressed, out_condensed, currentChunkPos, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
                TIMER_END(timer_compressed);
		CHECK_ERROR(cuda_error);
                ERRORCHECK();

		free(compressedChunkPositions);
		free(compressedChunkSizes);
		cudaFree(cuda_uncompressed);
		cudaFree(cuda_compressed);

		printf("Input: %d bytes\n", ilen);
		printf("Output: %lld bytes\n", currentChunkPos);
		printf("Compression factor: %f\n", (float) currentChunkPos / (float) ilen);
		printf("Transfer time to GPU (uncompressed): %f ms\n", timer_uncompressed);
		printf("Transfer time to GPU (compressed): %f ms\n", timer_compressed);

		printf("Compression- and decompression was successful!\n");
	} else {

		#ifdef USEHW
		hw842_compress(in, ilen, out, &olen);
		#else
		sw842_compress(in, ilen, out, &olen);
		#endif

		#ifdef USEHW
		hw842_decompress(out, olen, decompressed, &dlen);
		#else
		sw842_decompress(out, olen, decompressed, &dlen);
		#endif

		printf("Input: %d bytes\n", ilen);
		printf("Output: %d bytes\n", olen);
		printf("Compression factor: %f\n", (float) olen / (float) ilen);

		/*
		for (int i = 0; i < 32; i++) {
			printf("%02x:", in[i]);
		}

		printf("\n\n");

		for (int i = 0; i < 32; i++) {
			printf("%02x:", decompressed[i]);
		}

		printf("\n\n");
		*/ 
		
		if (memcmp(in, decompressed, ilen) == 0) {
			printf("Compression- and decompression was successful!\n");
		} else {
			fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
			return -1;
		}

	}
	
}
