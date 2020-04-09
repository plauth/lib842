#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#include "../../include/sw842.h"
#include "../../include/cuda842.h"

#define THREADS_PER_BLOCK 32
#define STRLEN 32
#define STREAM_COUNT 3
#ifdef USE_UNIFIED_MEM
#define CHUNKS_PER_THREAD 32
#else
#define CHUNKS_PER_THREAD 1024
#endif

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

size_t nextMultipleOfChunkSize(size_t input) {
	size_t size = CUDA842_CHUNK_SIZE * CHUNKS_PER_THREAD * THREADS_PER_BLOCK;
	return (input + (size-1)) & ~(size-1);
}

int main( int argc, const char* argv[])
{
	#ifdef CUDA842_STRICT
	printf("Running in strict mode (i.e. fully compatible to the hardware-based nx842 unit).\n");
	#endif
	uint8_t *in, *compressed, *decompressed;
	#ifndef USE_UNIFIED_MEM
	uint64_t *compressedD, *decompressedD;
	#else
	printf("Using unified memory model.\n");
	#endif
	#ifdef USE_STREAMS
		printf("Using streams for overlapping memory transfers and computation.\n");
		cudaStream_t streams[STREAM_COUNT];
		for(int i = 0; i < STREAM_COUNT; i++) {
			cudaStreamCreate(&streams[i]);
		}
	#endif
	size_t ilen, olen, dlen;
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
		#ifdef USE_UNIFIED_MEM
		cudaMallocManaged(&in, ilen);
		cudaMallocManaged(&compressed, olen);
		cudaMallocManaged(&decompressed, dlen);
		#else
		cudaHostAlloc((void**) &in, ilen, cudaHostAllocPortable);
		cudaHostAlloc((void**) &compressed, olen, cudaHostAllocPortable);
		cudaHostAlloc((void**) &decompressed, dlen, cudaHostAllocPortable);

		cudaMalloc((void**) &compressedD, olen);
		cudaMalloc((void**) &decompressedD, dlen);
		cudaMemset(compressedD, 0, olen);
		cudaMemset(decompressedD, 0, dlen);
		#endif

		memset(in, 0, ilen);
		memset(compressed, 0, olen);
		memset(decompressed, 0, dlen);

		uint8_t tmp[] = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45};//"0011223344556677889900AABBCCDDEE";

		memcpy(in, tmp, STRLEN);

	} else if (argc == 2) {
		FILE *fp;
		fp=fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		size_t flen = (size_t)ftell(fp);
		ilen = flen;
		printf("original file length: %zu\n", ilen);
		ilen = nextMultipleOfChunkSize(ilen);
		printf("original file length (padded): %zu\n", ilen);
		olen = ilen * 2;
		dlen = ilen;
		fseek(fp, 0, SEEK_SET);

		#ifdef USE_UNIFIED_MEM
        cudaMallocManaged(&in, ilen);
        cudaMallocManaged(&compressed, olen);
        cudaMallocManaged(&decompressed, dlen);
        #else
		cudaHostAlloc((void**) &in, ilen, cudaHostAllocPortable);
		cudaHostAlloc((void**) &compressed, olen, cudaHostAllocPortable);
		cudaHostAlloc((void**) &decompressed, dlen, cudaHostAllocPortable);

		cudaMalloc((void**) &compressedD, olen);
		cudaMalloc((void**) &decompressedD, dlen);
		cudaMemset(compressedD, 0, olen);
		cudaMemset(decompressedD, 0, dlen);
		#endif

		memset(in, 0, ilen);
		memset(compressed, 0, olen);
		memset(decompressed, 0, dlen);

		if(!fread(in, flen, 1, fp)) {
			fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
		}
		fclose(fp);
	}

	if(ilen > CUDA842_CHUNK_SIZE) {
		printf("Using chunks of %d bytes\n", CUDA842_CHUNK_SIZE);

		size_t num_chunks = ilen / CUDA842_CHUNK_SIZE;
		size_t *compressedChunkPositions = (size_t*) malloc(sizeof(size_t) * num_chunks);
		size_t *compressedChunkSizes = (size_t*) malloc(sizeof(size_t) * num_chunks);

		timestart_comp = timestamp();
		#pragma omp parallel for
		for(size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {

			size_t chunk_olen = CUDA842_CHUNK_SIZE * 2;
			uint8_t* chunk_in = in + (CUDA842_CHUNK_SIZE * chunk_num);
			uint8_t* chunk_out = compressed + ((CUDA842_CHUNK_SIZE * 2) * chunk_num);

			sw842_compress(chunk_in, CUDA842_CHUNK_SIZE, chunk_out, &chunk_olen);
			compressedChunkSizes[chunk_num] = chunk_olen;
		}
		timeend_comp = timestamp();

		printf("Threads per Block: %d\n", THREADS_PER_BLOCK );


		#if defined USE_STREAMS
			const size_t chunks_per_kernel = CHUNKS_PER_THREAD * THREADS_PER_BLOCK;
			int stream_counter = 0;
			timestart_decomp = timestamp();
			for(size_t i = 0; i < num_chunks; i += chunks_per_kernel) {
				cudaMemcpyAsync(compressedD + ((i * CUDA842_CHUNK_SIZE * 2)/8), compressedH + (i * CUDA842_CHUNK_SIZE * 2), chunks_per_kernel * CUDA842_CHUNK_SIZE * 2, cudaMemcpyHostToDevice, streams[stream_counter%STREAM_COUNT]);
				cuda842_decompress<<<chunks_per_kernel / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[stream_counter%STREAM_COUNT]>>>(compressedD + (i * (CUDA842_CHUNK_SIZE/8) * 2), decompressedD + (i * (CUDA842_CHUNK_SIZE/8)));
				cudaMemcpyAsync(decompressedH + (i * CUDA842_CHUNK_SIZE), decompressedD + (i * (CUDA842_CHUNK_SIZE/8)), chunks_per_kernel * CUDA842_CHUNK_SIZE, cudaMemcpyDeviceToHost, streams[stream_counter%STREAM_COUNT]);
				stream_counter++;
			}
			cudaDeviceSynchronize();
			cuda_error = cudaGetLastError();
			CHECK_ERROR(cuda_error);
			timeend_decomp = timestamp();
		#elif defined USE_UNIFIED_MEM
 			const size_t chunks_per_kernel = CHUNKS_PER_THREAD * THREADS_PER_BLOCK;
			timestart_decomp = timestamp();
			for(size_t i = 0; i < num_chunks; i += chunks_per_kernel) {
				cuda842_decompress<<<chunks_per_kernel / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0>>>(((uint64_t *)compressed) + (i * (CUDA842_CHUNK_SIZE/8) * 2), ((uint64_t*) decompressed) + (i * (CUDA842_CHUNK_SIZE/8)));
			}
			cudaDeviceSynchronize();
			cuda_error = cudaGetLastError();
			CHECK_ERROR(cuda_error);
			timeend_decomp = timestamp();
		#else
			cuda_error = cudaMemcpy(compressedD, compressed, olen, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CHECK_ERROR(cuda_error);

			timestart_decomp = timestamp();
			cuda842_decompress<<<num_chunks / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(compressedD, decompressedD);
			cudaDeviceSynchronize();
			cuda_error = cudaGetLastError();
			CHECK_ERROR(cuda_error);
	        timeend_decomp = timestamp();

			cuda_error = cudaMemcpy(decompressed, decompressedD, dlen, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
	        CHECK_ERROR(cuda_error);
		#endif




		printf("Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));
		printf("Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (ilen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));


	} else {

		sw842_compress(in, ilen, compressed, &olen);
		#ifdef USE_UNIFIED_MEM
			cuda842_decompress<<<1,1>>>((uint64_t*)compressed, (uint64_t*)decompressed);
			cudaDeviceSynchronize();
			cuda_error = cudaGetLastError();
			CHECK_ERROR(cuda_error);
		#else
			cuda_error = cudaMemcpy(compressedD, compressed, olen, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CHECK_ERROR(cuda_error);
			cuda842_decompress<<<1,1>>>(compressedD, decompressedD);
			cudaMemcpy(decompressed, decompressedD, dlen, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cuda_error = cudaGetLastError();
			CHECK_ERROR(cuda_error);
        #endif

	}

	if (memcmp(in, decompressed, ilen) == 0) {
		printf("Compression- and decompression was successful!\n");
	} else {
		fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
	}

	#ifdef USE_UNIFIED_MEM
	cudaFree(in);
	cudaFree(compressed);
	cudaFree(decompressed);
	#else
	cudaFreeHost(in);
	cudaFreeHost(compressed);
	cudaFreeHost(decompressed);

	cudaFree(compressedD);
	cudaFree(decompressedD);
	#endif
	#ifdef USE_STREAMS
	for(int i = 1; i < STREAM_COUNT; i++) {
		cudaStreamDestroy(streams[i]);
	}
	#endif
	printf("\n\n");
	return 0;
}
