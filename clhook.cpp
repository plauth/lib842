#include <dlfcn.h>
#include <stdio.h>
#include "cpu_serial/842.h"
#include <cassert>


#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

int nextMultipleOfEight(unsigned int input) {
	return (input + 7) & ~7;
} 

typedef cl_int (*clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, 
	const void*, cl_uint, const cl_event*, cl_event*);
static clEnqueueWriteBuffer_t realClEnqueueWriteBuffer = NULL;
extern "C"
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
 	cl_mem buffer,
 	cl_bool blocking_write,
 	size_t offset,
 	size_t cb,
 	const void *ptr,
 	cl_uint num_events_in_wait_list,
 	const cl_event *event_wait_list,
 	cl_event *event) {
	if(offset == 0) {
		void* wmem_comp = malloc(sizeof(struct sw842_param)); 
		uint8_t *in, *out, *decompressed;
		unsigned int ilen, olen, dlen;
		ilen = nextMultipleOfEight(cb);
		olen = ilen * 2;
		dlen = ilen;
		
		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		memcpy(in, ptr, cb);

		sw842_compress(in, ilen, out, &olen, wmem_comp);
		
		printf("\n");
		printf("### lib842 test-hook (start) ###\n");
		printf("Input: %d bytes (padded)\n",ilen);
		printf("Output: %d bytes\n", olen);

		printf("Compression factor: %f\n", (float) olen / (float) ilen);
		sw842_decompress(out, olen, decompressed, &dlen);

		if (memcmp(in, decompressed, ilen) == 0) {
			//printf("Compression- and decompression was successful!\n");
		} else {
			fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
		}
		printf("### lib842 test-hook (end) ###\n");
	}

	if (realClEnqueueWriteBuffer == NULL)
    	realClEnqueueWriteBuffer = (clEnqueueWriteBuffer_t)dlsym(RTLD_NEXT,"clEnqueueWriteBuffer");
	assert(realClEnqueueWriteBuffer != NULL && "clEnqueueWriteBuffer is null");
	return realClEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb,
		ptr, num_events_in_wait_list, event_wait_list, event);
}
