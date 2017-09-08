#include <stdio.h>

int main() {

	float * ha;
	float * da;

	ha = (float*) malloc(512*sizeof(float));
	cudaMalloc((void**) &da, 512*sizeof(float));

	for (int i = 0; i < 512; i++) {
		ha[i] = i;
	}
	
	cudaMemcpy(da, ha, 512*sizeof(float), cudaMemcpyHostToDevice);
	
	return 0;
}
