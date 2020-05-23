/*
 * Cacheflusher.h
 *
 *  Created on: Mar 14, 2020
 *      Author: neville
 */

#ifndef CACHEFLUSHER_H_
#define CACHEFLUSHER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * Flushes the L2 Cache of all the GPUs
 * @return nothing interesting
 */
int flushCache() {
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	int res = 0;

	for (int dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		double *a;

		if (deviceProp.l2CacheSize) {
			size_t length = (deviceProp.l2CacheSize + sizeof(double) - 1)
					/ sizeof(double);

			checkCudaErrors(cudaMalloc(&a, length * sizeof(double)));

			cublasHandle_t handle;
			checkCudaErrors(cublasCreate(&handle));

			checkCudaErrors(cublasIdamin(handle, length, a, 1, &res));

			checkCudaErrors(cublasDestroy(handle));
			checkCudaErrors(cudaFree(a));

		}

	}
	return res;
}
#endif /* CACHEFLUSHER_H_ */
