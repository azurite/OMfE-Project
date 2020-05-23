/*
 * test_correctness.h
 *
 *  Created on: Mar 19, 2020
 *      Author: neville
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Matrix.h"
#include <string>
#include "cuCOSMAv5.cuh"
#include "config.h"

#ifndef TEST_CORRECTNESS_H_
#define TEST_CORRECTNESS_H_

void test_correctness() {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> cuBLAS(M, N);
	Matrix<TYPE> cuCOSMA(M, N);

	A.fillRandom();
	B.fillRandom();
	cuBLAS.fillZero();
	cuCOSMA.fillZero();

#if DEBUG
	auto CPU = A * B;
#endif


	const TYPE alpha = 1.0f;
	const TYPE beta = 0.0f;

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_cuBLAS(M, N, false);
	Matrix<TYPE> CUDA_cuCOSMA(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;

	checkCudaErrors(
			cudaMallocPitch(&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_cuBLAS.elems,&pitch_C, N * sizeof(TYPE),M));


	checkCudaErrors(
			cudaMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_B.elems, pitch_B,B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_cuBLAS.elems, pitch_C, cuBLAS.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));



	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	checkCudaErrors(
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_A.elems, pitch_A/sizeof(TYPE), &beta, CUDA_cuBLAS.elems, pitch_C/sizeof(TYPE)));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
				cudaMemcpy2D(cuBLAS.elems, N * sizeof(TYPE), CUDA_cuBLAS.elems, pitch_C, N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(CUDA_cuBLAS.elems));

#if DEBUG
	std::cout << "A:" << std::endl;
	A.printMatrix();
	std::cout << "B:" << std::endl;
	B.printMatrix();
	std::cout << "CPU:" << std::endl;
	CPU.printMatrix();
	std::cout << "cuBLAS:" << std::endl;
	cuBLAS.printMatrix();
#endif

	checkCudaErrors(
			cudaMallocPitch(&CUDA_cuCOSMA.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(
				cudaMemcpy2D(CUDA_cuCOSMA.elems, pitch_C, cuCOSMA.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cosmaSgemm(&alpha, CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), &beta, CUDA_cuCOSMA.elems, pitch_C/sizeof(TYPE)));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors(
			cudaMemcpy2D(cuCOSMA.elems, N * sizeof(TYPE), CUDA_cuCOSMA.elems, pitch_C, N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));



#if DEBUG


	std::cout << "cuCOSMA:" << std::endl;
	cuCOSMA.printMatrix();

#endif
	cuBLAS.compareMatrix(cuCOSMA, 0.1);




	checkCudaErrors(cudaFree(CUDA_A.elems));
	checkCudaErrors(cudaFree(CUDA_B.elems));
	checkCudaErrors(cudaFree(CUDA_cuCOSMA.elems));

	checkCudaErrors(cublasDestroy(handle));

}
#endif /* TEST_CORRECTNESS_H_ */
