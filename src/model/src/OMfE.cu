/*
 ============================================================================
 Name        : OMfE.cu
 Author      : Neville Walo
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdlib.h>
#include "Matrix.h"
#include "CUTLASS.cuh"
#include <iostream>

#include <chrono>

#include "config.h"
#include "Util/helper_cuda.h"





int main(void) {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> C(M, N);

	A.fillConst(1.0);
	B.fillConst(0.5);

	const TYPE alpha = 1.0f;
	const TYPE beta = 0.0f;

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_C(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;

	checkCudaErrors(
			cudaMallocPitch(&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_C.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_B.elems, pitch_B,B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, cudaMemcpyHostToDevice));


	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < RUNS; ++i) {

		checkCudaErrors(
					multiply(&alpha, CUDA_A.elems, pitch_A / sizeof(TYPE),
							CUDA_B.elems, pitch_B / sizeof(TYPE), &beta,
							CUDA_C.elems, pitch_C / sizeof(TYPE)));
	}



	auto end = std::chrono::steady_clock::now();


//	checkCudaErrors(
//			cudaMemcpy2D(C.elems, N * sizeof(TYPE), CUDA_C.elems, pitch_C,
//			N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));



	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaFree(CUDA_A.elems));
	checkCudaErrors(cudaFree(CUDA_B.elems));
	checkCudaErrors(cudaFree(CUDA_C.elems));

	int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	std::cout << time/RUNS << std::endl;


	return time/RUNS;
}

