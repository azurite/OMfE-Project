/*
 * benchmark.h
 *
 *  Created on: Mar 8, 2020
 *      Author: neville
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_
#include "cutlass/gemm/device/gemm.h"
//#include <liblsb.h>
#include <string>
#include "Matrix.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "OnGPU.h"
#include "Cacheflusher.h"
#include "cuCOSMAv5.cuh"
#include <functional>
#include "cuda_profiler_api.h"
#include <iostream>
#include <fstream>

#include "config.h"

#include "Util/helper_cuda.h"

#define RUNS 100 // Defines how often the function will be run
#define WARMUP 10 // Defines the amount of warm up round

/**
 * Benchmarks a function with RUNS repetition and WARMUP warumup rounds
 *
 * @param f Function to be benchmarked
 * @param name Name of the function, need for lsb
 */
template<typename F>
void benchmark_function(F f, const char * name) {

//	std::cout << name << " WarmUp" << std::flush;

// Warmup phase
	for (int i = 0; i < WARMUP; i++) {
		f();
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		flushCache();
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
//		std::cout << "." << std::flush; // Use dots to measure progress
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double total = 0;

//	std::cout << "\n";
//	cudaProfilerStart();
// The actual benchmarking
//	LSB_Set_Rparam_string("Impl", name);



	for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(start);

		f();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		flushCache();
//		std::cout << "." << std::flush; // Use dots to measure progress
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


	}

    std::cout << "Result_CUDA: " << total << std::endl;
    std::ofstream ofs("../time.txt", std::ofstream::trunc);

    ofs << total;

    ofs.close();


//	cudaProfilerStop();
//	std::cout << "\n";

}
/**
 *  Benchmarks a multiplication
 *
 * @param mult The multiplication to benchmark
 */
void benchmark() {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> C(M, N);

	A.fillRandom();
	B.fillRandom();

	const TYPE alpha = 1.0f;
	const TYPE beta = 0.0f;

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_C(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;

	checkCudaErrors(cudaMallocPitch(&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M));
	checkCudaErrors(cudaMallocPitch(&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K));
	checkCudaErrors(cudaMallocPitch(&CUDA_C.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(cudaMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy2D(CUDA_B.elems, pitch_B,B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy2D(CUDA_C.elems, pitch_C, C.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	using ElementAccumulator = TYPE;
	using ElementComputeEpilogue = TYPE;
	using ElementInputA = TYPE;
	using ElementInputB = TYPE;
	using ElementOutput = TYPE;
	using LayoutInputA = cutlass::layout::RowMajor;
	using LayoutInputB = cutlass::layout::RowMajor;
	using LayoutOutput = cutlass::layout::RowMajor;

	using MMAOp = cutlass::arch::OpClassSimt;

	using SmArch = cutlass::arch::Sm50;

	using Gemm = cutlass::gemm::device::Gemm<
	ElementInputA, LayoutInputA,
	ElementInputB, LayoutInputB,
	ElementOutput, LayoutOutput,
	ElementAccumulator,
	MMAOp,
	SmArch
	>;

	Gemm gemm_op;

	typename Gemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
			{ CUDA_A.elems, pitch_A / sizeof(TYPE) }, // Tensor-ref for source matrix A
			{ CUDA_B.elems, pitch_B / sizeof(TYPE) }, // Tensor-ref for source matrix B
			{ CUDA_C.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for source matrix C
			{ CUDA_C.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
			{ alpha, beta }); // Scalars used in the Epilogue

	// Layout of C matrix

//	std::function<void()> Cutlass = [&]() {
//		cutlass::Status status = gemm_op(args);
//	};
//	benchmark_function(Cutlass, "Cutlass");
//
//	cublasHandle_t handle;
//	checkCudaErrors(cublasCreate(&handle));
//
//	std::function<void()> cublas =
//			[&]() {
//				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_A.elems, pitch_A/sizeof(TYPE), &beta, CUDA_C.elems, pitch_C/sizeof(TYPE)));
//			};
//	benchmark_function(cublas, "Cublas");

	std::function<void()> cucosma = [&]() {
		checkCudaErrors(cosmaSgemm(&alpha, CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), &beta, CUDA_C.elems, pitch_C/sizeof(TYPE)));
	};
	benchmark_function(cucosma, "Cosma");

//	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cudaFree(CUDA_A.elems));
	checkCudaErrors(cudaFree(CUDA_B.elems));
	checkCudaErrors(cudaFree(CUDA_C.elems));

}

#endif /* BENCHMARK_H_ */
