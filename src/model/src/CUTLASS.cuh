/*
 * cuCOSMAv1.cuh
 *
 *  Created on: Mar 15, 2020
 *      Author: neville
 */

#include "cublas_v2.h"
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/helper.h"

#include "config.h"

#include "Util/helper_cuda.h"

#ifndef CUCOSMAV1_CUH_
#define CUCOSMAV1_CUH_

/**
 * This function performs the matrix-matrix multiplication
 * C = α op ( A ) op ( B ) + β C
 * where α and β are scalars, and A , B and C are matrices stored in RowMajor-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A
 * op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T A H if  transa == CUBLAS_OP_C
 * and op ( B ) is defined similarly for matrix B .
 *
 * Uses the CUTLASS algorithm
 * Assumes RowMajor Storage
 *
 * @param handle	handle to the cuBLAS library context.
 * @param transa	operation op(A) that is non- or (conj.) transpose.
 * @param transb	operation op(B) that is non- or (conj.) transpose.
 * @param m			 number of rows of matrix op(A) and C.
 * @param n			 number of columns of matrix op(B) and C.
 * @param k 		number of columns of op(A) and rows of op(B).
 * @param alpha 	<type> scalar used for multiplication.
 * @param A 		<type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
 * @param lda 		leading dimension of two-dimensional array used to store the matrix A.
 * @param B			<type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
 * @param ldb		leading dimension of two-dimensional array used to store matrix B.
 * @param beta		<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
 * @param C			<type> array of dimensions ldc x n with ldc>=max(1,m).
 * @param ldc		leading dimension of a two-dimensional array used to store the matrix C.
 * @return cublasStatus_t CUBLAS_STATUS_SUCCESS || CUBLAS_STATUS_NOT_INITIALIZED || CUBLAS_STATUS_INVALID_VALUE || CUBLAS_STATUS_ARCH_MISMATCH || CUBLAS_STATUS_EXECUTION_FAILED
 */
cublasStatus_t multiply(const float *alpha, const TYPE *__restrict__ A,
		const int lda, const TYPE * __restrict__ B, const int ldb,
		const float *beta, TYPE *__restrict__ C, const int ldc) {

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

	using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<THREADBLOCK_TILE_M, THREADBLOCK_TILE_N, THREADBLOCK_WARP_TILE_K>;
	using ShapeMMAWarp = cutlass::gemm::GemmShape<WARP_TILE_M, WARP_TILE_N, THREADBLOCK_WARP_TILE_K>;

	using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

	if (SPLIT_K == 1) {

		static int const kEpilogueElementsPerAccess = 1;

		using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
		ElementAccumulator, kEpilogueElementsPerAccess, ElementAccumulator, ElementAccumulator>;

		using Gemm = cutlass::gemm::device::Gemm<
		ElementInputA, LayoutInputA,
		ElementInputB, LayoutInputB,
		ElementOutput, LayoutOutput,
		ElementAccumulator,
		MMAOp,
		SmArch,
		ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
		EpilogueOutputOp,
		cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle,
		2 // Stages
		>;

		typename Gemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
				{ A, lda }, // Tensor-ref for source matrix A
				{ B, ldb }, // Tensor-ref for source matrix B
				{ C, ldc }, // Tensor-ref for source matrix C
				{ C, ldc }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
				{ *alpha, *beta }); // Scalars used in the Epilogue

		size_t workspace_size = Gemm::get_workspace_size(args);
		cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

		Gemm gemm_op;

		// Initialize CUTLASS kernel with arguments and workspace pointer
		cutlass::Status status = gemm_op.initialize(args, workspace.get());
		CUTLASS_CHECK(status);
		status = gemm_op();
		CUTLASS_CHECK(status);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


	} else {

		using Gemm = cutlass::gemm::device::GemmSplitKParallel<
		ElementInputA,
		LayoutInputA,
		ElementInputB,
		LayoutInputB,
		ElementOutput,
		LayoutOutput,
		ElementAccumulator,
		MMAOp,
		SmArch,
		ShapeMMAThreadBlock,
		ShapeMMAWarp,
		ShapeMMAOp
		>;

		typename Gemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
				{ A, lda }, // Tensor-ref for source matrix A
				{ B, ldb }, // Tensor-ref for source matrix B
				{ C, ldc }, // Tensor-ref for source matrix C
				{ C, ldc }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
				{ *alpha, *beta },
				SPLIT_K); // Scalars used in the Epilogue

		size_t workspace_size = Gemm::get_workspace_size(args);
		cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

		Gemm gemm_op;

		// Initialize CUTLASS kernel with arguments and workspace pointer
		cutlass::Status status = gemm_op.initialize(args, workspace.get());
		CUTLASS_CHECK(status);
		status = gemm_op();
		CUTLASS_CHECK(status);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	}

	return CUBLAS_STATUS_SUCCESS;

}

#endif /* CUCOSMAV1_CUH_ */
