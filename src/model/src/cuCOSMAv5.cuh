/*
 * cuCOSMAv1.cuh
 *
 *  Created on: Mar 15, 2020
 *      Author: neville
 */

#include "config.h"
#include <cuda_runtime.h>

#pragma once

#ifndef CUCOSMAV1_CUH_
#define CUCOSMAV1_CUH_

#define BLOCK_DIM_Y THREADBLOCK_TILE_M / THREAD_TILE_M
#define BLOCK_DIM_X THREADBLOCK_TILE_N / THREAD_TILE_N

#define ELEMENT(Matrix, Row, Column, Stride) Matrix[Row*Stride+Column]

/**
 * Kernel for cosmaSgemm,
 * Stores C into shared memory and reduces in the end, because we also split the K Dimension
 *
 *
 * @param m
 * @param n
 * @param k
 * @param alpha
 * @param A
 * @param lda
 * @param B
 * @param ldb
 * @param beta
 * @param C
 * @param ldc
 */
__global__ void cosmaSgemm_kernel(const TYPE alpha, const TYPE * __restrict__ A, const int lda, const TYPE * __restrict__ B, const int ldb, const TYPE beta,
                                  TYPE * __restrict__ C, const int ldc) {

    __shared__ TYPE A_Shared[THREADBLOCK_TILE_M][LOAD_K];
    __shared__ TYPE B_Shared[LOAD_K][THREADBLOCK_TILE_N];

    TYPE Thread_tile[THREAD_TILE_M][THREAD_TILE_N];

    TYPE A_register[THREAD_TILE_M];
    TYPE B_register[THREAD_TILE_N];

    // Load c, for now 0
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            Thread_tile[i][j] = 0.0;
        }
    }

// No unroll
    for (int cta_k = 0; cta_k < THREADBLOCK_TILE_K; cta_k += LOAD_K) {

        // Load A into shared memory

        for (int i = threadIdx.y; i < THREADBLOCK_TILE_M; i += BLOCK_DIM_Y) {

            for (int j = threadIdx.x; j < LOAD_K; j += BLOCK_DIM_X) {

                const auto global_i = blockIdx.y * THREADBLOCK_TILE_M + i;
                const auto global_j = blockIdx.z * THREADBLOCK_TILE_K + cta_k + j;

                TYPE a;

                if (global_i < M && global_j < K && cta_k + j < THREADBLOCK_TILE_K) {
                    a = A[global_i * lda + global_j];
                } else {
                    a = 0;
                }
                A_Shared[i][j] = a;

            }

        }

// Load B into shared memory

        for (int i = threadIdx.y; i < LOAD_K; i += BLOCK_DIM_Y) {

            for (int j = threadIdx.x; j < THREADBLOCK_TILE_N; j += BLOCK_DIM_X) {

                const auto global_i = blockIdx.z * THREADBLOCK_TILE_K + cta_k + i;
                const auto global_j = blockIdx.x * THREADBLOCK_TILE_N + j;

                TYPE a;

                if (global_i < K && global_j < N && cta_k + i < THREADBLOCK_TILE_K) {
                    a = B[global_i * ldb + global_j];
                } else {
                    a = 0;
                }
                B_Shared[i][j] = a;

            }

        }

        __syncthreads();

//		if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0
//				&& blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.z == 0) {
//			for (int i = 0; i < THREADBLOCK_TILE_K; ++i) {
//				for (int j = 0; j < THREADBLOCK_TILE_N; ++j) {
//
//					TYPE a = B_Shared[i][j];
//
//					printf("%f, ", a);
//
//				}
//				printf("\n ");
//
//			}
//			printf("\n ");
//			printf("\n ");
//			printf("\n ");
//		}
//		if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.z == 0) {
//			printf("here");
//		}

//
#pragma unroll
        for (int k = 0; k < LOAD_K; k++) {

#pragma unroll
            for (int counter = 0; counter < THREAD_TILE_M; ++counter) {
                A_register[counter] = A_Shared[THREAD_TILE_M * threadIdx.y + counter][k];

            }

#pragma unroll
            for (int counter = 0; counter < THREAD_TILE_N; ++counter) {
                B_register[counter] = B_Shared[k][THREAD_TILE_N * threadIdx.x + counter];

            }

#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {

//					TYPE a = A_Shared[THREAD_TILE_M * threadIdx.y + i][k];
//					TYPE b = B_Shared[k][THREAD_TILE_N * threadIdx.x + j];

                    TYPE a = A_register[i];
                    TYPE b = B_register[j];

                    Thread_tile[i][j] += a * b;
                }

            }
        }

    }

// Store the result
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {

            const auto global_i = THREAD_TILE_M * (blockIdx.y * BLOCK_DIM_Y + threadIdx.y) + i;
            const auto global_j = THREAD_TILE_N * (blockIdx.x * BLOCK_DIM_X + threadIdx.x) + j;

            if (global_i < M && global_j < N) {
                auto c = Thread_tile[i][j];

                if (SPLIT_K == 1) {
                    ELEMENT(C, global_i, global_j, ldc)= c;
                } else {
                    atomicAdd(&ELEMENT(C, global_i, global_j, ldc), c);
                }

            }

        }

    }

}

/**
 * This function performs the matrix-matrix multiplication
 * C = α op ( A ) op ( B ) + β C
 * where α and β are scalars, and A , B and C are matrices stored in RowMajor-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A
 * op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T A H if  transa == CUBLAS_OP_C
 * and op ( B ) is defined similarly for matrix B .
 *
 * Uses the CosmaAlgorithm
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
cublasStatus_t cosmaSgemm(const TYPE *alpha, const TYPE *__restrict__ A, const int lda, const TYPE * __restrict__ B, const int ldb, const TYPE *beta,
                          TYPE *__restrict__ C, const int ldc) {

    if (M < 0 || N < 0 || K < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    constexpr dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    constexpr dim3 dimGrid((N + THREADBLOCK_TILE_N - 1) / THREADBLOCK_TILE_N, (M + THREADBLOCK_TILE_M - 1) / THREADBLOCK_TILE_M, SPLIT_K);

    cosmaSgemm_kernel<<<dimGrid, dimBlock>>>(*alpha, A, lda, B, ldb, *beta, C, ldc);

    return CUBLAS_STATUS_SUCCESS;

}

#endif /* CUCOSMAV1_CUH_ */
