/*
 * Matrix.h
 *
 *  Created on: Feb 18, 2020
 *      Author: neville
 */
#define BLOCK_SIZE 32 //Define the Blocksize for various submatrix calculations

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdint.h>
#include <random>
#include <cassert>
#include <iostream>
#include <random>
#include <iostream>
#include <ctime>
#include <cuda_runtime_api.h>
#include     <cuda.h>
#include "CudaMatrixMultiply.cu"
#include "CudaMatrixMultiplyShared.cu"
#include <iomanip>
#include <assert.h>
#include <omp.h>

#include "Util/helper_functions.h"
#include "Util/helper_cuda.h"
#include <iomanip>

/**
 *  A Matrix Class
 *  Assumes Row-Major
 */
template<class T>
class Matrix {
public:
    int cols;
    int rows;
    T *elems;
    bool init;

    /**
     * Initializes a matrix with rows and cols
     * Allocates memory if init == true, otherwise does not allocate memory for the values
     *
     * @param rows
     * @param cols
     * @param init
     */
    __host__ __device__

    Matrix(int64_t rows, int64_t cols, bool init = true) {
        this->cols = cols;
        this->rows = rows;
        this->init = init;
        if (init) {
            this->elems = new T[cols * rows];
        }

    }

    /**
     * If memory for the matrix is allocated: delete the allocate memory
     * Otherwise do nothing
     */
    __host__ __device__

    virtual ~Matrix() {
        if (init) {
            delete[] elems;
        }

    }

    /**
     * Fills the Matrix with random values
     */
    void fillRandom() {
#if DEBUG
        std::mt19937 rng(0);
#else
        std::mt19937 rng(std::time(nullptr));
#endif

        std::uniform_real_distribution<double> uniform(0, 1);

        for (int i = 0; i < cols * rows; ++i) {
            elems[i] = (T) uniform(rng);
        }
    }

    /**
     * Fills the Matrix with zeros
     */
    void fillZero() {
        for (int i = 0; i < cols * rows; ++i) {
            elems[i] = 0;
        }
    }

    /**
     * Fills the Matrix with a constant value
     */
    void fillConst(const T val) {
        for (int i = 0; i < cols * rows; ++i) {
            elems[i] = val;
        }
    }

    /**
     * Prints the Matrix
     */
    void printMatrix() {

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << elems[i * cols + j] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    /**
     * Compares this Matrix to other Matrix.
     * If they differ the method prints the difference and aborts()
     *
     * @param other Matrix to compare to
     */
    void compareMatrix(const Matrix &other, double max_error_allowed = 1e-4) {
        double max_error = 0.0;

#pragma omp parallel
        {
#pragma omp for reduction(max:max_error)
            for (int i = 0; i < rows * cols; ++i) {

                double err = elems[i] - other.elems[i];
                max_error = std::max(max_error, std::abs(err));

            }

        }

//        std::cout << "Max Error: " << max_error << std::endl;

        if (max_error > max_error_allowed) {
//            std::cout << "Error too big!" << std::endl;
//			std::cout << "Row: " << i << " ,Col: " << j << std::endl;
//			std::cout << "This: " << std::setprecision(99)
//					<< elems[i * cols + j] << std::endl;
//			std::cout << "Other: " << std::setprecision(99)
//					<< other.elems[i * other.cols + j] << std::endl;
            exit(-1);
        } else {
//            std::cout << "Implementation seems correct!" << std::endl;
        }
    }

    /**
     * Multiplies this Matrix with B on the CPU
     * C = A*B
     *
     *
     * @param B Matrix
     * @return C Matrix
     */
    Matrix operator*(const Matrix<T> &B) {

        assert(cols == B.rows);

        Matrix<T> C(rows, B.cols);

        for (int i = 0; i < C.rows; ++i) {
            for (int j = 0; j < C.cols; ++j) {
                T acc = (T) 0;
                for (int k = 0; k < cols; ++k) {

                    acc += elems[i * cols + k] * B.elems[k * C.cols + j];
                }
                C.elems[i * C.cols + j] = acc;
            }
        }
        return C;

    }

    /**
     * Multiplies this Matrix with B on the GPU using global memory and boundchecking
     *
     * @param B Matrix
     * @return C Matrix
     */
    Matrix CUDAMatrixMultiply2DBoundChecking(const Matrix<T> &B) {

        Matrix<T> C(rows, B.cols);

        Matrix<T> Cuda_A(rows, cols, false);
        Matrix<T> Cuda_B(B.rows, B.cols, false);

        Matrix<T> Cuda_C(rows, B.cols, false);

        size_t size_A = rows * cols * sizeof(T);
        size_t size_B = B.rows * B.cols * sizeof(T);
        size_t size_C = rows * B.cols * sizeof(T);

        checkCudaErrors(cudaMalloc(&Cuda_A.elems, size_A));
        checkCudaErrors(cudaMalloc(&Cuda_B.elems, size_B));
        checkCudaErrors(cudaMalloc(&Cuda_C.elems, size_C));

        checkCudaErrors(
                cudaMemcpy(Cuda_A.elems, elems, size_A,
                           cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(Cuda_B.elems, B.elems, size_B,
                           cudaMemcpyHostToDevice));

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        CUDAMatrixMultiply2DBoundChecking_Kernel<<<dimGrid, dimBlock >> > (Cuda_A,
                Cuda_B, Cuda_C, (T) 0);

        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(
                cudaMemcpy(C.elems, Cuda_C.elems, size_C,
                           cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(Cuda_A.elems));
        checkCudaErrors(cudaFree(Cuda_B.elems));
        checkCudaErrors(cudaFree(Cuda_C.elems));

        return C;

    }

    /**
     * Multiplies this Matrix with B on the GPU using global memory and no boundchecking, but more memory
     *
     * @param B Matrix
     * @return C Matrix
     */
    Matrix CUDAMatrixMultiply2DNonBoundChecking(const Matrix<T> &B) {

        Matrix<T> C(rows, B.cols);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        size_t size_A = rows * cols * sizeof(T);
        size_t size_B = B.rows * B.cols * sizeof(T);
        size_t size_C = dimGrid.x * dimGrid.y * BLOCK_SIZE * BLOCK_SIZE
                        * sizeof(T);

        Matrix<T> Cuda_A(rows, cols, false);
        Matrix<T> Cuda_B(B.rows, B.cols, false);
        Matrix<T> Cuda_C(dimGrid.y * BLOCK_SIZE, dimGrid.x * BLOCK_SIZE, false);

        checkCudaErrors(cudaMalloc(&Cuda_A.elems, size_A));
        checkCudaErrors(cudaMalloc(&Cuda_B.elems, size_B));
        checkCudaErrors(cudaMalloc(&Cuda_C.elems, size_C));

        checkCudaErrors(
                cudaMemcpy(Cuda_A.elems, elems, size_A,
                           cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(Cuda_B.elems, B.elems, size_B,
                           cudaMemcpyHostToDevice));

        CUDAMatrixMultiply2DNonBoundChecking_Kernel<<<dimGrid, dimBlock >> > (
                Cuda_A, Cuda_B, Cuda_C, (T) 0);

        checkCudaErrors(cudaGetLastError());

        for (int i = 0; i < C.rows; i++) {

            gpuErrchk(
                    cudaMemcpy(&C.elems[i * C.cols],
                               &Cuda_C.elems[i * Cuda_C.cols], C.cols * sizeof(T),
                               cudaMemcpyDeviceToHost));

        }

        checkCudaErrors(cudaFree(Cuda_A.elems));
        checkCudaErrors(cudaFree(Cuda_B.elems));
        checkCudaErrors(cudaFree(Cuda_C.elems));

        return C;

    }

    /**
     * Multiplies this Matrix with B on the GPU using shared memory and, boundchecking
     *
     * @param B Matrix
     * @return C Matrix
     */
    Matrix CUDAMatrixMultiply2DBoundCheckingShared(const Matrix<T> &B) {

        Matrix<T> C(rows, B.cols);

        Matrix<T> Cuda_A(rows, cols, false);
        Matrix<T> Cuda_B(B.rows, B.cols, false);
        Matrix<T> Cuda_C(rows, B.cols, false);

        size_t size_A = rows * cols * sizeof(T);
        size_t size_B = B.rows * B.cols * sizeof(T);
        size_t size_C = rows * B.cols * sizeof(T);

        checkCudaErrors(cudaMalloc(&Cuda_A.elems, size_A));
        checkCudaErrors(cudaMalloc(&Cuda_B.elems, size_B));
        checkCudaErrors(cudaMalloc(&Cuda_C.elems, size_C));

        checkCudaErrors(
                cudaMemcpy(Cuda_A.elems, elems, size_A,
                           cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(Cuda_B.elems, B.elems, size_B,
                           cudaMemcpyHostToDevice));

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        CUDAMatrixMultiply2DBoundCheckingShared_Kernel<<<dimGrid, dimBlock >> > (
                Cuda_A, Cuda_B, Cuda_C, (T) 0);

        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(
                cudaMemcpy(C.elems, Cuda_C.elems, size_C,
                           cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(Cuda_A.elems));
        checkCudaErrors(cudaFree(Cuda_B.elems));
        checkCudaErrors(cudaFree(Cuda_C.elems));

        return C;

    }

    /**
     * Multiplies this Matrix with B on the GPU using shared memory and no boundchecking, but more memory
     *
     * @param B Matrix
     * @return C Matrix
     */
    Matrix CUDAMatrixMultiply2DNonBoundCheckingShared(const Matrix<T> &B) {

        Matrix<T> C(rows, B.cols);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        size_t size_A = rows * cols * sizeof(T);
        size_t size_B = B.rows * B.cols * sizeof(T);
        size_t size_C = dimGrid.x * dimGrid.y * BLOCK_SIZE * BLOCK_SIZE
                        * sizeof(T);

        Matrix<T> Cuda_A(rows, cols, false);
        Matrix<T> Cuda_B(B.rows, B.cols, false);
        Matrix<T> Cuda_C(dimGrid.y * BLOCK_SIZE, dimGrid.x * BLOCK_SIZE, false);

        checkCudaErrors(cudaMalloc(&Cuda_A.elems, size_A));
        checkCudaErrors(cudaMalloc(&Cuda_B.elems, size_B));
        checkCudaErrors(cudaMalloc(&Cuda_C.elems, size_C));

        checkCudaErrors(
                cudaMemcpy(Cuda_A.elems, elems, size_A,
                           cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(Cuda_B.elems, B.elems, size_B,
                           cudaMemcpyHostToDevice));

        CUDAMatrixMultiply2DNonBoundCheckingShared_Kernel<<<dimGrid, dimBlock >> > (
                Cuda_A, Cuda_B, Cuda_C, (T) 0);

        checkCudaErrors(cudaGetLastError());

        for (int i = 0; i < C.rows; i++) {
            checkCudaErrors(
                    cudaMemcpy(&C.elems[i * C.cols],
                               &Cuda_C.elems[i * Cuda_C.cols], C.cols * sizeof(T),
                               cudaMemcpyDeviceToHost));

        }

        checkCudaErrors(cudaFree(Cuda_A.elems));
        checkCudaErrors(cudaFree(Cuda_B.elems));
        checkCudaErrors(cudaFree(Cuda_C.elems));

        return C;

    }

};

#endif /* MATRIX_H_ */
