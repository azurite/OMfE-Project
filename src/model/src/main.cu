/*
 ============================================================================
 Name        : main.cu
 Author      : Neville Walo
 Version     :
 Copyright   : Your copyright notice
 Description : Cuda matrix multiplication based on cosma
 ============================================================================
 */

//#include <liblsb.h>
#include "benchmark.h"
#include "test_correctness.h"
#include "config.h"
#include <iostream>

int main(int argc, const char *argv[]) {
//	std::cout << std::string(100, '_') << std::endl;
//	std::cout << std::string(100, '_') << std::endl;

//	const auto matrix_mult_str = std::to_string(M) + "x" + std::to_string(K)
//			+ " * " + std::to_string(K) + "x" + std::to_string(N);
//	std::cout << matrix_mult_str << std::endl;


#ifdef CORRECTNESS_TEST
	test_correctness();
#endif

#ifdef BENCHMARK
//	LSB_Init("cuCOSMA", 0);
//
//	LSB_Reg_param("MAX_COMPUTATION_DEVIATION_LOWER: %f", MAX_COMPUTATION_DEVIATION_LOWER);
//	LSB_Reg_param("MAX_COMPUTATION_DEVIATION_UPPER: %f", MAX_COMPUTATION_DEVIATION_UPPER);
//
//	LSB_Set_Rparam_int("M", M);
//	LSB_Set_Rparam_int("N", N);
//	LSB_Set_Rparam_int("K", K);
//
//	LSB_Reg_param("THREADBLOCK_TILE_M: %i", THREADBLOCK_TILE_M);
//	LSB_Reg_param("THREADBLOCK_TILE_N: %i", THREADBLOCK_TILE_N);
//	LSB_Reg_param("THREADBLOCK_TILE_K: %i", THREADBLOCK_TILE_K);
//
//	LSB_Reg_param("WARP_TILE_M: %i", WARP_TILE_M);
//	LSB_Reg_param("WARP_TILE_N: %i", WARP_TILE_N);
//	LSB_Reg_param("WARP_TILE_K: %i", WARP_TILE_K);
//
//	LSB_Reg_param("THREAD_TILE_M: %i", THREAD_TILE_M);
//	LSB_Reg_param("THREAD_TILE_N: %i", THREAD_TILE_N);
//	LSB_Reg_param("THREAD_TILE_K: %i", THREAD_TILE_K);
//
//	LSB_Reg_param("SPLIT_K: %i", SPLIT_K);
//
//	LSB_Reg_param("TYPE: %s", TYPE_STRING);
//
//	LSB_Reg_param("ADDITIONAL_OCCUPANCY: %i", ADDITIONAL_OCCUPANCY);

	benchmark();

//	LSB_Finalize();
#endif

	return 0;
}

