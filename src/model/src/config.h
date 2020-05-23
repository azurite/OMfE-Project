/*
 * config.h
 *
 *  Created on: Sa 23 Mai 2020 15:57:04 CEST
 *      Author: Automatically generated
 */
#ifndef CONFIG_H_
#define CONFIG_H_
// This STAYS ////////////////
#define M 499				//
#define N 2377				//
#define K 5857				//
#define TYPE float			//
//////////////////////////////
// THIS ARE THE ARGUMENTS ////////////
#define THREADBLOCK_TILE_M 27		//
#define THREADBLOCK_TILE_N 10		//
#define THREADBLOCK_TILE_K 345	//
#define LOAD_K 49	//
									//
#define WARP_TILE_M 27				//
#define WARP_TILE_N 2				//
									//
#define THREAD_TILE_M 3				//
#define THREAD_TILE_N 2				//
									//
#define SPLIT_K 17					//
#define CORRECTNESS_TEST
#define BENCHMARK
//////////////////////////////////////
#endif /* CONFIG_H_ */
