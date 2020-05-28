/*
 * config.h
 *
 *  Created on: Thu 28 May 14:23:37 CEST 2020
 *      Author: Automatically generated
 */
#ifndef CONFIG_H_
#define CONFIG_H_
// This STAYS ////////////////
#define M 10				//
#define N 11				//
#define K 12				//
#define TYPE float			//
//////////////////////////////
// THIS ARE THE ARGUMENTS ////////////
#define THREADBLOCK_TILE_M 5		//
#define THREADBLOCK_TILE_N 6		//
#define THREADBLOCK_TILE_K 7	//
#define LOAD_K 8	//
									//
#define WARP_TILE_M 3				//
#define WARP_TILE_N 4				//
									//
#define THREAD_TILE_M 1				//
#define THREAD_TILE_N 2				//
									//
#define SPLIT_K 9					//
#define CORRECTNESS_TEST
#define BENCHMARK
//////////////////////////////////////
#endif /* CONFIG_H_ */
