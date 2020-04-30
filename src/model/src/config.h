/*
 * config.h
 *
 *  Created on: Do 30 Apr 2020 18:13:58 CEST
 *      Author: Automatically generated
 */
#ifndef CONFIG_H_
#define CONFIG_H_
// This STAYS ////////////////
#define M 499				//
#define N 2377				//
#define K 5857				//
#define TYPE float			//
#define RUNS 50 			//
//////////////////////////////
// THIS ARE THE ARGUMENTS ////////////
#define THREADBLOCK_TILE_M 475		//
#define THREADBLOCK_TILE_N 1581		//
#define THREADBLOCK_WARP_TILE_K 61	//
									//
#define WARP_TILE_M 92				//
#define WARP_TILE_N 662				//
									//
#define SPLIT_K 12					//
//////////////////////////////////////
#endif /* CONFIG_H_ */
