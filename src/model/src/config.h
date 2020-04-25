/*
 * config.h 
 *
 *  Created on: Sa Apr  4 23:11:41 CEST 2020
 *      Author: Automatically generated
 */
#ifndef CONFIG_H_
#define CONFIG_H_

// This STAYS ////////////////
#define M 1024				//
#define N 1024				//
#define K 1024				//
#define TYPE float			//
#define RUNS 16 			//
//////////////////////////////


// THIS ARE THE ARGUMENTS ////////////
#define THREADBLOCK_TILE_M 128		//
#define THREADBLOCK_TILE_N 128		//
#define THREADBLOCK_WARP_TILE_K 8	//
									//
#define WARP_TILE_M 32				//
#define WARP_TILE_N 64				//
									//
#define SPLIT_K 1					//
//////////////////////////////////////
#endif /* CONFIG_H_ */
