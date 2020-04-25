# include "model.hpp"
#include "korali.hpp"
#include <bits/stdc++.h>
#include <iostream>

#define WRONG 999999999

int fitness(korali::Sample &k) {

    int THREADBLOCK_TILE_M = k["Parameters"][0];
    int THREADBLOCK_TILE_N = k["Parameters"][1];
    int THREADBLOCK_WARP_TILE_K = k["Parameters"][2];
    int WARP_TILE_M = k["Parameters"][3];
    int WARP_TILE_N = k["Parameters"][4];
    int SPLIT_K = k["Parameters"][5];


    if(WARP_TILE_M > THREADBLOCK_TILE_M){
        return WRONG
    }

    if(WARP_TILE_N >  THREADBLOCK_TILE_N){
        return WRONG
    }


    String cmd = "./run.sh " + THREADBLOCK_TILE_M + " " + THREADBLOCK_TILE_N + " " + THREADBLOCK_WARP_TILE_K + " " +
                 WARP_TILE_M + " " + WARP_TILE_N + " " + SPLIT_K;

    std::cout << cmd << std::endl;

    int res = system(cmd.c_str());


    return res;
}
