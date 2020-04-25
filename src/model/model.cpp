# include "model.hpp"
#include "korali.hpp"

int fitness(korali::Sample &k) {

    int THREADBLOCK_TILE_M = k["Parameters"][0];
    int THREADBLOCK_TILE_N = k["Parameters"][1];
    int THREADBLOCK_WARP_TILE_K = k["Parameters"][2];
    int WARP_TILE_M = k["Parameters"][3];
    int WARP_TILE_N = k["Parameters"][4];
    int SPLIT_K = k["Parameters"][5];

    // TODO Check if parameter are even possible, if not return before compiling program

//    TODO Write config file
//      TODO compile program
// TODO Run Program
//  Return time


    return 0;
}
