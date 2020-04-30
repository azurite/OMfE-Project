#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <limits>
#include <fstream>


#include "model.hpp"
#include "korali.hpp"

#define WRONG std::numeric_limits<int>::min()

void fitness(korali::Sample &k) {

    int THREADBLOCK_TILE_M = k["Parameters"][0];
    int THREADBLOCK_TILE_N = k["Parameters"][1];
    int THREADBLOCK_WARP_TILE_K = k["Parameters"][2];
    int WARP_TILE_M = k["Parameters"][3];
    int WARP_TILE_N = k["Parameters"][4];
    int SPLIT_K = k["Parameters"][5];


    if(WARP_TILE_M > THREADBLOCK_TILE_M){
        k["F(x)"] =  WRONG;
        return ;
    }

    if(WARP_TILE_N >  THREADBLOCK_TILE_N){
        k["F(x)"] =  WRONG;
        return ;
    }


    std::stringstream ss;
    ss << "./src/model/run.sh " <<
          THREADBLOCK_TILE_M << " " <<
          THREADBLOCK_TILE_N << " " <<
          THREADBLOCK_WARP_TILE_K << " " <<
          WARP_TILE_M << " " <<
          WARP_TILE_N << " " <<
          SPLIT_K;

    std::string cmd = ss.str();

    std::cout << cmd << std::endl;

    std::ifstream myfile ("./src/model/time.txt");

    if (!myfile.is_open()) {
        std::cerr << "There was a problem opening the input file!\n";
        exit(1);//exit or do additional error checking
    }


    system(cmd.c_str());

    double res = 0.0;
    myfile >> res;

    myfile.close();

    std::cout << "Result_model: " << res << std::endl;

    k["F(x)"] = -res;

}
