#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <limits>
#include <fstream>


#include "model.hpp"
#include "korali.hpp"

#define WRONG std::numeric_limits<int>::min()

void fitness(korali::Sample &k) {

    int M = 499;
    int N = 2377;
    int K = 5857;


    int THREAD_TILE_M = k["Parameters"][0];
    int THREAD_TILE_N = k["Parameters"][1];
    int WARPTILE_MULTIPLICATOR_M = k["Parameters"][2];
    int WARPTILE_MULTIPLICATOR_N = k["Parameters"][3];
    int THREADBLOCK_MULTIPLICATOR_M = k["Parameters"][4];
    int THREADBLOCK_MULTIPLICATOR_N = k["Parameters"][5];
    int LOAD_K = k["Parameters"][6];
    int SPLIT_K = k["Parameters"][7];

    int WARPTILE_M = THREAD_TILE_M * WARPTILE_MULTIPLICATOR_M;
    int WARPTILE_N = THREAD_TILE_N * WARPTILE_MULTIPLICATOR_N;
    int THREADBLOCK_TILE_M = WARPTILE_M * THREADBLOCK_MULTIPLICATOR_M;
    int THREADBLOCK_TILE_N = WARPTILE_N * THREADBLOCK_MULTIPLICATOR_N;

    int THREADBLOCK_K = ceil(K / (double) SPLIT_K);


    std::stringstream ss;
    ss << "./src/model/run.sh " <<
       THREAD_TILE_M << " " <<
       THREAD_TILE_N << " " <<
       WARPTILE_M << " " <<
       WARPTILE_N << " " <<
       THREADBLOCK_TILE_M << " " <<
       THREADBLOCK_TILE_N << " " <<
       THREADBLOCK_K << " " <<
       LOAD_K << " " <<
       SPLIT_K << " " <<
       M << " " <<
       N << " " <<
       K;

    std::string cmd = ss.str();


    std::ifstream myfile("./src/model/time.txt");

    if (!myfile.is_open()) {
        std::cerr << "There was a problem opening the input file!\n";
        exit(1);//exit or do additional error checking
    }

//    std::cout << cmd << std::endl;

    system(cmd.c_str());

    double res = 0.0;
    myfile >> res;

    myfile.close();


    k["F(x)"] = -res;

}
