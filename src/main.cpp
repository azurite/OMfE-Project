#include <iostream>

#include "korali.hpp"
#include "model/model.hpp"
#inlcude "src/config.h"

int main(int argc, char **argv) {

    auto e = korali::Experiment();


    e["Problem"]["Type"] = "Optimization/Stochastic";
    e["Problem"]["Objective"] = "Minimize";
    e["Problem"]["Objective Function"] = &direct;

// TODO allow only integer variables
// Defining the problem's variables.
    e["Variables"][0]["Name"] = "THREADBLOCK_TILE_M";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = M;

    e["Variables"][0]["Name"] = "THREADBLOCK_TILE_N";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = N;

    e["Variables"][0]["Name"] = "THREADBLOCK_WARP_TILE_K";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = K;

    e["Variables"][0]["Name"] = "WARP_TILE_M";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = M;

    e["Variables"][0]["Name"] = "WARP_TILE_N";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = N;

    e["Variables"][0]["Name"] = "SPLIT_K";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = K;

    // TODO Decide which solver to use
    // Configuring CMA-ES parameters
    e["Solver"]["Type"] = "CMAES";
    e["Solver"]["Population Size"] = 16;
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7;
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100;

    auto k = korali::Engine();
    std::cout << "Engine and Experiment loaded" << std::endl;
    std::cout << "fitness(3.14) = " << fitness(3.14) << std::endl;

    //k.run(e);
    return 0;
}
