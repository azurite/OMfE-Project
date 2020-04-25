#include <iostream>

#include "korali.hpp"
#include "model/model.hpp"

int main(int argc, char **argv) {

    auto e = korali::Experiment();


    e["Problem"]["Type"] = "Optimization/Stochastic";
    // TODO Decide if maximize or minimize, error state?
    // e["Problem"]["Objective"] = "Maximize";
    e["Problem"]["Objective Function"] = &direct;

//    TODO Set bounds, depends on matrix dimensions
// TODO allow only integer variables
// Defining the problem's variables.
    e["Variables"][0]["Name"] = "THREADBLOCK_TILE_M";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

    e["Variables"][0]["Name"] = "THREADBLOCK_TILE_N";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

    e["Variables"][0]["Name"] = "THREADBLOCK_WARP_TILE_K";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

    e["Variables"][0]["Name"] = "WARP_TILE_M";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

    e["Variables"][0]["Name"] = "WARP_TILE_N";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

    e["Variables"][0]["Name"] = "SPLIT_K";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 5.0;

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
