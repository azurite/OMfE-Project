#include <iostream>

#include "korali.hpp"
#include "model/model.hpp"
#include "model/src/config.h"

int main(int argc, char **argv) {

    auto e = korali::Experiment();


    e["Problem"]["Type"] = "Optimization/Stochastic";
//    e["Problem"]["Objective"] = "Maximize";
    e["Problem"]["Objective Function"] = &fitness;


// Defining the problem's variables.
    e["Variables"][0]["Name"] = "THREADBLOCK_TILE_M";
    e["Variables"][0]["Lower Bound"] = 1.0;
    e["Variables"][0]["Upper Bound"] = M;
    e["Variables"][0]["Granularity"] = 1.0;

    e["Variables"][1]["Name"] = "THREADBLOCK_TILE_N";
    e["Variables"][1]["Lower Bound"] = 1.0;
    e["Variables"][1]["Upper Bound"] = N;
    e["Variables"][1]["Granularity"] = 1.0;

    e["Variables"][2]["Name"] = "THREADBLOCK_WARP_TILE_K";
    e["Variables"][2]["Lower Bound"] = 1.0;
    e["Variables"][2]["Upper Bound"] = 128.0;
    e["Variables"][2]["Granularity"] = 2.0;

    e["Variables"][3]["Name"] = "WARP_TILE_M";
    e["Variables"][3]["Lower Bound"] = 1.0;
    e["Variables"][3]["Upper Bound"] = M;
    e["Variables"][3]["Granularity"] = 1.0;

    e["Variables"][4]["Name"] = "WARP_TILE_N";
    e["Variables"][4]["Lower Bound"] = 1.0;
    e["Variables"][4]["Upper Bound"] = N;
    e["Variables"][4]["Granularity"] = 1.0;

    e["Variables"][5]["Name"] = "SPLIT_K";
    e["Variables"][5]["Lower Bound"] = 1.0;
    e["Variables"][5]["Upper Bound"] = 20.0;
    e["Variables"][5]["Granularity"] = 1.0;

    // TODO Decide which solver to use
    e["Solver"]["Type"] = "DEA";
    e["Solver"]["Population Size"] = 1000;
    e["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 10'000;

    auto k = korali::Engine();


    k.run(e);
    return 0;
}
