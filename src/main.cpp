#include <iostream>

#include "korali.hpp"
#include "model/model.hpp"
#include "model/src/config.h"

int main(int argc, char **argv) {

    auto e = korali::Experiment();


//    e["Problem"]["Type"] = "Optimization/Constrained";
    e["Problem"]["Type"] = "Optimization/Stochastic";
//    e["Problem"]["Objective"] = "Maximize";
    e["Problem"]["Objective Function"] = &fitness;


// Defining the problem's variables.
    e["Variables"][0]["Name"] = "THREAD_TILE_M";
    e["Variables"][0]["Lower Bound"] = 1.0;
    e["Variables"][0]["Upper Bound"] = 20.0;
    e["Variables"][0]["Granularity"] = 1.0;

    e["Variables"][1]["Name"] = "THREAD_TILE_N";
    e["Variables"][1]["Lower Bound"] = 1.0;
    e["Variables"][1]["Upper Bound"] = 20.0;
    e["Variables"][1]["Granularity"] = 1.0;

    e["Variables"][2]["Name"] = "WARPTILE_MULTIPLICATOR_M";
    e["Variables"][2]["Lower Bound"] = 1.0;
    e["Variables"][2]["Upper Bound"] = 10.0;
    e["Variables"][2]["Granularity"] = 2.0;

    e["Variables"][3]["Name"] = "WARPTILE_MULTIPLICATOR_N";
    e["Variables"][3]["Lower Bound"] = 1.0;
    e["Variables"][3]["Upper Bound"] = 10.0;
    e["Variables"][3]["Granularity"] = 1.0;


    e["Variables"][4]["Name"] = "THREADBLOCK_MULTIPLICATOR_M";
    e["Variables"][4]["Lower Bound"] = 1.0;
    e["Variables"][4]["Upper Bound"] = 10.0;
    e["Variables"][4]["Granularity"] = 2.0;

    e["Variables"][5]["Name"] = "THREADBLOCK_MULTIPLICATOR_N";
    e["Variables"][5]["Lower Bound"] = 1.0;
    e["Variables"][5]["Upper Bound"] = 10.0;
    e["Variables"][5]["Granularity"] = 1.0;

    e["Variables"][6]["Name"] = "LOAD_K";
    e["Variables"][6]["Lower Bound"] = 2.0;
    e["Variables"][6]["Upper Bound"] = 64.0;
    e["Variables"][6]["Granularity"] = 2.0;




    e["Variables"][7]["Name"] = "SPLIT_K";
    e["Variables"][7]["Lower Bound"] = 1.0;
    e["Variables"][7]["Upper Bound"] = 20.0;
    e["Variables"][7]["Granularity"] = 1.0;

    // TODO Decide which solver to use
    e["Solver"]["Type"] = "DEA";
    e["Solver"]["Population Size"] = 5;

//    e["Solver"]["Type"] = "CMAES";
//    e["Solver"]["Population Size"] = 9;
    e["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 100;
    e["Solver"]["Termination Criteria"]["Max Generations"] = 10;

    auto k = korali::Engine();


    k.run(e);
    return 0;
}
