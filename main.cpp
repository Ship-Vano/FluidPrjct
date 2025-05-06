//
// Created by ivan on 06.04.2025.
//
#include "src/FluidSim/test.cuh"
#include "src/FluidSim/utility.cuh"
#include "src/json/json.h"
#include "src/FluidSim/FluidSolver2D.cuh"

int main(){
    //cuDSStest();
    //out();
    FluidSolver2D solver(512, 512, 0.4, 0.001);
    solver.init("InputData/labels_simple.txt");
    //solver.run(10000);
    std::cout << "success!" << std::endl;
    return 0;
}
