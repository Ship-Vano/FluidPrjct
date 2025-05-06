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
    FluidSolver2D solver(512, 512, 0.9, 0.1);
    solver.init("InputData/test1.txt");
    std::cout << "success!" << std::endl;
    return 0;
}
