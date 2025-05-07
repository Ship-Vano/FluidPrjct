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
    float dt = 0.01;
    float dx = 0.5;
    FluidSolver2D solver(512, 512, dx, dt);
    solver.PIC_WEIGHT = (6 * dt)/(dx*dx);
    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;
    solver.init("InputData/labels_simple.txt");
    solver.run(3000);
    std::cout << "success!" << std::endl;
    return 0;
}
