//
// Created by ivan on 06.04.2025.
//
#include "src/FluidSim/test.cuh"
#include "src/FluidSim/utility.cuh"
#include "src/json/json.h"
#include "src/FluidSim/FluidSolver2D.cuh"
#include "src/FluidSim/FluidSolver3D.cuh"

int main(){
    //cuDSStest();
    //out();
    float dt = 0.01f;
    float dx = 0.5f;

    FluidSolver3D solver(5, 5, 5, dx, dt);
    solver.PIC_WEIGHT = (6.0f * dt)/(dx*dx);
    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;
    solver.init("InputData/labels_3d_sphere.txt");
    solver.run(100);

//    FluidSolver2D solver2d(5, 5, dx, dt);
//    solver2d.PIC_WEIGHT = (6.0f * dt)/(dx*dx);
//    std::cout << "alpha = " << solver2d.PIC_WEIGHT << std::endl;
//    solver2d.init("InputData/labels_simple.txt");
//    solver2d.run(1);
//    std::cout << "success!" << std::endl;

    return 0;
}
