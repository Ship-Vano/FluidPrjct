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

    FluidSolver3D solver(50, 50, 50, dx, dt);
    solver.PIC_WEIGHT = (6.0f * dt)/(dx*dx);
    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;
    solver.init("InputData/labels_3d_sphere.txt");
    solver.run(100);

    /*FluidSolver2D solver(100, 100, dx, dt);
    solver.PIC_WEIGHT = (6.0f * dt)/(dx*dx);
    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;
    solver.init("InputData/labels_colimn.txt");
    solver.run(3000);*/
    std::cout << "success!" << std::endl;

    struct is_odd
    {
        __host__ __device__
        bool operator()(int &x)
        {
            return x & 1;
        }
    };



    return 0;
}
