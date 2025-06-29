//
// Created by ivan on 06.04.2025.
//
#include "src/FluidSim/test.cuh"
#include "src/FluidSim/utility.cuh"
#include "src/json/json.h"
#include "src/FluidSim/FluidSolver2D.cuh"
#include "src/FluidSim/FluidSolver3D.cuh"

int main(){

    std::string configPath = "InputData/config.json";
    std::ifstream in(configPath, std::ios::in);
    Json::Reader json_reader;
    Json::Value json_root;
    bool read_succeeded = json_reader.parse(in, json_root);
    assert(read_succeeded);

    float dt = json_root.get("dt", 0.02f).asFloat();
    float dx = json_root.get("dx", 0.1f).asFloat();

    FluidSolver3D solver(5, 5, 5, dx, dt);

    solver.PIC_WEIGHT = json_root.get("PIC_WEIGHT", 0.01f).asFloat();
    solver.iterPerFrame = json_root.get("iterPerFrame", 1).asInt();

    std::string outFmt = json_root.get("outputFormat", "PLY").asString();
    if(outFmt == "PLY"){
        solver.outputFormat = PLY;
    }else{
        solver.outputFormat = OFF;
    }

    solver.outputTemplate = json_root.get("outputTemplate", "InputData/particles_").asString();

    std::string inputLabelData = json_root.get("inputLabelData", "InputData/labels.txt").asString();
    assert(inputLabelData != "");
    solver.init(inputLabelData);

    int frameAmount = json_root.get("frameAmount", 100).asInt();

    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;

    solver.run(frameAmount);

    //cuDSStest();
    //out();
    //float dx = 1.0f/75.0f;
    //dt = dx*0.1;
//    FluidSolver2D solver2d(5, 5, dx, dt);
//    solver2d.PIC_WEIGHT = (6.0f * dt)/(dx*dx);
//    std::cout << "alpha = " << solver2d.PIC_WEIGHT << std::endl;
//    solver2d.init("InputData/labels_simple.txt");
//    solver2d.run(1);
//    std::cout << "success!" << std::endl;

    return 0;
}
