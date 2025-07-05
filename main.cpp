//
// Created by ivan on 06.04.2025.
//
#include "src/FluidSim/test.cuh"
#include "src/FluidSim/utility.cuh"
#include "src/json/json.h"
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

    int frameAmount = json_root.get("frameAmount", 100).asInt();

    std::cout << "alpha = " << solver.PIC_WEIGHT << std::endl;

    // body setup
    float body_x = json_root.get("body_x", 0.5).asFloat();
    float body_y = json_root.get("body_y", 0.5).asFloat();
    float body_z = json_root.get("body_z", 0.5).asFloat();
    float3 body_pos = make_float3(body_x,body_y,body_z);
    float body_mass = json_root.get("body_mass", 50.0f).asFloat();
    solver.initialBodyPos = body_pos;
    solver.bodyMass = body_mass;

    solver.init(inputLabelData);
    solver.run(frameAmount);

    return 0;
}
