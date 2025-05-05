#pragma once
#include "utility.cuh"
#include <limits.h>
#include <fstream>
#include <random>

__global__ void labelCellWrap(int* labels, Utility::Particle2D* particles, float dx, int gridHeight);
__device__ void labelCellFluid(int ind, int* labels, Utility::Particle2D* particles, float dx, int gridHeight);
__device__ void labelCellClean(int ind, int* labels);
__global__ void accumulateDenAndNum( Utility::Particle2D particle, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight, float dx);
__global__ void applyNumDen(float* u_device, float* v_device, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight);

class FluidSolver2D{
private:
    int w_x_h;
    //nx
    int gridWidth;
    //ny
    int gridHeight;
    // distance between each grid cell
    float dx;
    float dy;
    // simulation time step
    float dt;
    // cell labels
    std::vector<int> labels;

    std::vector<float> levelSet;
    // pressure and velocity are held in a MAC grid so that
    // p(i, j, k) = p_i_j_k
    // u(i, j, k) = u_i-1/2_j_k
    // v(i, j, k) = v_i_j-1/2_k

    //pressures, size (nx, ny)
    std::vector<float> p;
    // grid of vel x component, size (nx+1, ny)
    std::vector<float> u;
    // grid of vel y component, size (nx, ny+1)
    std::vector<float> v;

    // saved grid of vel x component for FLIP update, size (nx+1, ny)
    std::vector<float> uSaved;
    // saved grid of vel y component for FLIP update, size (nx, ny+1)
    std::vector<float> vSaved;

    //Simulation parameters
    const int VEL_UNKNOWN = INT_MIN;
    // number of particles to seed in each cell at start of sim
    const int PARTICLES_PER_CELL = 4;
    // the amount of weight to give to PIC in PIC/FLIP update
    const float PIC_WEIGHT = 0.02f;
    // the maximum number of grid cells a particle should move when advected
    const int ADVECT_MAX = 1;
    // acceleration due to gravity
    const float2 GRAVITY{ 0.0f, -9.81f };
    // density of the fluid (kg/m^3)
    const float FLUID_DENSITY = 1000.0f;
    // error tolerance for PCG
    const float PCG_TOL = 0.000001f;
    // max iterations for PCG
    const int PCG_MAX_ITERS = 200;

    //gpu params
    int threadsPerBlock = 256;
    int blocksForCells;
    int blocksForVelocityGrid;
    int blocksForParticles;

    // list of all particles in the simulation
    std::vector<Utility::Particle2D> *particles;

    // FUNCTIONS
    // solver steps
    void seedParticles(int particlesPerCell, std::vector<Utility::Particle2D>* particleList);
    void labelGrid();
    void frameStep();
    void particlesToGrid();
    void saveVelocities();
    void applyForces();
    void pressureSolve();
    void applyPressure();
    void gridToParticles(float alpha);
    void advectParticles(int C);
    void cleanUpParticles(float delta);
    // helpers
    bool isFluid(int i, int j);
    void constructRHS(std::vector<float>& rhs);
    void constructA(std::vector<float>& Adiag, std::vector<float>& Ax, std::vector<float>& Ay);

public:
    /*
	Creates a new 2D fluid solver.
	Args:
	width - width of the grid to use
	height - height of the grid to use
	dx - the grid cell width
	dt - the timestep to use
	*/
    FluidSolver2D(int, int, float, float);
    ~FluidSolver2D();

    /*
     Init a solver
     Args: name of a geometry file
     * */
    void init(std::string);

};