#pragma once
#include "utility.cuh"
#include <limits.h>
#include <float.h>
#include <fstream>
#include <random>
#include <cassert>


__global__ void labelCellWrap(int* labels, Utility::Particle2D* particles, float dx, int gridHeight);
__global__ void clearLabelsKernel(int* labels, int gridWidth, int gridHeight);
__global__ void markFluidCellsKernel(const float2* particles, int numParticles,
                                     float dx, int gridWidth, int gridHeight, int* labels);
__global__ void accumulateDenAndNum( Utility::Particle2D particle, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight, float dx);
__global__ void applyNumDen(float* u_device, float* v_device, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight);
__global__ void interpVelKernel(
        const float* uGrid,
        const float* vGrid,
        const float2* particles,
        float2* particleVelocities,
        int numParticles,
        float dx,
        int gridWidth,
        int gridHeight
);
__global__ void computeDeltaUGridKernel(
        const float* u,
        const float* uSaved,
        float* duGrid,
        int gridWidth,
        int gridHeight
);
__global__ void computeDeltaVGridKernel(
        const float* v,
        const float* vSaved,
        float* dvGrid,
        int gridWidth,
        int gridHeight
);
__global__ void updateParticleVelocitiesKernel(
        float2* particleVelocities,
        const float2* duGridInterp,
        const float2* dvGridInterp,
        int numParticles,
        float alpha
);

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

    // new renumbering of fluid cells for pressure solve fluidNumbers(i * gridHeight + j) = new_index in A indexing
    // Important! new_ind = -1 => a cell is NOT a FLUID one.
    std::vector<int> fluidNumbers;
    int fluidCellsAmount;

    //Simulation parameters

    // number of particles to seed in each cell at start of sim
    const int PARTICLES_PER_CELL = 6;
    // the amount of weight to give to PIC in PIC/FLIP update

    // the maximum number of grid cells a particle should move when advected
    const int ADVECT_MAX = 1;
    // acceleration due to gravity
    const float2 GRAVITY{ 0.0f, -9.81f };
    // density of the fluid (kg/m^3)
    const float FLUID_DENSITY = 1000.0f;

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
    int labelGrid();
    int labelGrid_gpu();
    void frameStep();
    void particlesToGrid();
    void saveVelocities();
    void applyForces();
    int pressureSolve();
    void applyPressure();
    void gridToParticles(float alpha);
    void gridToParticles_gpu(float alpha);
    void advectParticles(float C);
    void cleanUpParticles(float delta);
    void extrapolateGridFluidData(std::vector<float>& grid, int x, int y, int depth);
    // helpers
    bool isFluid(int i, int j);
    bool isCellValid(int x, int y);
    void constructRHS(std::vector<float>& rhs);
    void constructA(std::vector<float>& csr_values, std::vector<int>& csr_columns, std::vector<int>& csr_offsets);
    float2 interpVel(std::vector<float>& uGrid, std::vector<float>& vGrid, float2 pos);
    bool projectParticle(Utility::Particle2D* particle, float max_h);
    void RK3(Utility::Particle2D *particle, float2 initVel, float dt, std::vector<float>& uGrid, std::vector<float>& vGrid);
    std::vector<int> checkNeighbors(std::vector<int> grid, int2 dim, int2 index, int neighbors[][2], int numNeighbors, int value);

    public:
    float PIC_WEIGHT = 0.5f;
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

    void run(int max_steps);
};