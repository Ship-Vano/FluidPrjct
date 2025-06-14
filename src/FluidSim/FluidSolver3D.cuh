#ifndef FLUIDSOLVER3D_H
#define FLUIDSOLVER3D_H

#include "utility.cuh"

#include <limits.h>
#include <float.h>
#include <fstream>
#include <random>
#include <cassert>


class FluidSolver3D{
private:
    int w_x_h_x_d;

    //nx
    int gridWidth;

    //ny
    int gridHeight;

    //nz
    int gridDepth;

    // distance between each grid cell
    float dx;
    float dy;
    float dz;

    // simulation time step
    float dt;

    // cell labels
    Utility::Grid3D<int>  labels;

    std::vector<float> levelSet;

    // pressure and velocity are held in a MAC grid so that
    // p(i, j, k) = p_i_j_k
    // u(i, j, k) = u_i-1/2_j_k
    // v(i, j, k) = v_i_j-1/2_k
    //pressures, size (nx, ny)
    Utility::Grid3D<float> p;

    // grid of vel x component, size (nx+1, ny, nz)
    Utility::Grid3D<float>  u;

    // grid of vel y component, size (nx, ny+1, nz)
    Utility::Grid3D<float>  v;

    //grid of vel z component, size  (nx, ny, nz+1)
    Utility::Grid3D<float>  w;

    // saved grid of vel x component for FLIP update, size (nx+1, ny, nz)
    Utility::Grid3D<float>  uSaved;

    // saved grid of vel y component for FLIP update, size (nx, ny+1, nz)
    Utility::Grid3D<float>  vSaved;

    // saved grid of vel z component for FLIP update, size (nx, ny, nz+1)
    Utility::Grid3D<float>  wSaved;

    // new renumbering of fluid cells for pressure solve fluidNumbers(i + j* gridWidth + k * gridWidth * gridHeight) = new_index in A indexing
    // Important! new_ind = -1 => a cell is NOT a FLUID one.
    //Utility::Grid3D<int>  fluidNumbers; as a temp var now
    int fluidCellsAmount;

    //Simulation parameters

    // number of particles to seed in each cell at start of sim
    const int PARTICLES_PER_CELL = 6;
    // the amount of weight to give to PIC in PIC/FLIP update

    // the maximum number of grid cells a particle should move when advected
    const int ADVECT_MAX = 1;
    // acceleration due to gravity
    const float3 GRAVITY{ 0.0f, -9.81f, 0.0f };
    // density of the fluid (kg/m^3)
    const float FLUID_DENSITY = 1000.0f;

    //gpu params
    // Размеры блоков
    dim3 blockSize3D;       // Для 3D сеток (например, 8x8x8)
    int threadsPerBlock1D;  // Для одномерных операций (частицы)

    // Размеры гридов
    dim3 gridSizeCells;     // Для ячеек (pressure, labels)
    dim3 gridSizeU;         // Для U-компоненты скорости
    dim3 gridSizeV;         // Для V-компоненты скорости
    dim3 gridSizeW;         // Для W-компоненты скорости

    int blocksForParticles;
    int threadsPerBlock = 256;

    //cudss vars
    cudaStream_t stream;
    cudssHandle_t handle;
    cudssConfig_t solverConfig;
    cudssData_t solverData;


    // list of all particles in the simulation
    thrust::host_vector<Utility::Particle3D> h_particles;
    thrust::device_vector<Utility::Particle3D> d_particles;

    // FUNCTIONS
    // solver steps
    __host__ void seedParticles(int particlesPerCell);
    int labelGrid();
    void frameStep();
    void particlesToGrid();
    void saveVelocities();
    void applyForces();
    int pressureSolve();
    void applyPressure();
    void gridToParticles(float alpha);
    void advectParticles(float C);
    void cleanUpParticles(float delta);
    void extrapolateGridFluidData(std::vector<float>& grid, int x, int y, int depth);
    // helpers
    bool isFluid(int i, int j, int k);
    bool isCellValid(int x, int y, int z);
    void constructRHS(thrust::device_vector<float>& rhs, const thrust::device_vector<int>& fluidNumbers, const thrust::device_vector<int>& fluidFlags);
    void constructA(thrust::device_vector<float>& csr_values, thrust::device_vector<int>& csr_columns, thrust::device_vector<int>& csr_offsets, thrust::device_vector<int> fluidNumbers);
    float2 interpVel(std::vector<float>& uGrid, std::vector<float>& vGrid, std::vector<float>& wGrid, float3 pos);
    bool projectParticle(Utility::Particle3D* particle, float max_h);
    void RK3(Utility::Particle3D *particle, float3 initVel, float dt, std::vector<float>& uGrid, std::vector<float>& vGrid, std::vector<float>& wGrid);
    std::vector<int> checkNeighbors(std::vector<int> grid, int3 dim, int3 index, int neighbors[][3], int numNeighbors, int value);

public:
    float PIC_WEIGHT = 0.5f; // changed in the main
    int iterPerFrame = 10; // iterations to generate one frame (file out each iterPerFrame steps)
    /*
	Creates a new 2D fluid solver.
	Args:
	width - width of the grid to use
	height - height of the grid to use
    depth
	dx - the grid cell width
	dt - the timestep to use
	*/
    FluidSolver3D(int, int, int, float, float);
    ~FluidSolver3D();

    /*
     Init a solver
     Args: name of a geometry file
     * */
    __host__ void init(const std::string&);

    __host__ void run(int max_steps);
};

#endif