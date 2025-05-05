#include "FluidSolver2D.cuh"

//basic funcs################1
FluidSolver2D::FluidSolver2D(int width, int height, float dx_, float dt_){
    gridWidth = width;
    gridHeight = height;
    w_x_h = gridWidth * gridHeight;
    blocksForCells = (w_x_h + threadsPerBlock - 1) / threadsPerBlock;
    blocksForVelocityGrid = ((gridWidth + 1) * (gridHeight + 1) + threadsPerBlock - 1) / threadsPerBlock;
    blocksForParticles = blocksForCells; //(!warning: this is an initial val not actual one!!!)
    dx = dx_;
    dy = dx;
    dt = dt_;
    std::cout << "dx=" << dx<<"\n";
    particles = new std::vector<Utility::Particle2D>();
}

FluidSolver2D::~FluidSolver2D(){
    // clean up

    delete particles;
}

//public funcs###############2

void FluidSolver2D::init(std::string fileName){
    std::ifstream file(fileName);

    // Читаем ширину и высоту
    file >> gridWidth >> gridHeight;
    w_x_h = gridWidth * gridHeight;
    blocksForCells = (w_x_h + threadsPerBlock - 1) / threadsPerBlock;
    blocksForVelocityGrid = ((gridWidth + 1) * (gridHeight + 1) + threadsPerBlock - 1) / threadsPerBlock;

    labels = std::vector<int>( w_x_h);
    p = std::vector<float>( w_x_h);
    u = std::vector<float>((gridWidth+1)*gridHeight, VEL_UNKNOWN);
    uSaved = std::vector<float>((gridWidth+1)*gridHeight, VEL_UNKNOWN);
    v = std::vector<float>(gridWidth*(gridHeight+1), VEL_UNKNOWN);
    vSaved = std::vector<float>(gridWidth*(gridHeight+1), VEL_UNKNOWN);

    // Читаем метки построчно
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Пропускаем пустые строки

        int col = 0;
        std::istringstream iss(line);
        char symbol;
        while (iss >> symbol) {
            int idx = row * gridHeight + col;
            switch (symbol) {
                case 'S': labels[idx] = Utility::SOLID; break;
                case 'F': labels[idx] = Utility::FLUID; break;
                case 'A': labels[idx] = Utility::AIR;   break;
                default:
                    throw std::runtime_error("Unknown cell symbol: " + std::string(1, symbol));
            }
            ++col;
        }
        ++row;
    }

    file.close();

    seedParticles(3, particles);
    blocksForParticles = (particles->size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout <<"Number of particles is" << particles->size() << std::endl;
    Utility::saveParticlesToPLY(*particles, "InputData/particles_0.ply");
    labelGrid();
    frameStep();
}


void FluidSolver2D::seedParticles(int particlesPerCell, std::vector<Utility::Particle2D>* particleList) {
    // Инициализация генератора (один раз вне функции!)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> subCellDist(0, 3);
    static std::uniform_real_distribution<> jitterDist(-0.24f, 0.24f);

    // Проходим по всем ячейкам с жидкостью
    for (int i = 0; i < gridWidth; ++i) {
        for (int j = 0; j < gridHeight; ++j) {
            if (labels[i * gridHeight + j] == Utility::FLUID) {
                float2 cellCenter = Utility::getGridCellPosition(i, j, dx);
                float2 subCenters[4] = {
                        {cellCenter.x - 0.25f * dx, cellCenter.y + 0.25f * dx}, // top-left
                        {cellCenter.x + 0.25f * dx, cellCenter.y + 0.25f * dx}, // top-right
                        {cellCenter.x + 0.25f * dx, cellCenter.y - 0.25f * dx}, // bottom-right
                        {cellCenter.x - 0.25f * dx, cellCenter.y - 0.25f * dx}  // bottom-left
                };

                // Равномерное распределение частиц по субрегионам
                for (int k = 0; k < particlesPerCell; ++k) {
                    // Случайный выбор субрегиона для каждой частицы
                    int subCellIdx = subCellDist(gen);

                    // Случайное смещение
                    float jitterX = jitterDist(gen) * dx;
                    float jitterY = jitterDist(gen) * dx;

                    // Позиция частицы
                    float2 pos{
                        subCenters[subCellIdx].x + jitterX,
                        subCenters[subCellIdx].y + jitterY
                    };

                    // Проверка границ (опционально)
                    pos.x = std::clamp(pos.x, cellCenter.x - 0.5f * dx, cellCenter.x + 0.5f * dx);
                    pos.y = std::clamp(pos.y, cellCenter.y - 0.5f * dx, cellCenter.y + 0.5f * dx);
                    particleList->emplace_back(pos, float2{0.0f, 0.0f});
                }
            }
        }
    }
    blocksForParticles = (particles->size() + threadsPerBlock - 1) / threadsPerBlock;
}

__global__ void labelCellWrap(int* labels, Utility::Particle2D* particles, float dx, int gridHeight){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    labelCellClean(ind, labels);
}

__device__ void labelCellFluid(int ind, int* labels, Utility::Particle2D* particles, float dx, int gridHeight){
    int cellInd = Utility::getGridCellIndex(particles[ind].pos, dx, gridHeight);
    labels[cellInd] = Utility::FLUID;
}

__device__ void labelCellClean(int ind, int* labels){
    if(labels[ind] != Utility::SOLID){
        labels[ind] = Utility::AIR;
    }
}

void FluidSolver2D::labelGrid() {
    int* labels_for_device;
    cudaMalloc(&labels_for_device, sizeof(int)*w_x_h);
    cudaMemcpy(labels_for_device, labels.data(), sizeof(int)*w_x_h, cudaMemcpyHostToDevice);
    labelCellWrap<<<blocksForCells, threadsPerBlock>>>(labels_for_device, particles->data(), dx, gridHeight);
    cudaDeviceSynchronize();
    cudaFree(labels_for_device);
    /*DEBUG COUT
     * for(int i = 0; i < gridWidth; ++i){
        for(int j =0; j < gridHeight; ++j){
            int idx = i * gridHeight + j;
            switch (labels[idx]) {
                case Utility::SOLID: std::cout<<"S"; break;
                case Utility::FLUID: std::cout << "F"; break;
                case Utility::AIR: std::cout << "A" ;   break;
                default:
                    throw std::runtime_error("Unknown cell val");
            }
        }
        std::cout<<std::endl;
    }*/
}



__global__ void accumulateDenAndNum( Utility::Particle2D particle, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight, float dx){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    // save check
    if(ind < uSize){
        int2 uCellInd = Utility::getGridIndicesU(ind, gridHeight);
        float kernel = Utility::bilinearHatKernel(particle.pos- Utility::getGridCellPosition_device(uCellInd.x, uCellInd.y, dx), dx, dx );
        atomicAdd(&uNum[ind], particle.vel.x * kernel);
        atomicAdd(&uDen[ind], kernel);
    }
    if(ind < vSize){
        int2 vCellInd = Utility::getGridIndicesV(ind, gridHeight);
        float kernel = Utility::bilinearHatKernel(particle.pos- Utility::getGridCellPosition_device(vCellInd.x, vCellInd.y, dx), dx, dx );
        atomicAdd(&vNum[ind], particle.vel.y * kernel);
        atomicAdd(&vDen[ind], kernel);
    }
}

__global__ void applyNumDen(float* u_device, float* v_device, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if(ind < uSize){
        if(uDen[ind] != 0.0f){
            u_device[ind] = uNum[ind] / uDen[ind];
        }
    }
    if(ind < vSize){
        if(vDen[ind] != 0.0f){
            v_device[ind] = vNum[ind] / vDen[ind];
        }
    }
}

void FluidSolver2D::particlesToGrid(){

    //clear velocities
    int uSize = (gridWidth+1)*gridHeight;
    int vSize = gridWidth*(gridHeight+1);
    u.assign(uSize,VEL_UNKNOWN);
    v.assign(vSize,VEL_UNKNOWN);

    float* uNum;
    float* uDen;
    float* vNum;
    float* vDen;
    float* u_device;
    float* v_device;

    cudaMalloc(&uNum, sizeof(float)*uSize);
    cudaMalloc(&uDen, sizeof(float)*uSize);
    cudaMalloc(&vNum, sizeof(float)*vSize);
    cudaMalloc(&vDen, sizeof(float)*vSize);
    cudaMalloc(&u_device, sizeof(float )*uSize);
    cudaMalloc(&v_device, sizeof(float )*vSize);

    std::vector<float> zerosForU(uSize, 0.0f);
    std::vector<float> zerosForV(vSize, 0.0f);
    cudaMemcpy(uNum, zerosForU.data(), sizeof(float)*uSize, cudaMemcpyHostToDevice);
    cudaMemcpy(uDen, zerosForU.data(), sizeof(float)*uSize, cudaMemcpyHostToDevice);
    cudaMemcpy(vNum, zerosForV.data(), sizeof(float)*vSize, cudaMemcpyHostToDevice);
    cudaMemcpy(vDen, zerosForV.data(), sizeof(float)*vSize, cudaMemcpyHostToDevice);
    cudaMemcpy(u_device, zerosForU.data(), sizeof(float)*uSize, cudaMemcpyHostToDevice);
    cudaMemcpy(v_device, zerosForV.data(), sizeof(float)*vSize, cudaMemcpyHostToDevice);

    for(int pInd = 0; pInd < particles->size(); ++pInd){
        Utility::Particle2D curParticle = particles->at(pInd);
        accumulateDenAndNum<<<blocksForVelocityGrid, threadsPerBlock>>>(curParticle, uNum, uDen, vNum, vDen, uSize, vSize, gridWidth, gridHeight, dx);
        cudaDeviceSynchronize();
    }

    applyNumDen<<<blocksForVelocityGrid, threadsPerBlock>>>(u_device,v_device,uNum,uDen,vNum,vDen,uSize,vSize, gridWidth,gridHeight);
    cudaDeviceSynchronize();
    cudaMemcpy(u.data(), u_device, sizeof(float)*uSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), v_device, sizeof(float)*vSize, cudaMemcpyDeviceToHost);

    cudaFree( uNum);
    cudaFree( uDen);
    cudaFree( vNum);
    cudaFree( vDen);
    cudaFree(u_device);
    cudaFree(v_device);
}

void FluidSolver2D::saveVelocities(){
    // save u grid
    for (int i = 0; i < gridWidth + 1; i++) {
        for (int j = 0; j < gridHeight; j++) {
            uSaved[i*gridHeight + j] = u[i*gridHeight + j];
        }
    }

    // save v grid
    for (int i = 0; i < gridWidth; i++) {
        for (int j = 0; j < gridHeight + 1; j++) {
            vSaved[i*gridHeight + j] = v[i*gridHeight + j];
        }
    }
}

void FluidSolver2D::applyForces() {
    // traverse all grid cells and apply force to each velocity component
    // The new velocity is calculated using forward euler
    for (int i = 0; i < gridWidth + 1; i++) {
        for (int j = 0; j < gridHeight + 1; j++) {
            if (j < gridHeight) {
                // make sure we know the velocity
                if (u[i*gridHeight + j] != VEL_UNKNOWN) {
                    // update u component
                    u[i*gridHeight + j] += dt*GRAVITY.x;
                }
            }
            if (i < gridWidth) {
                if (v[i*gridHeight + j] != VEL_UNKNOWN) {
                    // update v component
                    v[i*gridHeight + j] += dt*GRAVITY.y;
                }
            }
        }
    }
}

void FluidSolver2D::constructRHS(std::vector<float>& rhs){
    // calculate negative divergence
    float scale = 1.0f / dx;
    for (int i = 0; i < gridWidth; i++) {
        for (int j = 0; j < gridHeight; j++) {
            if (isFluid(i, j)) {
                rhs[i*gridHeight+j] = -scale * (u[(i + 1)*gridHeight+j] - u[i*gridHeight+j] + v[i*gridHeight + (j + 1)] - v[i*gridHeight+j]);
                // if it's on boundary must update to consider solid velocity
                // TODO create actual solid velocity grids, for right now just 0
                if (labels[(i - 1)*gridHeight+j] == Utility::SOLID) {
                    rhs[i*gridHeight+j] -= scale * (u[i*gridHeight+j] - 0.0f); //m_usolid[i][j]
                }
                if (labels[(i + 1)*gridHeight+j] == Utility::SOLID) {
                    rhs[i*gridHeight+j] += scale * (u[(i + 1)*gridHeight+j] - 0.0f); //m_usolid[i+1][j]
                }
                if (labels[i*gridHeight+(j - 1)] == Utility::SOLID) {
                    rhs[i*gridHeight+j] -= scale * (v[i*gridHeight+j] - 0.0f); //m_vsolid[i][j]
                }
                if (labels[i*gridHeight+(j + 1)] == Utility::SOLID) {
                    rhs[i*gridHeight+j] += scale * (v[i*gridHeight+(j + 1)] - 0.0f); //m_vsolid[i][j+1]
                }
            }
        }
    }

}

/*
Constructs the A matrix for the system to solve for pressure. This a sparse coefficient matrix
for the pressure terms, stored in 3 separate grids. If index i, j, k is not a fluid cell, then
it is 0.0 in all 3 grids that store the matrix.
Args:
Adiag - grid to store the diagonal of the matrix in.
Ax - grid to store the coefficients for pressure in the (i+1) cell for each grid cell with x index i
Ay - grid to store the coefficients for pressure in the (j+1) cell for each grid cell with y index j
*/
void FluidSolver2D::constructA(std::vector<float>& Adiag, std::vector<float>& Ax, std::vector<float>& Ay) {

    // populate with coefficients for pressure unknowns
    float scale = dt / (FLUID_DENSITY * dx * dx);
    for (int i = 0; i < gridWidth; ++i) {
        for (int j = 0; j < gridHeight; ++j) {
            if (isFluid(i, j)) {
                // handle negative x neighbor
                if (labels[(i - 1)*gridHeight+j] == Utility::FLUID || labels[(i - 1)*gridHeight+j] == Utility::AIR) {
                    Adiag[i*gridHeight+j] += scale;
                }
                // handle positive x neighbor
                if (labels[(i + 1)*gridHeight+j] == Utility::FLUID) {
                    Adiag[i*gridHeight+j] += scale;
                    Ax[i*gridHeight+j] = -scale;
                } else if (labels[(i + 1)*gridHeight+j] == Utility::AIR) {
                    Adiag[i*gridHeight+j] += scale;
                }
                // handle negative y neighbor
                if (labels[i*gridHeight +(j - 1)] == Utility::FLUID || labels[i*gridHeight +(j - 1)] == Utility::AIR) {
                    Adiag[i*gridHeight+j] += scale;
                }
                // handle positive y neighbor
                if (labels[i*gridHeight +(j + 1)] == Utility::FLUID) {
                    Adiag[i*gridHeight+j] += scale;
                    Ay[i*gridHeight+j] = -scale;
                } else if (labels[i*gridHeight +(j + 1)] == Utility::AIR) {
                    Adiag[i*gridHeight+j] += scale;
                }
            }
        }
    }
}

void FluidSolver2D::pressureSolve() {
    std::vector<float> rhs(w_x_h, 0.0f);
    constructRHS(rhs);
    std::vector<float> Adiag(w_x_h, 0.0f);
    std::vector<float> Ax(w_x_h, 0.0f);
    std::vector<float> Ay(w_x_h, 0.0f);
    constructA(Adiag, Ax, Ay);
    //TODO:
    //precon = constructPrecon()
    //PCG(Adiag, Ax, Ay, rhs, precon);
}

void FluidSolver2D::applyPressure() {
    float scale = dt / (FLUID_DENSITY * dx);
    for (int i = 0; i < gridWidth; ++i) {
        for (int j = 0; j < gridHeight; ++j) {
            // update u
            if (i - 1 >= 0) {
                if (labels[(i - 1)*gridHeight + j] == Utility::FLUID || labels[i*gridHeight +j] == Utility::FLUID) {
                    if (labels[(i - 1)*gridHeight + j] == Utility::SOLID ||  labels[i*gridHeight +j] == Utility::SOLID) {
                        // TODO add solid velocities
                        u[i*gridHeight + j] = 0.0f; // usolid[i][j]
                    } else {
                        u[i*gridHeight + j] -= scale * (p[i*gridHeight + j] - p[(i - 1)*gridHeight+j]);
                    }
                } else {
                    u[i*gridHeight + j] = VEL_UNKNOWN;
                }
            } else {
                // edge of grid, keep the same velocity
            }

            // update v
            if (j - 1 >= 0) {
                if (labels[i * gridHeight + (j - 1)] == Utility::FLUID || labels[i*gridHeight +j] == Utility::FLUID) {
                    if (labels[i * gridHeight + (j - 1)] == Utility::SOLID || labels[i*gridHeight +j] == Utility::SOLID) {
                        // TODO add solid velocities
                        v[i*gridHeight +j] = 0.0f; // vsolid[i][j]
                    }
                    else {
                        v[i*gridHeight +j] -= scale * (p[i*gridHeight +j] - p[i * gridHeight + (j - 1)]);
                    }
                } else {
                    v[i*gridHeight +j] = VEL_UNKNOWN;
                }
            } else {
                // edge of grid, keep the same velocity
            }
        }
    }
}

void FluidSolver2D::gridToParticles(float alpha) {
    //TODO
}

//C - the maximum number of grid cells a particle should move when advected. This helps define substep sizes.
void FluidSolver2D::advectParticles(int C) {
    //TODO
}

//delta - the amount to project stray particles away from the wall.
void FluidSolver2D::cleanUpParticles(float delta) {
    //TODO
}

void FluidSolver2D::frameStep(){
    labelGrid();

    //particles velocities to grid
    particlesToGrid();

    //saving a copy of the current grid velocities (for FLIP)
    saveVelocities();

    //applying body forces on grid (e.g. gravity force)
    applyForces();

    pressureSolve();
    applyPressure();

    //grid velocities to particles
    gridToParticles(PIC_WEIGHT);

    //advection of particles
    advectParticles(ADVECT_MAX);

    //boundary penetration detection (if so --- move back inside)
    cleanUpParticles(dx/4.0f);
}

/*
Determines if the given grid cell is considered a fluid based on the label grid. Also takes
into account velocity components on the edge of the grid. For example, if the index passed in
is one cell outside the label grid, it is assumed to be a velocity component index, and whether the cell is
fluid or not is determined by the cell that it borders. Otherwise false is returned.
Args
i - x cell index
j - y cell index
*/
bool FluidSolver2D::isFluid(int i, int j) {
    bool isFluid = false;
    // see if velocity on edge of grid
    // if it is we can't check the label at that point, must check the one previous
    if (i == gridWidth || j == gridHeight) {
        // i and j should never both be out of label range
        // should only happen in one dimension because of vel comp grids
        if (i == gridWidth && j == gridHeight) {
            isFluid = false;
        }
        else if (i == gridWidth) {
            if (labels[(i - 1)*gridHeight + j] == Utility::FLUID) {
                isFluid = true;
            }
        }
        else if (j == gridHeight) {
            if (labels[i*gridHeight + (j - 1)] == Utility::FLUID) {
                isFluid = true;
            }
        }
    }
    else if (labels[i*gridHeight + j] == Utility::FLUID) {
        isFluid = true;
    }

    return isFluid;
}