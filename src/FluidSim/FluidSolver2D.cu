#include "FluidSolver2D.cuh"

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("CALL FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("CALL FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);

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

/**
 * ########################## ЭТАП "ПОДГОТОВИТЕЛЬНЫЙ"
 * ########################################
 * ########################################
 * */
void FluidSolver2D::init(std::string fileName){
    std::ifstream file(fileName);
    assert(file.is_open());
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

    fluidNumbers = std::vector<int>(w_x_h, -1);

    std::string line;
    int totalRows = gridHeight - 1; // Индекс последней строки (низ сетки)
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Обрабатываем строки снизу вверх
        int currentRow = totalRows--;
        if (currentRow < 0) {
            throw std::runtime_error("File has more lines than grid height");
        }

        int col = 0;
        std::istringstream iss(line);
        char symbol;
        while (iss >> symbol && col < gridWidth) {
            // Индекс рассчитывается как: column + row * gridWidth
            int idx = col + currentRow * gridWidth;

            switch (symbol) {
                case 'S': labels[idx] = Utility::SOLID; break;
                case 'F': labels[idx] = Utility::FLUID; break;
                case 'A': labels[idx] = Utility::AIR;   break;
                default:
                    throw std::runtime_error("Unknown cell symbol: " + std::string(1, symbol));
            }
            ++col;
        }

        if (col != gridWidth) {
            throw std::runtime_error("Line width does not match grid width");
        }
    }


    file.close();

    seedParticles(PARTICLES_PER_CELL, particles);
    blocksForParticles = (particles->size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout <<"Number of particles is" << particles->size() << std::endl;
    Utility::saveParticlesToPLY(*particles, "InputData/particles_0.ply");
    labelGrid();
    frameStep();
}


void FluidSolver2D::seedParticles(int particlesPerCell, std::vector<Utility::Particle2D>* particleList) {
    particleList->clear();
    // Инициализация генератора (один раз вне функции!)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> subCellDist(0, 3);
    static std::uniform_real_distribution<> jitterDist(-0.24f, 0.24f);

    // Проходим по всем ячейкам с жидкостью
    for (int j = 0; j < gridHeight; ++j) {
    for (int i = 0; i < gridWidth; ++i) {
            if (labels[i + j*gridWidth] == Utility::FLUID) {
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


/**
 * ########################## ЭТАП "ОПРЕДЕЛЕНИЕ ТИПА ЯЧЕЕК"
 * ########################################
 * ########################################
 * */

// Ядро для очистки сетки
__global__ void clearLabelsKernel(int* labels, int gridWidth, int gridHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < gridWidth && j < gridHeight) {
        int index = i + j * gridWidth;
        if (labels[index] != Utility::SOLID) {
            labels[index] = Utility::AIR;
        }
    }
}


// Ядро для пометки ячеек с частицами
__global__ void markFluidCellsKernel(const Utility::Particle2D* particles, int numParticles,
                                     float dx, int gridWidth, int gridHeight, int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float2 pos = particles[idx].pos;
    int2 cell =  make_int2((int)(pos.x / dx), (int)(pos.y/dx));

    // Проверка границ
    if (cell.x >= 0 && cell.x < gridWidth && cell.y >= 0 && cell.y < gridHeight) {
        int cellIndex = cell.x + cell.y * gridWidth;
        atomicCAS(&labels[cellIndex], Utility::AIR, Utility::FLUID); // Атомарное обновление
    }
}


int FluidSolver2D::labelGrid() {
    // first clear grid labels (mark everything air, but leave solids)
    for (int i = 0; i < gridWidth; i++) {
        for (int j = 0; j < gridHeight; j++) {
            if (labels[i+j*gridWidth] != Utility::SOLID) {
                labels[i+j*gridWidth] = Utility::AIR;
            }
        }
    }

    // mark any cell containing a particle FLUID
    for (int i = 0; i < particles->size(); i++) {
        // get cell containing the particle
        int2 cell = Utility::getGridCellIndex(particles->at(i).pos, dx);
        labels[cell.x + cell.y * gridWidth] = Utility::FLUID;
    }
    return 0;
}

int FluidSolver2D::labelGrid_gpu() {
    int* labels_for_device = NULL;
    cudaDeviceSynchronize();
    Utility::Particle2D* particles_for_device;
    cudaError_t err;
    err = cudaMalloc(&labels_for_device, sizeof(int)*w_x_h);
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc labels error: " << cudaGetErrorString(err) << err << std::endl;
        return -1;
    }
    cudaMemcpy(labels_for_device, labels.data(), sizeof(int)*w_x_h, cudaMemcpyHostToDevice);
    err = cudaMalloc(&particles_for_device, sizeof(Utility::Particle2D)*particles->size());
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc particles error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(labels_for_device);
        return -1;
    }
    int numParticles = static_cast<int>(particles->size());
    cudaMemcpy(particles_for_device, particles->data(), sizeof(Utility::Particle2D)*numParticles, cudaMemcpyHostToDevice);
    // 1. Очистка сетки
    dim3 blockDim_loc(16, 16); //16*16 = 256
    dim3 gridDim_loc((gridWidth + 15) / 16, (gridHeight + 15) / 16);
    clearLabelsKernel<<<gridDim_loc, blockDim_loc>>>(labels_for_device, gridWidth, gridHeight);

    // 2. Пометка fluid-ячеек
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    markFluidCellsKernel<<<gridSize, blockSize>>>(particles_for_device, numParticles,
                                                  dx, gridWidth, gridHeight, labels_for_device);
    cudaDeviceSynchronize();
    cudaMemcpy(labels.data(), labels_for_device, sizeof(int)*w_x_h, cudaMemcpyDeviceToHost);
    cudaFree(labels_for_device);
    cudaFree(particles_for_device);
    return 0;
}


/**
 * ########################## ЭТАП "ЧАСТИЦЫ -> СЕТКА"
 * ########################################
 * ########################################
 * */
__global__ void accumulateDenAndNum( Utility::Particle2D particle, float* uNum, float* uDen, float* vNum, float* vDen, int uSize, int vSize, int gridWidth, int gridHeight, float dx){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    // save check
    if(ind < uSize){
        int2 uCellInd = Utility::getGridIndicesU(ind, gridWidth);
        float kernel = Utility::bilinearHatKernel(particle.pos- Utility::getGridCellPosition_device(uCellInd.x, uCellInd.y, dx), dx, dx );
        atomicAdd(&uNum[ind], particle.vel.x * kernel);
        atomicAdd(&uDen[ind], kernel);
    }
    if(ind < vSize){
        int2 vCellInd = Utility::getGridIndicesV(ind, gridWidth);
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
    for (int j = 0; j < gridHeight; j++) {
    for (int i = 0; i < gridWidth + 1; i++) {
            uSaved[i+ j*(gridWidth+1)] = u[i+ j*(gridWidth+1)];
        }
    }

    // save v grid
    for (int j = 0; j < gridHeight + 1; j++) {
    for (int i = 0; i < gridWidth; i++) {
            vSaved[i+ j*gridWidth] = v[i+ j*gridWidth];
        }
    }
}


/**
 * ########################## ЭТАП "ВНЕШНИЕ СИЛЫ"
 * ########################################
 * ########################################
 * */
void FluidSolver2D::applyForces() {
    // traverse all grid cells and apply force to each velocity component
    // The new velocity is calculated using forward euler
    for (int j = 0; j < gridHeight + 1; j++) {
    for (int i = 0; i < gridWidth + 1; i++) {
            if (j < gridHeight) {
                // make sure we know the velocity
                if (u[i+ j*(gridWidth+1)] != VEL_UNKNOWN) {
                    // update u component
                    u[i+ j*(gridWidth+1)] += dt*GRAVITY.x;
                }
            }
            if (i < gridWidth) {
                if (v[i+ j*gridWidth] != VEL_UNKNOWN) {
                    // update v component
                    v[i+ j*gridWidth] += dt*GRAVITY.y;
                }
            }
        }
    }
}



/**
 * ########################## ЭТАП "ДАВЛЕНИЕ"
 * ########################################
 * ########################################
 * */
void FluidSolver2D::constructRHS(std::vector<float>& rhs){
    // calculate negative divergence
    //float scale = 1.0f / dx;
    float scale = (FLUID_DENSITY * dx) / dt;
    int counterFluidCells = 0; //TODO: move to a separate function or change logic!
    //std::vector<float> rhs2;
    for (int j = 0; j < gridHeight; j++) {
        for (int i = 0; i < gridWidth; i++) {
            if (isFluid(i, j)) {
                int newFluidInd = counterFluidCells++;
                fluidNumbers[i+ j*gridWidth] = newFluidInd; //TODO: move to a separate function or change logic!
                //std::cout << "(i,j) = " << j << ", " << i << " ; newFluidInd = " << newFluidInd << std::endl;
                //rhs[i*gridHeight+j] = -scale * (u[(i + 1)*gridHeight+j] - u[i*gridHeight+j] + v[i*gridHeight + (j + 1)] - v[i*gridHeight+j]);
                rhs[newFluidInd] =  -scale * (u[(i + 1)+j*(gridWidth+1)] - u[i+j*(gridWidth+1)] + v[i + (j + 1)*gridWidth] - v[i+j*gridWidth]);
                // if it's on boundary must update to consider solid velocity
                // TODO create actual solid velocity grids, for right now just 0
                if (labels[(i - 1)+j*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] -= scale * (u[i*gridHeight+j] - 0.0f); //m_usolid[i][j]
                    rhs[newFluidInd]-= scale * (u[i+j*(gridWidth+1)] - 0.0f);
                }
                if (labels[(i + 1)+j*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] += scale * (u[(i + 1)*gridHeight+j] - 0.0f); //m_usolid[i+1][j]
                    rhs[newFluidInd] +=  scale * (u[(i + 1)+j*(gridWidth+1)] - 0.0f);
                }
                if (labels[i+(j - 1)*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] -= scale * (v[i*gridHeight+j] - 0.0f); //m_vsolid[i][j]
                    rhs[newFluidInd]  -= scale * (v[i+j*gridWidth] - 0.0f);
                }
                if (labels[i+(j + 1)*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] += scale * (v[i*gridHeight+(j + 1)] - 0.0f); //m_vsolid[i][j+1]
                    rhs[newFluidInd] += scale * (v[i+(j + 1)*gridWidth] - 0.0f);
                }
            }
        }
    }
    fluidCellsAmount = counterFluidCells;
}

/*
Adiag - grid to store the diagonal of the matrix in.
Ax - grid to store the coefficients for pressure in the (i+1) cell for each grid cell with x index i
Ay - grid to store the coefficients for pressure in the (j+1) cell for each grid cell with y index j
*/
void FluidSolver2D::constructA(std::vector<float>& csr_values, std::vector<int>& csr_columns, std::vector<int>& csr_offsets) {

    // populate with coefficients for pressure unknowns
    //float scale = dt / (FLUID_DENSITY * dx * dx);

    int offset = 0;
    float scale = 1.0f;
    for (int j = 0; j < gridHeight; ++j) {
    for (int i = 0; i < gridWidth; ++i) {
            int newFluidInd = fluidNumbers[i+ j*gridWidth];
            if (newFluidInd != -1) {
                float diagVal = 0.0f;
                float rightVal = 0.0f;
                bool rvIsNZ = false;
                float botVal = 0.0f;
                bool bvIsNZ = false;
                // handle negative x neighbor
                if (labels[(i - 1)+j*gridWidth] == Utility::FLUID || labels[(i - 1)+j*gridWidth] == Utility::AIR) {
                    //Adiag[i*gridHeight+j] += scale;
                    diagVal += scale;
                }
                // handle positive x neighbor
                if (labels[(i + 1)+j*gridWidth] == Utility::FLUID) {
                    //Adiag[i*gridHeight+j] += scale;
                    //Ax[i*gridHeight+j] = -scale;
                    diagVal += scale;
                    botVal = -scale;
                    bvIsNZ = true;
                } else if (labels[(i + 1)+j*gridWidth] == Utility::AIR) {
                   // Adiag[i*gridHeight+j] += scale;
                    diagVal += scale;
                }
                // handle negative y neighbor
                if (labels[i+(j - 1)*gridWidth] == Utility::FLUID || labels[i +(j - 1)*gridWidth] == Utility::AIR) {
                    //Adiag[i*gridHeight+j] += scale;
                    diagVal += scale;
                }
                // handle positive y neighbor
                if (labels[i+(j + 1)*gridWidth] == Utility::FLUID) {
                    //Adiag[i*gridHeight+j] += scale;
                    //Ay[i*gridHeight+j] = -scale;
                    diagVal += scale;
                    rightVal = -scale;
                    rvIsNZ = true;
                } else if (labels[i+(j + 1)*gridWidth] == Utility::AIR) {
                    //Adiag[i*gridHeight+j] += scale;
                    diagVal += scale;
                }

                csr_offsets[newFluidInd] = offset;
                csr_values.push_back(diagVal);
                csr_columns.push_back(newFluidInd);
                ++offset;
                if(rvIsNZ){
                    csr_values.push_back(rightVal);
                    csr_columns.push_back(fluidNumbers[i +(j + 1)*gridWidth]);
                    ++offset;
                }
                if(bvIsNZ){
                    csr_values.push_back(botVal);
                    csr_columns.push_back(fluidNumbers[(i + 1)+j*gridWidth]);
                    ++offset;
                }

            }
        }
    }
    csr_offsets.back() = csr_values.size();

    //DEBUG COUT
//    for(int i = 0; i < csr_values.size(); ++i){
//        std::cout << csr_values[i] << ", ";
//    }
//    std::cout << "\n------\n";
//    for(int i = 0; i <csr_columns.size(); ++i){
//        std::cout << csr_columns[i] << ", ";
//    }
//    std::cout << "\n------\n";
//    for(int i = 0; i < csr_offsets.size(); ++i){
//        std::cout << csr_offsets[i] << ", ";
//    }
//    std::cout << "\n------\n";
}



#define CUDSS_EXAMPLE_FREE \
    do { \
        free(x_values_h);        \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d);    \
    } while(0);

//Строим A, но по-своему
// 1)создаём массив размера w_x_h, состоящий из новой нумерации
// по индексу i*gridHeight+j будем получать новую нумерацию для системы
// 2) проходимся по созданному массиву новой нумерации (i = 0; i < gridWidth, j = 0; j < gridHeight)
// 2.1) чекаем соседей сверху, слева, снизу, справа для подсчёта количества (коэф на диагонали в новой матрице)
// (i+1,j), (i,  j+1) по старой нумерации: берём, находим новую нумерацию и в эти новые индексы заиписываем для новой матрицы
// 2.2) создаём ещё параллельно вектор правой части в новой нумерации
// 3) после создания массивов для csr создаём cuSparce матрицу, указываем всвойства симметрии и тд
// 4) запихиваем в решатель, получаем результат
// 5) полученные давления в старую нумерацию и делаем pressureProjectiion (почитать, возможно, не потребуется возвращение нумерации)
int FluidSolver2D::pressureSolve() {
    p.assign(w_x_h, 0.0f);
    fluidNumbers = std::vector<int>(w_x_h, -1);
    std::vector<float> rhs(w_x_h, 0.0f);
    constructRHS(rhs);
    std::vector<float> csr_values;
    std::vector<int> csr_columns;
    std::vector<int> csr_offsets(fluidCellsAmount+1, fluidCellsAmount);
    constructA(csr_values, csr_columns, csr_offsets);

    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = fluidCellsAmount;
    //std::cout << "n = " << n<< std::endl;

    int nnz = csr_values.size();
    int nrhs = 1;

    float *x_values_h = NULL;
    x_values_h = (float*)malloc(nrhs * n * sizeof(float));
    if (!x_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    int *csr_offsets_d;
    int *csr_columns_d;
    float *csr_values_d;
    float *x_values_d, *b_values_d;

    cudaDeviceSynchronize();
    //cudaError err = cudaGetLastError();
    //std::cout << cudaGetErrorString(err) << std::endl;


    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(float )),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(float)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(float)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets.data(), (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns.data(), nnz * sizeof(int),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values.data(), nnz * sizeof(float),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, rhs.data(), nrhs * n * sizeof(float),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_32F,
                                             CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_32F,
                                             CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_32F, mtype, mview,
                                              base), status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                                      A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                                      A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(float),
                                   cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    /* Release the data allocated on the user side */
    for(int j = 0; j < gridHeight; ++j){
        for(int i = 0; i < gridWidth; ++i){
            int newFluidInd = fluidNumbers[i+ j*gridWidth];
            if(newFluidInd != -1){
                p[i + j*gridWidth] = x_values_h[newFluidInd];
            }
        }
    }

    CUDSS_EXAMPLE_FREE;

    return 0;

}

void FluidSolver2D::applyPressure() {
    float scale = dt / (FLUID_DENSITY * dx);
    for (int j = 0; j < gridHeight; ++j) {
    for (int i = 0; i < gridWidth; ++i) {
            // update u
            if (i - 1 >= 0) {
                if (labels[(i - 1) + j*gridWidth] == Utility::FLUID || labels[i +j*gridWidth] == Utility::FLUID) {
                    if (labels[(i - 1) + j*gridWidth] == Utility::SOLID ||  labels[i +j*gridWidth] == Utility::SOLID) {
                        // TODO add solid velocities
                        u[i + j*(gridWidth+1)] = 0.0f; // usolid[i][j]
                    } else {
                        u[i + j*(gridWidth+1)] -= scale * (p[i + j*gridWidth] - p[(i - 1)+j*gridWidth]);
                    }
                } else {
                    u[i + j*(gridWidth+1)] = VEL_UNKNOWN;
                }
            } else {

            }

            // update v
            if (j - 1 >= 0) {
                if (labels[i + (j - 1)*gridWidth] == Utility::FLUID || labels[i +j*gridWidth] == Utility::FLUID) {
                    if (labels[i + (j - 1)*gridWidth] == Utility::SOLID || labels[i +j*gridWidth] == Utility::SOLID) {
                        // TODO add solid velocities
                        v[i +j*gridWidth] = 0.0f; // vsolid[i][j]
                    }
                    else {
                        v[i +j*gridWidth] -= scale * (p[i +j*gridWidth] - p[i  + (j - 1)*gridWidth]);
                    }
                } else {
                    v[i +j*gridWidth] = VEL_UNKNOWN;
                }
            } else {
                // edge of grid, keep the same velocity
            }
        }
    }
}


/**
 * ########################## ЭТАП "СЕТКА -> ЧАСТИЦЫ"
 * ########################################
 * ########################################
 * */

/* !!!!cpu!!!
Interpolates the value in the given velocity grid at the given position using bilinear interpolation.
Returns velocity unkown if position is not on simulation grid.
Args:
uGrid - the u component grid to interpolate from
vGrid - the v component grid to interpolate from
pos - the position to interpolate at
*/
float2 FluidSolver2D::interpVel(std::vector<float>& uGrid, std::vector<float>& vGrid, float2 pos) {
    // get grid cell containing position
    int2 cell = Utility::getGridCellIndex(pos, dx);
    int i = cell.x;
    int j = cell.y;

    // make sure this is a valid index
    if (i >= 0 && i < gridWidth && j >= 0 && j < gridHeight) {
        // get positions of u and v component stored on each side of cell
        float2 cellLoc = Utility::getGridCellPosition(i, j, dx);
        float offset = dx / 2.0f;
        float x1 = cellLoc.x - offset;
        float x2 = cellLoc.x + offset;
        float y1 = cellLoc.y - offset;
        float y2 = cellLoc.y + offset;
        // get actual values at these positions
        float u1 = uGrid[i+ j*(gridWidth+1)];
        float u2 = uGrid[(i + 1)+j*(gridWidth+1)];
        float v1 = vGrid[i+ j*gridWidth];
        float v2 = vGrid[i+(j + 1)*gridWidth];

        // the interpolated values
        float u = ((x2 - pos.x) / (x2 - x1)) * u1 + ((pos.x - x1) / (x2 - x1)) * u2;
        float v = ((y2 - pos.y) / (y2 - y1)) * v1 + ((pos.y - y1) / (y2 - y1)) * v2;
        return make_float2(u, v);
    } else {
        return make_float2(VEL_UNKNOWN, VEL_UNKNOWN);
    }
}

void FluidSolver2D::gridToParticles(float alpha) {
    std::vector<float> duGrid((gridWidth+1)*gridHeight, 0.0f);
    std::vector<float> dvGrid((gridWidth)*(gridHeight+1), 0.0f);
    // calc u grid
    for (int j = 0; j < gridHeight; j++) {
    for (int i = 0; i < gridWidth + 1; i++) {

            duGrid[i + j*(gridWidth+1)] = u[i + j*(gridWidth+1)] - uSaved[i + j*(gridWidth+1)];
        }
    }
    // calc v grid
    for (int j = 0; j < gridHeight + 1; j++) {
    for (int i = 0; i < gridWidth; i++) {

            dvGrid[i + j*gridWidth] = v[i+ j*gridWidth] - vSaved[i+ j*gridWidth];
        }
    }

    // go through particles and interpolate each velocity component
    // the update is a PIC/FLIP mix weighted with alpha
    // alpha = 1.0 is entirely PIC, alpha = 0.0 is all FLIP
    for (int i = 0; i < particles->size(); i++) {
        Utility::Particle2D *curParticle = &(particles->at(i));
        float2 picInterp = interpVel(u, v, curParticle->pos);
        float2 flipInterp = interpVel(duGrid, dvGrid, curParticle->pos);
        // u_new = alpha * interp(u_gridNew, x_p) + (1 - alpha) * (u_pOld + interp(u_dGrid, x_p))
        //curParticle->vel = add(scale(picInterp, alpha), scale(add(curParticle->vel, flipInterp), 1.0f - alpha));
        curParticle->vel.x = picInterp.x * alpha + (curParticle->vel.x + flipInterp.x) * (1.0f - alpha);
        curParticle->vel.y = picInterp.y * alpha + (curParticle->vel.y + flipInterp.y) * (1.0f - alpha);
    }

}

////////GPU-версия

// Ядро для интерполяции скорости
__global__ void interpVelKernel(
        const float* uGrid,
        const float* vGrid,
        const float2* particles,
        float2* particleVelocities,
        int numParticles,
        float dx,
        int gridWidth,
        int gridHeight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float2 pos = particles[idx];
    float2 picInterp = make_float2(VEL_UNKNOWN, VEL_UNKNOWN);
    float2 flipInterp = make_float2(0.0f, 0.0f);

    // Билинейная интерполяция
    int2 cell = make_int2((int)(pos.x / dx), (int)(pos.y/dx));
    int i = cell.x;
    int j = cell.y;

    if (i >= 0 && i < gridWidth && j >= 0 && j < gridHeight) {
        float2 cellLoc = make_float2((i +0.5f)* dx, (j + 0.5f) * dx);
        float offset = dx / 2.0f;
        float x1 = cellLoc.x - offset;
        float x2 = cellLoc.x + offset;
        float y1 = cellLoc.y - offset;
        float y2 = cellLoc.y + offset;

        // Интерполяция u
        float u1 = uGrid[i + j * (gridWidth + 1)];
        float u2 = uGrid[(i + 1) + j * (gridWidth + 1)];
        float u = ((x2 - pos.x) / (x2 - x1)) * u1 + ((pos.x - x1) / (x2 - x1)) * u2;

        // Интерполяция v
        float v1 = vGrid[i + j * gridWidth];
        float v2 = vGrid[i + (j + 1) * gridWidth];
        float v = ((y2 - pos.y) / (y2 - y1)) * v1 + ((pos.y - y1) / (y2 - y1)) * v2;

        picInterp = make_float2(u, v);
    }

    // Сохраняем результат для частицы
    particleVelocities[idx] = picInterp;
}

//вычисление duGrid/dvGrid на gpu
//duGrid
__global__ void computeDeltaUGridKernel(
        const float* u,
        const float* uSaved,
        float* duGrid,
        int gridWidth,
        int gridHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < gridWidth + 1 && j < gridHeight) {
        int index = i + j * (gridWidth + 1);
        duGrid[index] = u[index] - uSaved[index];
    }
}
// Аналогично для dvGrid
__global__ void computeDeltaVGridKernel(
        const float* v,
        const float* vSaved,
        float* dvGrid,
        int gridWidth,
        int gridHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < gridWidth && j < gridHeight + 1) {
        int index = i + j * gridWidth;
        dvGrid[index] = v[index] - vSaved[index];
    }
}

__global__ void updateParticleVelocitiesKernel(
        float2* particleVelocities,
        const float2* duGridInterp,
        const float2* dvGridInterp,
        int numParticles,
        float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float2 picInterp = particleVelocities[idx];
    float2 flipInterp = duGridInterp[idx]; // Предполагаем, что duGridInterp и dvGridInterp уже вычислены

    // Обновление скорости по схеме PIC/FLIP
    particleVelocities[idx] = picInterp * alpha + (particleVelocities[idx] + flipInterp)*(1.0f-alpha);
}

//TODO:
//void FluidSolver2D::gridToParticles_gpu(float alpha) {
////    std::vector<float> duGrid((gridWidth+1)*gridHeight, 0.0f);
////    std::vector<float> dvGrid((gridWidth)*(gridHeight+1), 0.0f);
//
//    float* duGrid_device;
//    float* dvGrid_device;
//    cudaMalloc(&duGrid_device, (gridWidth + 1) * gridHeight * sizeof(float));
//    cudaMalloc(&dvGrid_device, (gridWidth)*(gridHeight+1) * sizeof(float));
//
//    float* u_device;
//    float* v_device;
//    cudaMalloc(&u_device, (gridWidth + 1) * gridHeight * sizeof(float));
//    cudaMalloc(&v_device, (gridWidth)*(gridHeight+1) * sizeof(float));
//    cudaMemcpy(u_device, u.data(), (gridWidth + 1) * gridHeight * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(v_device, v.data(), gridWidth * (gridHeight+1) * sizeof(float), cudaMemcpyHostToDevice);
//
//    float* uSaved_device;
//    float* vSaved_device;
//    cudaMalloc(&uSaved_device, (gridWidth + 1) * gridHeight * sizeof(float));
//    cudaMalloc(&vSaved_device, (gridWidth)*(gridHeight+1) * sizeof(float));
//    cudaMemcpy(u_device, uSaved.data(), (gridWidth + 1) * gridHeight * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(v_device, vSaved.data(), gridWidth * (gridHeight+1) * sizeof(float), cudaMemcpyHostToDevice);
//
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        printf("CUDA Error: %s\n", cudaGetErrorString(err));
//    }
//
//    // 1. Вычисление duGrid и dvGrid
//    dim3 blockDim(16, 16);
//    dim3 gridDimDU((gridWidth + 1 + 15) / 16, (gridHeight + 15) / 16);
//    computeDeltaUGridKernel<<<gridDimDU, blockDim>>>(u_device, uSaved_device, duGrid_device, gridWidth, gridHeight);
//
//    dim3 gridDimDV((gridWidth + 15) / 16, (gridHeight + 1 + 15) / 16);
//    computeDeltaVGridKernel<<<gridDimDV, blockDim>>>(v_device, vSaved_device, dvGrid_device, gridWidth, gridHeight);
//
//    // 2. Интерполяция duGrid и dvGrid для частиц (аналогично interpVel)
//    float2* d_duInterp, *d_dvInterp;
//    cudaMalloc(&d_duInterp, particles->size() * sizeof(float2));
//    cudaMalloc(&d_dvInterp, particles->size() * sizeof(float2));
//
//    int numParticles = static_cast<int>(particles->size());
//    int blockSize = 256;
//    int gridSize = (numParticles + blockSize - 1) / blockSize;
//
//    interpVelKernel<<<gridSize, blockSize>>>(
//            duGrid_device, dvGrid_device, d_particles, d_duInterp,
//            numParticles, dx, gridWidth, gridHeight, alpha
//    );
//
//    // 3. Обновление скоростей частиц
//    updateParticleVelocitiesKernel<<<gridSize, blockSize>>>(
//            d_particleVelocities, d_duInterp, d_dvInterp, numParticles, alpha
//    );
//
//    // Освобождение временной памяти
//    cudaFree(d_duInterp);
//    cudaFree(d_dvInterp);
//
//}

/**
 * ########################## ЭТАП "ПЕРЕДВИЖЕНИЕ ЧАСТИЦ"
 * ########################################
 * ########################################
 * */

/*
Advects a particle using Runge-Kutta 3 through the given velocity field.
Args:
particle - the particle to advect
initVel - the particles initial velocity in the current field, can leave UNKNOWN
dt - the time step
uGrid/vGrid - the velocity grids to advect through
*/
void FluidSolver2D::RK3(Utility::Particle2D *particle, float2 initVel, float dt, std::vector<float>& uGrid, std::vector<float>& vGrid) {
    if (initVel.x == VEL_UNKNOWN && initVel.y == VEL_UNKNOWN) {
        initVel = interpVel(uGrid, vGrid, particle->pos);
    }

    float2 k1 = initVel;
    float2 k1_cpy{k1.x * 0.5f*dt + particle->pos.x, k1.y * 0.5f*dt+ particle->pos.y};
    float2 k2 = interpVel(uGrid, vGrid, k1_cpy);
    float2 k2_cpy{k2.x * 0.75f*dt + particle->pos.x, k2.y * 0.75f*dt+ particle->pos.y};
    float2 k3 = interpVel(uGrid, vGrid, k2_cpy);
    k1 = make_float2(k1.x *  (2.0f / 9.0f)*dt, k1.y * (2.0f / 9.0f)*dt);
    k2 =  make_float2(k2.x *  (3.0f / 9.0f)*dt, k2.y * (3.0f / 9.0f)*dt);
    k3 = make_float2(k3.x *  (4.0f / 9.0f)*dt, k3.y * (4.0f / 9.0f)*dt);

    particle->pos = make_float2(particle->pos.x + k1.x + k2.x + k3.x, particle->pos.y + k1.y + k2.y + k3.y);
}

std::vector<int> FluidSolver2D::checkNeighbors(std::vector<int> grid, int2 dim, int2 index, int neighbors[][2], int numNeighbors, int value) {
    std::vector<int> neighborsTrue;
    for (int i = 0; i < numNeighbors; i++) {
        int offsetX = neighbors[i][0];
        int offsetY = neighbors[i][1];
        int neighborX = index.x + offsetX;
        int neighborY = index.y + offsetY;

        // make sure valid indices
        if ((neighborX >= 0 && neighborX < dim.x) && (neighborY >= 0 && neighborY < dim.y)) {
            if (grid[neighborX  + neighborY*gridWidth] == value) {
                neighborsTrue.push_back(i);
            }
        }
    }

    return neighborsTrue;
}

bool FluidSolver2D::isCellValid(int x, int y){
    return x >= 0 && x < gridWidth && y >= 0 && y < gridHeight;
};

bool FluidSolver2D::projectParticle(Utility::Particle2D* particle, float max_h){
    // project back into fluid
    // find neighbors that are fluid
    // define neighbors
    const int neighbors[8][2] = {{-1,1}, {-1,0}, {-1,-1}, {0,1},
                                 {0,-1}, {1,1}, {1,0}, {1,-1}};

    int2 cell = Utility::getGridCellIndex(particle->pos, dx);
    std::vector<float2> valid_positions;

    // Собираем все допустимые позиции
    for (const auto& n : neighbors) {
        int x = cell.x + n[0];
        int y = cell.y + n[1];
        if (isCellValid(x, y) && labels[x + y*gridWidth] == Utility::SOLID) {
            valid_positions.push_back(Utility::getGridCellPosition(x, y, dx));
        }
    }

    if (valid_positions.empty()){
        for (const auto& n : neighbors) {
            int x = cell.x + n[0];
            int y = cell.y + n[1];
            if (isCellValid(x, y) && labels[x + y*gridWidth] == Utility::AIR) {
                valid_positions.push_back(Utility::getGridCellPosition(x, y, dx));
            }
        }
        if (valid_positions.empty()) return false;
    }

    // Выбираем позицию с минимальным расстоянием
    float2 new_pos = particle->pos;
    float min_dist = std::numeric_limits<float>::max();
    for (const auto& pos : valid_positions) {
        float dist = std::hypot(pos.x - particle->pos.x, pos.y - particle->pos.y);
        if (dist < min_dist) {
            min_dist = dist;
            new_pos = pos;
        }
    }

    // Смещаем частицу на smoothCoef от расстояния для плавности
    const float smoothCoef = 1.0f;
    particle->pos.x += smoothCoef * (new_pos.x - particle->pos.x);
    particle->pos.y += smoothCoef * (new_pos.y - particle->pos.y);

    return true;
}

//C - the maximum number of grid cells a particle should move when advected. This helps define substep sizes.
void FluidSolver2D::advectParticles(float C) {
    for (int i = 0; i < particles->size(); i++) {
        Utility::Particle2D *curParticle = &(particles->at(i));
        float subTime = 0;
        bool finished = false;

        while (!finished) {
            float2 curVel = interpVel(u, v, curParticle->pos);

            // calc max substep size
            float dT = (C * dx) / (std::sqrt(curVel.x*curVel.x + curVel.y*curVel.y) + FLT_MIN);
            // update substep time so we don't go past normal time step
            if (subTime + dT >= dt) {
                dT = dt - subTime;
                finished = true;
            } else if (subTime + 2 * dT >= dt) {
                dT = 0.5f * (dt - subTime);
            }

            //RK3(curParticle, curVel, dT, u, v);
            curParticle->pos.x += curVel.x * dT;
            curParticle->pos.y += curVel.y * dT;
            subTime += dT;

            if (curParticle->pos.x < 0 || curParticle->pos.y < 0 || std::isnan(curParticle->pos.x) || std::isnan(curParticle->pos.y)) {
                // there's been an error in RK3, just skip it

                //std::cout << "RK3 error...skipping particle" << std::endl;
                break;
            }

            int2 cell = Utility::getGridCellIndex(curParticle->pos, dx);
            int j = cell.x;
            int k = cell.y;
            if (labels[j+ k*gridWidth] == Utility::SOLID) {
                //std::cout << "Advected into SOLID, projecting back!\n";
                if (!projectParticle(curParticle, dx / 4.0f)) {
                    //std::cout << "RK3 error...skipping particle" << std::endl;
                    break;
                }
            }

        }
    }
}



//delta - the amount to project stray particles away from the wall.
void FluidSolver2D::cleanUpParticles(float delta) {
    int numDeleted = 0;
    for (auto it = particles->begin(); it != particles->end();) {
        int2 cell = Utility::getGridCellIndex(it->pos, dx);
        int x = cell.x;
        int y = cell.y;

        // Условия удаления частицы
        if (x < 0 || y < 0 || x >= gridWidth || y >= gridHeight ||
            std::isnan(it->pos.x) || std::isnan(it->pos.y)) {
            it = particles->erase(it);
            numDeleted++;
            continue;
        }

        // Проверка на SOLID и проекция
        if (labels[x + y * gridWidth] == Utility::SOLID) {
            bool success = projectParticle(&(*it), delta); // Правильная передача указателя на частицу

            if (!success) {
                it = particles->erase(it);
                numDeleted++;
            } else {
                ++it; // Инкремент только если частица не удалена
            }
        } else {
            ++it; // Переход к следующей частице
        }
    }

    // Обновление количества блоков для частиц
    blocksForParticles = (particles->size() + threadsPerBlock - 1) / threadsPerBlock;
    //std::cout << "Removed " << numDeleted << " particles. Total: " << particles->size() << "\n";
}


void FluidSolver2D::extrapolateGridFluidData(std::vector<float>& grid, int x, int y, int depth){
    // marker array
    std::vector<int> d(x*y, 0);

    // 0 для известных величин, int_max для неизвестных
    for(int j = 0; j < y; ++j){
        for(int i = 0; i < x; ++i){
            if(grid[i + j * x] != VEL_UNKNOWN){
                d[i + j * x] = 0;
            }else{
                d[i + j * x] = INT_MAX;
            }
        }
    }

    //определяем всевозможных 2д-соседей
    int numNeighbours = 8;
    int neighbours[8][2] = {
            {-1, 1}, // top left
            {-1, 0}, // middle left
            {-1, -1}, // bottom left
            {0, 1}, // top middle
            {0, -1}, // bottom middle
            {1, 1}, // top right
            {1, 0}, // middle right
            {1, -1} // bottom right
    };

    // инициализируем первый фронт волны
    std::vector<int2> W;
    int2 dim{ x, y };
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            // текущее значение неизвестно
            if (d[i + j*x] != 0) {
                int2 ind{ i, j };
                if (!checkNeighbors(d, dim, ind, neighbours, numNeighbours, 0).empty()) {
                    // соседа знаем
                    d[i + j*x] = 1;
                    W.push_back(make_int2(i, j));
                }
            }
        }
    }

    //все фронты, по которым хотим пройтись
    std::vector<std::vector<int2>> wavefronts;
    wavefronts.push_back(W);
    int curWave = 0;
    while(curWave < depth){
        //получаем текущий фронт
        std::vector<int2> curW = wavefronts.at(curWave);
        //определяем следующий
        std::vector<int2> nextW;
        //проходимся по текущему фронту и экстраполируем значения
        for(int i = 0; i < curW.size(); ++i){
            int2 ind = curW.at(i);
            //среднее по соседям
            float avg = 0.0f;
            int numUsed = 0;
            for(int j = 0; j < numNeighbours; ++j){
                int offsetX = neighbours[j][0];
                int offsetY = neighbours[j][1];
                int neighborX = ind.x + offsetX;
                int neighborY = ind.y + offsetY;

                //проверяем индексы на выход за пределы
                if ((neighborX >= 0 && neighborX < dim.x) && (neighborY >= 0 && neighborY < dim.y)) {
                    // добавляем только в том случае, если маркер соседа меньше, чем текущий
                    if (d[neighborX + neighborY * x] < d[ind.x + ind.y*x]) {
                        avg += grid[neighborX + neighborY*x];
                        numUsed++;
                    } else if (d[neighborX + neighborY*x] == INT_MAX) {
                        d[neighborX + neighborY*x] = d[ind.x + ind.y*x] + 1;
                        nextW.push_back(make_int2(neighborX, neighborY));
                    }
                }
            }
            avg /= numUsed;
            //задаём текущее значение как среднее по соседям
            grid[ind.x + ind.y * x] = avg;
        }

        //следующий фронт закидываем ко всем предыдущим
        wavefronts.push_back(nextW);
        curWave++;
    }

}

/**
 * ########################## ЭТАП "ОБЩИЙ ЦИКЛ ДЛЯ ОТРИСОВКИ КАДРА"
 * ########################################
 * ########################################
 * */
void FluidSolver2D::frameStep(){
    labelGrid_gpu();

    //particles velocities to grid
    particlesToGrid();

    // экстраполяция в пределах +одна ячейка (для аккуратной очистки дивергенции)
    extrapolateGridFluidData(u, gridWidth + 1, gridHeight, 2);
    extrapolateGridFluidData(v, gridWidth, gridHeight + 1, 2);

    //saving a copy of the current grid velocities (for FLIP)
    saveVelocities();

    //applying body forces on grid (e.g. gravity force)
    applyForces();

    pressureSolve();
    applyPressure();

    //grid velocities to particles
    gridToParticles(PIC_WEIGHT);

    //advection of particles
    extrapolateGridFluidData(u, gridWidth + 1, gridHeight, gridWidth);
    extrapolateGridFluidData(v, gridWidth, gridHeight + 1, gridHeight);
    advectParticles(ADVECT_MAX);

    //boundary penetration detection (if so --- move back inside)
    cleanUpParticles(dx/2.0f);
}

void FluidSolver2D::run(int max_steps) {
    for(int i = 0; i < max_steps; ++i){
        frameStep();
        if(i%10 == 0){
            Utility::saveParticlesToPLY(*particles, "InputData/particles_" + std::to_string(i) + ".ply");
            std::cout << "frame = " << i/10 << "; numParticles = " << particles->size()<<std::endl;
        }

    }
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
            if (labels[(i - 1) + j*gridWidth] == Utility::FLUID) {
                isFluid = true;
            }
        }
        else if (j == gridHeight) {
            if (labels[i + (j - 1)*gridWidth] == Utility::FLUID) {
                isFluid = true;
            }
        }
    }
    else if (labels[i + j*gridWidth] == Utility::FLUID) {
        isFluid = true;
    }

    return isFluid;
}


