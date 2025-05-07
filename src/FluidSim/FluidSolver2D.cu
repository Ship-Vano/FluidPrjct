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

__global__ void labelCellWrap(int* labels, Utility::Particle2D* particles, float dx, int gridWidth){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    labelCellClean(ind, labels, particles, dx,gridWidth);
    labelCellFluid(ind, labels, particles, dx,gridWidth);
}

__device__ void labelCellFluid(int ind, int* labels, Utility::Particle2D* particles, float dx, int gridWidth){
    int cellInd = Utility::getGridCellIndex_device(particles[ind].pos, dx, gridWidth);
    labels[cellInd] = Utility::FLUID;
}

__device__ void labelCellClean(int ind, int* labels, Utility::Particle2D* particles, float dx, int gridWidth){
    int cellInd = Utility::getGridCellIndex_device(particles[ind].pos, dx, gridWidth);
    if(labels[cellInd] != Utility::SOLID){
        labels[cellInd] = Utility::AIR;
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

//int FluidSolver2D::labelGrid() {
//    int particlesAmount = particles->size();
//    int blockForParticles = (particlesAmount + threadsPerBlock - 1) / threadsPerBlock;
//    int* labels_for_device = NULL;
//    cudaDeviceSynchronize();
//    Utility::Particle2D* particles_for_device;
//    cudaError_t err;
//    std::cout << w_x_h << std::endl;
//    err = cudaMalloc(&labels_for_device, sizeof(int)*w_x_h);
//    if(err != cudaSuccess) {
//        std::cerr << "cudaMalloc labels error: " << cudaGetErrorString(err) << err << std::endl;
//        return -1;
//    }
//    cudaMemcpy(labels_for_device, labels.data(), sizeof(int)*w_x_h, cudaMemcpyHostToDevice);
//    err = cudaMalloc(&particles_for_device, sizeof(Utility::Particle2D)*particles->size());
//    if(err != cudaSuccess) {
//        std::cerr << "cudaMalloc particles error: " << cudaGetErrorString(err) << std::endl;
//        cudaFree(labels_for_device);
//        return -1;
//    }
//    cudaMemcpy(particles_for_device, particles->data(), sizeof(Utility::Particle2D)*particlesAmount, cudaMemcpyHostToDevice);
//    labelCellWrap<<<blockForParticles, threadsPerBlock>>>(labels_for_device, particles->data(), dx, gridWidth);
//    cudaDeviceSynchronize();
//    cudaMemcpy(labels.data(), labels_for_device, sizeof(int)*w_x_h, cudaMemcpyDeviceToHost);
//    cudaFree(labels_for_device);
//    cudaFree(particles_for_device);
////    //DEBUG COUT
////    for(int j =0; j < gridHeight; ++j){
////      for(int i = 0; i < gridWidth; ++i){
////
////            int idx = i  + j*gridWidth;
////            switch (labels[idx]) {
////                case Utility::SOLID: std::cout<<"S"; break;
////                case Utility::FLUID: std::cout << "F"; break;
////                case Utility::AIR: std::cout << "A" ;   break;
////                default:
////                    throw std::runtime_error("Unknown cell val");
////            }
////        }
////        std::cout<<std::endl;
////    }
////    std::cout<<"--------------\n\n"<<std::endl;
//    return 0;
//}



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
            uSaved[i+ j*gridWidth] = u[i+ j*gridWidth];
        }
    }

    // save v grid
    for (int j = 0; j < gridHeight + 1; j++) {
    for (int i = 0; i < gridWidth; i++) {
            vSaved[i+ j*gridWidth] = v[i+ j*gridWidth];
        }
    }
}

void FluidSolver2D::applyForces() {
    // traverse all grid cells and apply force to each velocity component
    // The new velocity is calculated using forward euler
    for (int j = 0; j < gridHeight + 1; j++) {
    for (int i = 0; i < gridWidth + 1; i++) {
            if (j < gridHeight) {
                // make sure we know the velocity
                if (u[i+ j*gridWidth] != VEL_UNKNOWN) {
                    // update u component
                    u[i+ j*gridWidth] += dt*GRAVITY.x;
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
                rhs[newFluidInd] =  -scale * (u[(i + 1)+j*gridWidth] - u[i+j*gridWidth] + v[i + (j + 1)*gridWidth] - v[i+j*gridWidth]);
                // if it's on boundary must update to consider solid velocity
                // TODO create actual solid velocity grids, for right now just 0
                if (labels[(i - 1)+j*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] -= scale * (u[i*gridHeight+j] - 0.0f); //m_usolid[i][j]
                    rhs[newFluidInd]-= scale * (u[i+j*gridWidth] - 0.0f);
                }
                if (labels[(i + 1)+j*gridWidth] == Utility::SOLID) {
                    //rhs[i*gridHeight+j] += scale * (u[(i + 1)*gridHeight+j] - 0.0f); //m_usolid[i+1][j]
                    rhs[newFluidInd] +=  scale * (u[(i + 1)+j*gridWidth] - 0.0f);
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
Constructs the A matrix for the system to solve for pressure. This a sparse coefficient matrix
for the pressure terms, stored in 3 separate grids. If index i, j, k is not a fluid cell, then
it is 0.0 in all 3 grids that store the matrix.
Args:
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

//    int passed = 1;
//    for (int i = 0; i < n; i++) {
//        printf("x[%d] = %1.4f r[%d] = %1.4f\n;", i, x_values_h[i], i, rhs[i]);
//    }

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
    //std::cout << "pressure solve end" << std::endl;
//    if(status !=CUDSS_STATUS_SUCCESS ){
//        printf("PRESSURE SOLVE FAILED\n");
//    }
//    if (status == CUDSS_STATUS_SUCCESS && passed)
//        printf("PRESSURE SOLVE PASSED\n");
//    else
//        printf("PRESSURE SOLVE FAILED\n");

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
                        u[i + j*gridWidth] = 0.0f; // usolid[i][j]
                    } else {
                        u[i + j*gridWidth] -= scale * (p[i + j*gridWidth] - p[(i - 1)+j*gridWidth]);
                    }
                } else {
                    u[i + j*gridWidth] = VEL_UNKNOWN;
                }
            } else {
                // edge of grid, keep the same velocity
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

/*
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
        float u1 = uGrid[i+ j*gridWidth];
        float u2 = uGrid[(i + 1)+j*gridWidth];
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

            duGrid[i + j*gridWidth] = u[i + j*gridWidth] - uSaved[i + j*gridWidth];
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


bool FluidSolver2D::projectParticle(Utility::Particle2D* particle, float dx){
    // project back into fluid
    // find neighbors that are fluid
    // define neighbors
    int numNeighbors = 8;
    int neighbors[8][2] = {
            { -1, 1 }, // top left
            { -1, 0 }, // middle left
            { -1, -1 }, // bottom left
            { 0, 1 }, // top middle
            { 0, -1 }, // bottom middle
            { 1, 1 }, // top right
            { 1, 0 }, // middle right
            { 1, -1 } // bottom right
    };
    int2 dim{ gridWidth, gridHeight };
    int2 cell = Utility::getGridCellIndex(particle->pos, dx);
    int2 index{ cell.x, cell.y };
    // get neighbors that are fluid
    std::vector<int> neighborInd = checkNeighbors(labels, dim, index, neighbors, numNeighbors, Utility::FLUID);
    if (neighborInd.size() == 0) {
        // try with air
        neighborInd = checkNeighbors(labels, dim, index, neighbors, numNeighbors, Utility::AIR);
    }
    // find closest to particle
    int closestInd = -1;
    float closestDist = std::numeric_limits<float>::max();
    float2 closestVec{0.0f, 0.0f};
    for (int j = 0; j < neighborInd.size(); j++) {
        // get vec from particle to neighbor ind
        int ind[2] = { cell.x + neighbors[neighborInd.at(j)][0], cell.y + neighbors[neighborInd.at(j)][1] };
        float2 cellPos = Utility::getGridCellPosition(ind[0], ind[1], dx);
        float2 distVec = make_float2(cellPos.x - particle->pos.x, cellPos.y - particle->pos.y);
        float dist = std::sqrt(distVec.x * distVec.x + distVec.y * distVec.y);
        if (dist < closestDist) {
            closestDist = dist;
            closestInd = neighborInd.at(j);
            closestVec = distVec;
        }
    }

    if (closestInd == -1) {
        return false;
    }
    else {
        // project different ways based on where closest neighbor is
        // also make sure to only project the amount given
        float2 projectVec{0.0f, 0.0f};
        if (closestInd == 1) { // middle left
            projectVec.x = closestVec.x + (-dx + (dx / 2.0f));
        }
        else if (closestInd == 3) { // top middle
            projectVec.y = closestVec.y + (dx - (dx / 2.0f));
        }
        else if (closestInd == 4) { // bottom middle
            projectVec.y = closestVec.y + (-dx + (dx / 2.0f));
        }
        else if (closestInd == 6) { // middle right
            projectVec.x = closestVec.x + (dx - (dx / 2.0f));
        }
        else if (closestInd == 5) { // top right
            projectVec.x = closestVec.x + (dx - (dx / 2.0f));
            projectVec.y = closestVec.y + (dx - (dx / 2.0f));
        }
        else if (closestInd == 0) { // top left
            projectVec.x = closestVec.x + (-dx + (dx / 2.0f));
            projectVec.y = closestVec.y + (dx - (dx / 2.0f));
        }
        else if (closestInd == 2) { // bottom left
            projectVec.x = closestVec.x + (-dx + (dx / 2.0f));
            projectVec.y = closestVec.y + (-dx + (dx / 2.0f));
        }
        else if (closestInd == 7) { // bottom right
            projectVec.x = closestVec.x + (dx - (dx / 2.0f));
            projectVec.y = closestVec.y + (-dx + (dx / 2.0f));
        }

        particle->pos = make_float2(particle->pos.x + projectVec.x, particle->pos.y + projectVec.y);

        return true;
    }
}

//C - the maximum number of grid cells a particle should move when advected. This helps define substep sizes.
void FluidSolver2D::advectParticles(int C) {
    for (int i = 0; i < particles->size(); i++) {
        Utility::Particle2D *curParticle = &(particles->at(i));
        float subTime = 0;
        bool finished = false;
        //float dT = m_dt / 4.999f;
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

            RK3(curParticle, curVel, dT, u, v);
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
    int i = 0;
    bool finished = false;
    int numDeleted = 0;
    while(!finished && particles->size() > 0) {
        int2 cell = Utility::getGridCellIndex(particles->at(i).pos, dx);
        int ind[2] = { cell.x, cell.y };
        // if either of cells are negative or greater than sim dimensions it has left sim area
        if (ind[0] < 0 || ind[1] < 0 || ind[0] >= gridWidth || ind[1] >= gridHeight || std::isnan(particles->at(i).pos.x) || std::isnan(particles->at(i).pos.y)) {
            particles->erase(particles->begin() + i);
            numDeleted++;
            if (i >= particles->size()) {
                finished = true;
            }
        } else if (labels[ind[0] + ind[1]*gridWidth] == Utility::SOLID) {
            // project back into fluid
            bool success = projectParticle(&(particles->at(i)), dx);
            if (!success) {
                // no near fluid, just delete
                particles->erase(particles->begin() + i);
                numDeleted++;
                if (i >= particles->size()) {
                    finished = true;
                }
            }
        } else {
            i++;
            if (i >= particles->size()) {
                finished = true;
            }
        }
    }

    //std::cout << "Removed " << numDeleted << " particles from sim.\n";
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


void FluidSolver2D::run(int max_steps) {
    for(int i = 0; i < max_steps; ++i){
        frameStep();
        if(i%10 == 0){
            Utility::saveParticlesToPLY(*particles, "InputData/particles_" + std::to_string(i) + ".ply");
            std::cout << "frame = " << i/10 << std::endl;
        }

    }
}