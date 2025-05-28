#include "FluidSolver3D.cuh"

//basic funcs################1
FluidSolver3D::FluidSolver3D(int width, int height, int depth, float dx_, float dt_){
    gridWidth = width;
    gridHeight = height;
    w_x_h_x_d = gridWidth * gridHeight;
    dx = dx_;
    dy = dx;
    dz = dx;
    dt = dt_;
    std::cout << "dx=" << dx<<"\n";
    h_particles = thrust::host_vector<Utility::Particle3D>();
    d_particles = thrust::device_vector<Utility::Particle3D>();
}

FluidSolver3D::~FluidSolver3D(){
}


__host__ void FluidSolver3D::init(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + fileName);
    }

    // Чтение размеров
    file >> gridWidth >> gridHeight >> gridDepth;

    // Инициализация сеток
    labels.resize(gridWidth, gridHeight, gridDepth);
    p.resize(gridWidth, gridHeight, gridDepth);
    u.resize(gridWidth+1, gridHeight, gridDepth);
    v.resize(gridWidth, gridHeight+1, gridDepth);
    w.resize(gridWidth, gridHeight, gridDepth+1);

    uSaved.resize(gridWidth+1, gridHeight, gridDepth);
    vSaved.resize(gridWidth, gridHeight+1, gridDepth);
    wSaved.resize(gridWidth, gridHeight, gridDepth+1);

    thrust::fill(u.device_data.begin(), u.device_data.end(), 0.0f);
    thrust::fill(v.device_data.begin(), v.device_data.end(), 0.0f);
    thrust::fill(w.device_data.begin(), w.device_data.end(), 0.0f);

    thrust::fill(uSaved.device_data.begin(), uSaved.device_data.end(), 0.0f);
    thrust::fill(vSaved.device_data.begin(), vSaved.device_data.end(), 0.0f);
    thrust::fill(wSaved.device_data.begin(), wSaved.device_data.end(), 0.0f);

    // Конфигурация блоков и потоков
    blockSize3D = dim3(8, 8, 8); // 512 потоков в блоке
    threadsPerBlock1D = 256;      // Для обработки частиц

    // Расчет размеров гридов для различных сеток
    gridSizeCells = dim3(
            (gridWidth + blockSize3D.x - 1) / blockSize3D.x,
            (gridHeight + blockSize3D.y - 1) / blockSize3D.y,
            (gridDepth + blockSize3D.z - 1) / blockSize3D.z
    );

    gridSizeU = dim3(
            (gridWidth + 1 + blockSize3D.x - 1) / blockSize3D.x,
            (gridHeight + blockSize3D.y - 1) / blockSize3D.y,
            (gridDepth + blockSize3D.z - 1) / blockSize3D.z
    );

    gridSizeV = dim3(
            (gridWidth + blockSize3D.x - 1) / blockSize3D.x,
            (gridHeight + 1 + blockSize3D.y - 1) / blockSize3D.y,
            (gridDepth + blockSize3D.z - 1) / blockSize3D.z
    );

    gridSizeW = dim3(
            (gridWidth + blockSize3D.x - 1) / blockSize3D.x,
            (gridHeight + blockSize3D.y - 1) / blockSize3D.y,
            (gridDepth + 1 + blockSize3D.z - 1) / blockSize3D.z
    );

    // Чтение меток
    for (int k = gridDepth-1; k > -1; --k) {
        for (int j = gridHeight-1; j > -1; --j) {
            for (int i = gridWidth-1; i > -1; --i) {
                char cellType;
                file >> cellType;

                switch (cellType) {
                    case 'S': labels(i, j, k) = Utility::SOLID; break;
                    case 'F': labels(i, j, k) = Utility::FLUID; break;
                    case 'A': labels(i, j, k) = Utility::AIR;   break;
                }
            }
        }
    }
    labels.copy_to_device();

    // Инициализация частиц
    seedParticles(PARTICLES_PER_CELL);
    std::cout <<"Number of particles is" << h_particles.size() << std::endl;
    Utility::save3dParticlesToPLY(h_particles, "InputData/particles_0.ply");
}

__host__ void FluidSolver3D::seedParticles(int particlesPerCell){
    // Инициализация генератора (один раз вне функции!)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> subCellDist(0, 7);
    static std::uniform_real_distribution<> jitterDist(-0.24f, 0.24f);

    // Сначала подсчитываем общее количество частиц
    h_particles.clear();
    size_t totalParticles = 0;
    for (int k = 0; k < gridDepth; ++k) {
        for (int j = 0; j < gridHeight; ++j) {
            for (int i = 0; i < gridWidth; ++i) {
                if (labels(i,j,k) == Utility::FLUID) {
                    totalParticles += particlesPerCell;
                }
            }
        }
    }

    // Резервируем память заранее
    h_particles.reserve(totalParticles);
    d_particles.reserve(totalParticles);

    // Проходим по всем ячейкам с жидкостью
    for(int k = 0; k < gridDepth; ++k)
    for (int j = 0; j < gridHeight; ++j) {
        for (int i = 0; i < gridWidth; ++i) {
            if (labels(i,j,k) == Utility::FLUID) {
                float3 cellCenter = Utility::getGridCellPosition(i, j, k, dx);
                // 8 субрегионов (октантов) в 3D ячейке
                float3 subCenters[8];
                const float offset = 0.25f * dx;

                // Генерируем центры субрегионов
                for (int octant = 0; octant < 8; ++octant) {
                    subCenters[octant] = {
                            cellCenter.x + (octant & 1 ? offset : -offset),
                            cellCenter.y + (octant & 2 ? offset : -offset),
                            cellCenter.z + (octant & 4 ? offset : -offset)
                    };
                }

                // Равномерное распределение частиц по субрегионам
                for (int p = 0; p < particlesPerCell; ++p) {
                    // Случайный выбор субрегиона для каждой частицы
                    int subCellIdx = subCellDist(gen);

                    // Случайное смещение
                    float jitterX = jitterDist(gen) * dx;
                    float jitterY = jitterDist(gen) * dx;
                    float jitterZ = jitterDist(gen) * dx;

                    // Позиция частицы
                    float3 pos = {
                            subCenters[subCellIdx].x + jitterX,
                            subCenters[subCellIdx].y + jitterY,
                            subCenters[subCellIdx].z + jitterZ
                    };

                    // Ограничение позиции в пределах ячейки
                    pos.x = std::clamp(pos.x, i * dx, (i + 1) * dx);
                    pos.y = std::clamp(pos.y, j * dx, (j + 1) * dx);
                    pos.z = std::clamp(pos.z, k * dx, (k + 1) * dx);

                    // Создаем частицу
                    Utility::Particle3D particle(pos, make_float3(0.0f, 0.0f, 0.0f));

                    // Добавляем в список
                    h_particles.push_back(particle);
                }
            }
        }
    }
    thrust::copy(h_particles.begin(), h_particles.end(), d_particles.begin());
    blocksForParticles = (h_particles.size() + threadsPerBlock- 1) / threadsPerBlock;
}



__host__ int FluidSolver3D::labelGrid() {
    // 1. Очистка сетки с помощью Thrust
    thrust::transform(thrust::device,
                      labels.device_ptr(),
                      labels.device_ptr() + labels.size(),
                      labels.device_ptr(),
                      ClearLabelsFunctor());

    // 2. Пометка ячеек с частицами
    const int numParticles = d_particles.size();
    if (numParticles > 0) {
        MarkFluidCellsFunctor functor(
                thrust::raw_pointer_cast(d_particles.data()),
                dx,
                labels.width(),
                labels.height(),
                labels.depth(),
                labels.device_ptr()
        );

        thrust::for_each(thrust::device,
                         thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(numParticles),
                         functor);
    }

    // 3. Проверка ошибок
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "labelGrid3D_gpu error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}

void FluidSolver3D::saveVelocities() {
    thrust::copy(thrust::device,
                 u.device_data.begin(), u.device_data.end(),
                 uSaved.device_data.begin());

    thrust::copy(thrust::device,
                 v.device_data.begin(), v.device_data.end(),
                 vSaved.device_data.begin());

    thrust::copy(thrust::device,
                 w.device_data.begin(), w.device_data.end(),
                 wSaved.device_data.begin());
}

// Структура для хранения временных данных (числители и знаменатели в p2g)
struct VelocityAccumulators {
    thrust::device_vector<float> uNum, uDen;
    thrust::device_vector<float> vNum, vDen;
    thrust::device_vector<float> wNum, wDen;

    VelocityAccumulators(int uSize, int vSize, int wSize) :
            uNum(uSize), uDen(uSize),
            vNum(vSize), vDen(vSize),
            wNum(wSize), wDen(wSize)
    {
        thrust::fill(uNum.begin(), uNum.end(), 0.0f);
        thrust::fill(uDen.begin(), uDen.end(), 0.0f);
        thrust::fill(vNum.begin(), vNum.end(), 0.0f);
        thrust::fill(vDen.begin(), vDen.end(), 0.0f);
        thrust::fill(wNum.begin(), wNum.end(), 0.0f);
        thrust::fill(wDen.begin(), wDen.end(), 0.0f);
    }
};

// Функтор для накопления скоростей
struct AccumulateVelocities {
    float dx;
    int gridWidth, gridHeight, gridDepth;
    int uStride, vStride, wStride;
    const Utility::Particle3D* particles;

    // Указатели на временные данные
    float* uNum, *uDen;
    float* vNum, *vDen;
    float* wNum, *wDen;

    __device__
    void operator()(int pidx) const {
        Utility::Particle3D p = particles[pidx];
        float3 pos = p.pos;
        float3 vel = p.vel;

        // Для u-компоненты (грани по X)
        for (int j_offset = 0; j_offset < 2; j_offset++) {
            for (int k_offset = 0; k_offset < 2; k_offset++) {
                int i = static_cast<int>(pos.x / dx);
                int j = static_cast<int>((pos.y - 0.5f * dx) / dx) + j_offset;
                int k = static_cast<int>((pos.z - 0.5f * dx) / dx) + k_offset;

                if (i >= 0 && i <= gridWidth &&
                    j >= 0 && j < gridHeight &&
                    k >= 0 && k < gridDepth) {

                    float3 node_pos = make_float3(i * dx, (j + 0.5f) * dx, (k + 0.5f) * dx);
                    float weight = computeWeight(pos, node_pos, dx);

                    int index = i + j * (gridWidth + 1) + k * (gridWidth + 1) * gridHeight;
                    atomicAdd(&uNum[index], vel.x * weight);
                    atomicAdd(&uDen[index], weight);
                }
            }}

        // Для v-компоненты (грани по Y)
        for (int i_offset = 0; i_offset < 2; i_offset++) {
            for (int k_offset = 0; k_offset < 2; k_offset++) {
                int i = static_cast<int>((pos.x - 0.5f * dx) / dx) + i_offset;
                int j = static_cast<int>(pos.y / dx);
                int k = static_cast<int>((pos.z - 0.5f * dx) / dx) + k_offset;

                if (i >= 0 && i < gridWidth &&
                    j >= 0 && j <= gridHeight &&
                    k >= 0 && k < gridDepth) {

                    float3 node_pos = make_float3((i + 0.5f) * dx, j * dx, (k + 0.5f) * dx);
                    float weight = computeWeight(pos, node_pos, dx);
                    int index = i + j * gridWidth + k * vStride;
                    atomicAdd(&vNum[index], vel.y * weight);
                    atomicAdd(&vDen[index], weight);
                }
            }}

        // Для w-компоненты (грани по Z)
        for (int i_offset = 0; i_offset < 2; i_offset++) {
            for (int j_offset = 0; j_offset < 2; j_offset++) {
                int i = static_cast<int>((pos.x - 0.5f * dx) / dx) + i_offset;
                int j = static_cast<int>((pos.y - 0.5f * dx) / dx) + j_offset;
                int k = static_cast<int>(pos.z / dx);

                if (i >= 0 && i < gridWidth &&
                    j >= 0 && j < gridHeight &&
                    k >= 0 && k <= gridDepth) {

                    float3 node_pos = make_float3((i + 0.5f) * dx, (j + 0.5f) * dx, k * dx);
                    float weight = computeWeight(pos, node_pos, dx);
                    int index = i + j * gridWidth + k * wStride;
                    atomicAdd(&wNum[index], vel.z * weight);
                    atomicAdd(&wDen[index], weight);
                }
            }}
    }
    __device__
    float computeWeight(float3 p_pos, float3 node_pos, float dx) const {
        float3 dist = make_float3(p_pos.x - node_pos.x,
                                  p_pos.y - node_pos.y,
                                  p_pos.z - node_pos.z);
        float wx = fmaxf(0.0f, 1.0f - fabsf(dist.x / dx));
        float wy = fmaxf(0.0f, 1.0f - fabsf(dist.y / dx));
        float wz = fmaxf(0.0f, 1.0f - fabsf(dist.z / dx));
        return wx * wy * wz;
    }
};

//функтор для вычисления скорости (делим собранные num на den)
struct ComputeVelocityFunc {
    __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        float num = thrust::get<0>(t);
        float den = thrust::get<1>(t);
        return (den > 1e-8f) ? num / den : 0.0f;
    }
};

void FluidSolver3D::particlesToGrid() {
    // Размеры компонент скорости
    const int uSize = (gridWidth + 1) * gridHeight * gridDepth;
    const int vSize = gridWidth * (gridHeight + 1) * gridDepth;
    const int wSize = gridWidth * gridHeight * (gridDepth + 1);

    // Инициализация временных аккумуляторов
    VelocityAccumulators accum(uSize, vSize, wSize);

    // Создание функтора для накопления
    AccumulateVelocities accFunc;
    accFunc.dx = dx;
    accFunc.gridWidth = gridWidth;
    accFunc.gridHeight = gridHeight;
    accFunc.gridDepth = gridDepth;
    accFunc.uStride = (gridWidth + 1) * gridHeight;
    accFunc.vStride = gridWidth * (gridHeight + 1);
    accFunc.wStride = gridWidth * gridHeight;
    accFunc.particles = thrust::raw_pointer_cast(d_particles.data());
    accFunc.uNum = thrust::raw_pointer_cast(accum.uNum.data());
    accFunc.uDen = thrust::raw_pointer_cast(accum.uDen.data());
    accFunc.vNum = thrust::raw_pointer_cast(accum.vNum.data());
    accFunc.vDen = thrust::raw_pointer_cast(accum.vDen.data());
    accFunc.wNum = thrust::raw_pointer_cast(accum.wNum.data());
    accFunc.wDen = thrust::raw_pointer_cast(accum.wDen.data());

    // Запуск накопления через Thrust
    thrust::for_each(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(d_particles.size()),
            accFunc
    );

    // Для u-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.uNum.begin(), accum.uDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.uNum.end(), accum.uDen.end())),
            u.device_data.begin(),
            ComputeVelocityFunc()
    );

// Для v-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.vNum.begin(), accum.vDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.vNum.end(), accum.vDen.end())),
            v.device_data.begin(),
            ComputeVelocityFunc()
    );

// Для w-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.wNum.begin(), accum.wDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.wNum.end(), accum.wDen.end())),
            w.device_data.begin(),
            ComputeVelocityFunc()
    );
}

__host__ void FluidSolver3D::frameStep(){
    labelGrid();

    //particles velocities to grid
    particlesToGrid();

    //saving a copy of the current grid velocities (for FLIP)
    saveVelocities();

/*
    // экстраполяция в пределах +одна ячейка (для аккуратной очистки дивергенции)
    extrapolateGridFluidData(u, gridWidth + 1, gridHeight, 2);
    extrapolateGridFluidData(v, gridWidth, gridHeight + 1, 2);


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
    cleanUpParticles(dx/2.0f);*/
}

__host__ void FluidSolver3D::run(int max_steps) {
    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    for(int i = 0; i < max_steps; ++i){
        frameStep();
        if(i%10 == 0){
            Utility::save3dParticlesToPLY(h_particles, "InputData/particles_" + std::to_string(i) + ".ply");
            std::cout << "frame = " << i/10 << "; numParticles = " << h_particles.size()<<std::endl;
        }

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    std::cout << "elapsed time = " << elapsedTime / 1000.0f << std::endl;
}
