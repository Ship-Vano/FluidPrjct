#include "FluidSolver3D.cuh"

//basic funcs################1
FluidSolver3D::FluidSolver3D(int width, int height, int depth, float dx_, float dt_){
    gridWidth = width;
    gridHeight = height;
    gridDepth = depth;
    w_x_h_x_d = gridWidth * gridHeight * gridDepth;
    dx = dx_;
    dy = dx;
    dz = dx;
    dt = dt_;
    std::cout << "dx=" << dx<<"\n";
    h_particles = thrust::host_vector<Utility::Particle3D>();
    d_particles = thrust::device_vector<Utility::Particle3D>();

    // Инициализация CUDA stream
    stream = nullptr;
    cudaStreamCreate(&stream);

    // Создание cuDSS handle
    handle = nullptr;
    cudssStatus_t status = cudssCreate(&handle);
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error("cuDSS init failed: " + std::to_string(status));
    }

    // Привязка stream к handle
    cudssSetStream(handle, stream);

    // Создание конфигурации и данных решателя
    cudssConfigCreate(&solverConfig);
    cudssDataCreate(handle, &solverData);

}

FluidSolver3D::~FluidSolver3D(){
    // Уничтожение объектов cuDSS
    if (solverData) cudssDataDestroy(handle, solverData);
    if (solverConfig) cudssConfigDestroy(solverConfig);
    if (handle) cudssDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
}

struct MarkBodyCellsFunctor {
    float* sdf_data;
    float3 sdf_origin;
    float sdf_cell_size;
    int sdf_w; int sdf_h; int sdf_d;
    int* labels_ptr;
    int W, H;
    float dx;
    float3 body_pos;
    float* rotation_matrix;

    __host__ __device__
    MarkBodyCellsFunctor(float* sdf_data_, float3 sdf_orig, float sdf_cell_size_, int sdf_w_, int sdf_h_, int sdf_d_,
                         int* labels, int width, int height, float cell_size, float3 body_pos_, float* rotation_matrix_)
            : labels_ptr(labels), W(width), H(height), dx(cell_size),
              sdf_data(sdf_data_), sdf_origin(sdf_orig), sdf_cell_size(sdf_cell_size_),
              sdf_w(sdf_w_), sdf_h(sdf_h_), sdf_d(sdf_d_), body_pos(body_pos_), rotation_matrix(rotation_matrix_){}

    __device__
    void operator()(int idx) const {
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W*H);

        float3 cell_center = make_float3(
                (i+0.5f)*dx, (j+0.5f)*dx, (k+0.5f)*dx
        );

        if (Utility::contains(sdf_data, sdf_origin, body_pos, rotation_matrix, cell_center, sdf_cell_size, sdf_w, sdf_h, sdf_d) && labels_ptr[idx] != Utility::SOLID) {
            labels_ptr[idx] = Utility::BODY;
        }
    }
};


__host__ void FluidSolver3D::init(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + fileName);
    }

    // Чтение размеров
    file >> gridWidth >> gridHeight >> gridDepth;
    w_x_h_x_d = gridWidth * gridHeight * gridDepth;
    // Инициализация сеток
    labels.change_size(gridWidth, gridHeight, gridDepth);
    p.change_size(gridWidth, gridHeight, gridDepth);
    u.change_size(gridWidth+1, gridHeight, gridDepth);
    v.change_size(gridWidth, gridHeight+1, gridDepth);
    w.change_size(gridWidth, gridHeight, gridDepth+1);

    uSaved.change_size(gridWidth+1, gridHeight, gridDepth);
    vSaved.change_size(gridWidth, gridHeight+1, gridDepth);
    wSaved.change_size(gridWidth, gridHeight, gridDepth+1);

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


    // Инициализация тела
    //float3 initial_position = make_float3(1.75f, 2.1f, 1.75f);  // Желаемая начальная позиция
    body.loadSDF("InputData/ball.sdf", initialBodyPos);

    // Физические свойства
    body.mass = bodyMass;
    body.vel = make_float3(0.0f, 0.0f, 0.0f);
    body.force = make_float3(0.0f, 0.0f, 0.0f);

    // Момент инерции (для сферы)
    float radius = body.size.x / 2.0f;  // Предполагаем сферическое тело
    body.inertia = 0.4f * body.mass * radius * radius;
    body.inv_inertia = 1.0f / body.inertia;
    body.omega = make_float3(0.0f, 0.0f, 0.0f); //угловая скорость
    body.torque = make_float3(0.0f,0.0f,0.0f);      // суммарный момент

    thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator(0),
            gridWidth* gridHeight * gridDepth,
            MarkBodyCellsFunctor {
                    body.sdf_data.device_ptr(),
                    body.sdf_origin,
                    body.sdf_cell_size,
                    body.sdf_data.width(),
                    body.sdf_data.height(),
                    body.sdf_data.depth(),
                    labels.device_ptr(),
                    gridWidth,
                    gridHeight,
                    dx,body.pos,  thrust::raw_pointer_cast(body.rotation_matrix_d.data())
            }
    );
    cudaDeviceSynchronize();
    labels.host_data = labels.device_data;

    // Инициализация частиц
    seedParticles(PARTICLES_PER_CELL);
    std::cout <<"Number of particles is" << h_particles.size() << std::endl;
    Utility::save3dParticlesToPLY(h_particles, "InputData/particles_-1.ply");


}

struct FluidFlagFunctor {
    __host__ __device__
    int operator()(int label) const {
        return label == Utility::FLUID ? 1 : 0;
    }
};

__host__ void FluidSolver3D::seedParticles(int particlesPerCell){
    // Инициализация генератора (один раз вне функции!)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> subCellDist(0, 7);
    static std::uniform_real_distribution<> jitterDist(-0.24f, 0.24f);

    // Сначала подсчитываем общее количество частиц
    h_particles.clear();
    size_t totalParticles = 0;
    thrust::device_vector<int> flags(w_x_h_x_d, 0);
    thrust::transform(
            thrust::device,
            labels.device_ptr(),
            labels.device_ptr() + w_x_h_x_d,
            flags.begin(),
            FluidFlagFunctor()
    );
    fluidCellsAmount = thrust::reduce(
            thrust::device,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
    );
    totalParticles = fluidCellsAmount * particlesPerCell;

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
                    for (int pind = 0; pind < particlesPerCell; ++pind) {
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
    //thrust::copy(h_particles.begin(), h_particles.end(), d_particles.begin());
    d_particles = h_particles;
    blocksForParticles = (h_particles.size() + threadsPerBlock- 1) / threadsPerBlock;
}


struct ClearNonSolidFunctor {
    __host__ __device__
    int operator()(const int& oldLabel) const {
        return (oldLabel == Utility::SOLID) ? Utility::SOLID : Utility::AIR;
    }
};
struct MarkFluidCellsFunctor {
    const Utility::Particle3D* particles;
    float dx;
    int W, H, D;
    int* labels;    // raw‐pointer на labels.device_data

    MarkFluidCellsFunctor(const Utility::Particle3D* _particles,
                          float _dx, int _W, int _H, int _D,
                          int* _labels)
            : particles(_particles),
              dx(_dx),
              W(_W), H(_H), D(_D),
              labels(_labels) {}

    __device__
    void operator()(int pid) const {
        float3 p = particles[pid].pos;
        int i = static_cast<int>(floorf(p.x / dx));
        int j = static_cast<int>(floorf(p.y / dx));
        int k = static_cast<int>(floorf(p.z / dx));
        if (i < 0)   i = 0;
        if (i >= W) i = W - 1;
        if (j < 0)   j = 0;
        if (j >= H) j = H - 1;
        if (k < 0)   k = 0;
        if (k >= D) k = D - 1;
        int idx = i + j * W + k * (W * H);
        if(labels[idx]!= Utility::SOLID && labels[idx] != Utility::BODY){
            labels[idx] = Utility::FLUID;
        }
    }
};


__host__ int FluidSolver3D::labelGrid() {
    int totalCells = labels.width() * labels.height() * labels.depth();

    // 1) Для каждой ячейки: если была SOLID, остаётся SOLID; иначе – AIR
    thrust::transform(
            thrust::device,
            labels.device_data.begin(),
            labels.device_data.begin() + totalCells,
            labels.device_data.begin(),
            ClearNonSolidFunctor()
    );

     thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator(0),
            gridWidth* gridHeight * gridDepth,
            MarkBodyCellsFunctor {
                    body.sdf_data.device_ptr(),
                    body.sdf_origin,
                    body.sdf_cell_size,
                    body.sdf_data.width(),
                    body.sdf_data.height(),
                    body.sdf_data.depth(),
                    labels.device_ptr(),
                    gridWidth,
                    gridHeight,
                    dx, body.pos,  thrust::raw_pointer_cast(body.rotation_matrix_d.data())
            }
    );

    // 2) Пометка FLUID-ячееk по текущим частицам
    int numParticles = static_cast<int>(d_particles.size());
    if (numParticles > 0) {
        MarkFluidCellsFunctor functor(
                thrust::raw_pointer_cast(d_particles.data()),
                dx,
                gridWidth, gridHeight, gridDepth,
                thrust::raw_pointer_cast(labels.device_data.data())
        );
        thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<int>(0),
                thrust::make_counting_iterator<int>(numParticles),
                functor
        );
    }

   

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
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "saveVelocities() error: " << cudaGetErrorString(err) << std::endl;
    }
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
    // Размеры сетки
    int gridW, gridH, gridD;
    float dx;

    // Полушаги для MAC-узлов
    // u-грань: узлы по X имеют размер (gridW+1)×gridH×gridD
    // v-грань: узлы по Y имеют размер gridW×(gridH+1)×gridD
    // w-грань: узлы по Z имеют размер gridW×gridH×(gridD+1)

    // Сырой указатель на массив частиц
    const Utility::Particle3D* particles;
    int numParticles;

    // Указатели на временные аккумуляторы
    float* uNum;
    float* uDen;
    float* vNum;
    float* vDen;
    float* wNum;
    float* wDen;

    // Индексы stride для удобства
    // Для u: strideU_xy = (gridW+1) * gridH  (то есть шаг по Z)
    // Для v: strideV_xy = gridW * (gridH+1)
    // Для w: strideW_xy = gridW * gridH
    int strideU_z;
    int strideV_z;
    int strideW_z;

    __device__
    void operator()(int pid) const {
        // Получаем частицу
        const Utility::Particle3D& P = particles[pid];
        float3 pos = P.pos;
        float3 vel = P.vel;

        // Нормализуем к “ячейковым” координатам
        float rx = pos.x / dx;
        float ry = pos.y / dx;
        float rz = pos.z / dx;

        // --- Расчёт вкладов для компоненты U (face-centered по X) ---
        // базовые индексы iU ∈ [0..gridW], jU ∈ [0..gridH-1], kU ∈ [0..gridD-1]
        int iU = floorf(rx);
        int jU = floorf(ry - 0.5f);
        int kU = floorf(rz - 0.5f);
        iU = min(max(iU, 0),     gridW);
        jU = min(max(jU, 0),     gridH - 1);
        kU = min(max(kU, 0),     gridD - 1);

        float fxU = rx      - iU;
        float fyU = (ry - 0.5f) - jU;
        float fzU = (rz - 0.5f) - kU;
        fxU = fminf(fmaxf(fxU, 0.0f), 1.0f);
        fyU = fminf(fmaxf(fyU, 0.0f), 1.0f);
        fzU = fminf(fmaxf(fzU, 0.0f), 1.0f);

        // Проходим по 8 соседним узлам U
        for (int dz = 0; dz < 2; ++dz) {
            int k = kU + dz;
            if (k < 0 || k >= gridD) continue;
            for (int dy = 0; dy < 2; ++dy) {
                int j = jU + dy;
                if (j < 0 || j >= gridH) continue;
                for (int dxU = 0; dxU < 2; ++dxU) {
                    int i = iU + dxU;
                    if (i < 0 || i > gridW) continue;

                    float wx = (dxU == 0 ? (1.0f - fxU) : fxU);
                    float wy = (dy  == 0 ? (1.0f - fyU) : fyU);
                    float wz = (dz  == 0 ? (1.0f - fzU) : fzU);
                    float wgt = wx * wy * wz;

                    int idx = i + j * (gridW + 1) + k * strideU_z;
                    atomicAdd(&uNum[idx], vel.x * wgt);
                    atomicAdd(&uDen[idx],        wgt);
                }
            }
        }

        // --- Расчёт вкладов для компоненты V (face-centered по Y) ---
        // базовые индексы iV ∈ [0..gridW-1], jV ∈ [0..gridH], kV ∈ [0..gridD-1]
        int iV = floorf(rx - 0.5f);
        int jV = floorf(ry);
        int kV = floorf(rz - 0.5f);
        iV = min(max(iV, 0),     gridW - 1);
        jV = min(max(jV, 0),     gridH);
        kV = min(max(kV, 0),     gridD - 1);

        float fxV = (rx - 0.5f) - iV;
        float fyV = ry        - jV;
        float fzV = (rz - 0.5f) - kV;
        fxV = fminf(fmaxf(fxV, 0.0f), 1.0f);
        fyV = fminf(fmaxf(fyV, 0.0f), 1.0f);
        fzV = fminf(fmaxf(fzV, 0.0f), 1.0f);

        for (int dz = 0; dz < 2; ++dz) {
            int k = kV + dz;
            if (k < 0 || k >= gridD) continue;
            for (int dy = 0; dy < 2; ++dy) {
                int j = jV + dy;
                if (j < 0 || j > gridH) continue;
                for (int dxV = 0; dxV < 2; ++dxV) {
                    int i = iV + dxV;
                    if (i < 0 || i >= gridW) continue;

                    float wx = (dxV == 0 ? (1.0f - fxV) : fxV);
                    float wy = (dy  == 0 ? (1.0f - fyV) : fyV);
                    float wz = (dz  == 0 ? (1.0f - fzV) : fzV);
                    float wgt = wx * wy * wz;

                    int idx = i + j * gridW + k * strideV_z;
                    atomicAdd(&vNum[idx], vel.y * wgt);
                    atomicAdd(&vDen[idx],        wgt);
                }
            }
        }

        // --- Расчёт вкладов для компоненты W (face-centered по Z) ---
        // базовые индексы iW ∈ [0..gridW-1], jW ∈ [0..gridH-1], kW ∈ [0..gridD]
        int iW = floorf(rx - 0.5f);
        int jW = floorf(ry - 0.5f);
        int kW = floorf(rz);
        iW = min(max(iW, 0),     gridW - 1);
        jW = min(max(jW, 0),     gridH - 1);
        kW = min(max(kW, 0),     gridD);

        float fxW = (rx - 0.5f) - iW;
        float fyW = (ry - 0.5f) - jW;
        float fzW = rz        - kW;
        fxW = fminf(fmaxf(fxW, 0.0f), 1.0f);
        fyW = fminf(fmaxf(fyW, 0.0f), 1.0f);
        fzW = fminf(fmaxf(fzW, 0.0f), 1.0f);

        for (int dz = 0; dz < 2; ++dz) {
            int k = kW + dz;
            if (k < 0 || k > gridD) continue;
            for (int dy = 0; dy < 2; ++dy) {
                int j = jW + dy;
                if (j < 0 || j >= gridH) continue;
                for (int dxW = 0; dxW < 2; ++dxW) {
                    int i = iW + dxW;
                    if (i < 0 || i >= gridW) continue;

                    float wx = (dxW == 0 ? (1.0f - fxW) : fxW);
                    float wy = (dy  == 0 ? (1.0f - fyW) : fyW);
                    float wz = (dz  == 0 ? (1.0f - fzW) : fzW);
                    float wgt = wx * wy * wz;

                    int idx = i + j * gridW + k * strideW_z;
                    atomicAdd(&wNum[idx], vel.z * wgt);
                    atomicAdd(&wDen[idx],        wgt);
                }
            }
        }
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
    // 3.1) Размеры компонент скорости
    const int uSize = (gridWidth + 1) * gridHeight * gridDepth;
    const int vSize = gridWidth * (gridHeight + 1) * gridDepth;
    const int wSize = gridWidth * gridHeight * (gridDepth + 1);

    // 3.2) Создаём временные аккумуляторы и обнуляем их
    VelocityAccumulators accum(uSize, vSize, wSize);

    // Заполняем нулями
    thrust::fill(accum.uNum.begin(), accum.uNum.end(), 0.0f);
    thrust::fill(accum.uDen.begin(), accum.uDen.end(), 0.0f);
    thrust::fill(accum.vNum.begin(), accum.vNum.end(), 0.0f);
    thrust::fill(accum.vDen.begin(), accum.vDen.end(), 0.0f);
    thrust::fill(accum.wNum.begin(), accum.wNum.end(), 0.0f);
    thrust::fill(accum.wDen.begin(), accum.wDen.end(), 0.0f);

    // 3.3) Настраиваем функтор накопления
    AccumulateVelocities accFunc;
    accFunc.dx        = dx;
    accFunc.gridW     = gridWidth;
    accFunc.gridH     = gridHeight;
    accFunc.gridD     = gridDepth;
    accFunc.particles = thrust::raw_pointer_cast(d_particles.data());
    accFunc.numParticles = static_cast<int>(d_particles.size());
    accFunc.uNum = thrust::raw_pointer_cast(accum.uNum.data());
    accFunc.uDen = thrust::raw_pointer_cast(accum.uDen.data());
    accFunc.vNum = thrust::raw_pointer_cast(accum.vNum.data());
    accFunc.vDen = thrust::raw_pointer_cast(accum.vDen.data());
    accFunc.wNum = thrust::raw_pointer_cast(accum.wNum.data());
    accFunc.wDen = thrust::raw_pointer_cast(accum.wDen.data());

    // Вычисляем strides (шаг по Z) для каждого массива
    accFunc.strideU_z = (gridWidth + 1) * gridHeight;
    accFunc.strideV_z =  gridWidth       * (gridHeight + 1);
    accFunc.strideW_z =  gridWidth       *  gridHeight;

    // 3.4) Запускаем накопление атомарными операциями
    thrust::for_each(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(static_cast<int>(d_particles.size())),
            accFunc
    );

    // 3.5) Нормализуем (num/den) → записываем в u,v,w
    // Для u-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.uNum.begin(), accum.uDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.uNum.end(),   accum.uDen.end())),
            u.device_data.begin(),
            ComputeVelocityFunc()
    );

    // Для v-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.vNum.begin(), accum.vDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.vNum.end(),   accum.vDen.end())),
            v.device_data.begin(),
            ComputeVelocityFunc()
    );

    // Для w-компоненты
    thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(accum.wNum.begin(), accum.wDen.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(accum.wNum.end(),   accum.wDen.end())),
            w.device_data.begin(),
            ComputeVelocityFunc()
    );

    // 3.6) Проверяем ошибки CUDA
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "P2G error: " << cudaGetErrorString(err) << std::endl;
    }
}

struct ApplyScalarForce
{
    float dt, a, vel_unknown;

    ApplyScalarForce(float _dt, float _a, float _vel_unknown)
            : dt(_dt), a(_a), vel_unknown(_vel_unknown) {}

    __host__ __device__
    float operator()(const float& x) const {
        return (x > vel_unknown) ? x + dt * a : x;
    }
};

void FluidSolver3D::applyForces(){

    thrust::transform(
            thrust::device,
            u.device_data.begin(), u.device_data.end(),         // вход
            u.device_data.begin(),                  // выход
            ApplyScalarForce(dt, GRAVITY.x, VEL_UNKNOWN)
    );
//    u.copy_to_host();
//        std::cout << "----u 3d---" << std::endl;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth+1; ++i){
//                std::cout << u.host_data[i + j*(gridWidth+1) + k * (gridWidth+1)*gridHeight] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }

    thrust::transform(
            thrust::device,
            v.device_data.begin(), v.device_data.end(),
            v.device_data.begin(),
            ApplyScalarForce(dt, GRAVITY.y, VEL_UNKNOWN)
    );

//    v.copy_to_host();
//    std::cout << "----v 3d---" << std::endl;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight+1; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << v.host_data[i + j*(gridWidth) + k * (gridWidth)*(gridHeight+1)] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }

    thrust::transform(
            thrust::device,
            w.device_data.begin(), w.device_data.end(),
            w.device_data.begin(),
            ApplyScalarForce(dt, GRAVITY.z, VEL_UNKNOWN)
    );

//    w.copy_to_host();
//    std::cout << "----w 3d---" << std::endl;
//    for(int k = 0; k < gridDepth+1; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << w.host_data[i + j*(gridWidth) + k * (gridWidth)*(gridHeight)] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ApplyForces() error: " << cudaGetErrorString(err) << std::endl;
    }
}

// ----------------------------------
// 1) Вспомогательная __device__-функция трёхлинейной интерполяции
__host__ __device__
float trilerp(const float* fld,
              int i,int j,int k,
              float fx,float fy,float fz,
              int stride_i,int stride_j)
{
#define SAMP(ii,jj,kk) fld[(ii) + (jj)*stride_i + (kk)*stride_j]
    float c00 = SAMP(i  , j  , k  )*(1-fx) + SAMP(i+1, j  , k  )*fx;
    float c10 = SAMP(i  , j+1, k  )*(1-fx) + SAMP(i+1, j+1, k  )*fx;
    float c01 = SAMP(i  , j  , k+1)*(1-fx) + SAMP(i+1, j  , k+1)*fx;
    float c11 = SAMP(i  , j+1, k+1)*(1-fx) + SAMP(i+1, j+1, k+1)*fx;
    float c0  = c00*(1-fy) + c10*fy;
    float c1  = c01*(1-fy) + c11*fy;
    float c   = c0*(1-fz) + c1*fz;
#undef SAMP
    return c;
}

__host__ __device__
float trilinearInterpolation(
        const float xd, const float yd, const float zd,
        const float c000, const float c100, const float c001, const float c101,
        const float c010, const float c110, const float c011, const float c111)
{
    // Интерполяция вдоль X
    float c00 = c000 * (1.0f - xd) + c100 * xd;  // z=0, y=0
    float c01 = c001 * (1.0f - xd) + c101 * xd;  // z=1, y=0
    float c10 = c010 * (1.0f - xd) + c110 * xd;  // z=0, y=1
    float c11 = c011 * (1.0f - xd) + c111 * xd;  // z=1, y=1

    // Интерполяция вдоль Y
    float c0 = c00 * (1.0f - yd) + c10 * yd;  // z=0
    float c1 = c01 * (1.0f - yd) + c11 * yd;  // z=1

    // Интерполяция вдоль Z
    return c0 * (1.0f - zd) + c1 * zd;
}
// ----------------------------------
struct GridToParticleFunctor
{
    int W, H, D;
    float  dx, alpha;

    // raw-указатели на device_data[]
    const float *u, *v, *w;
    const float *du, *dv, *dw;

    GridToParticleFunctor(int _W, int _H, int _D,
                          float _dx, float _alpha,
                          const float* _u, const float* _v, const float* _w,
                          const float* _du, const float* _dv, const float* _dw)
            : W(_W), H(_H), D(_D),
              dx(_dx), alpha(_alpha),
              u(_u), v(_v), w(_w),
              du(_du), dv(_dv), dw(_dw) {}

    __device__
    Utility::Particle3D operator()(const Utility::Particle3D& pin) const
    {
        Utility::Particle3D pout = pin;

        // 1) Нормализованные “cell-space” координаты
        float rx = pin.pos.x / dx;
        float ry = pin.pos.y / dx;
        float rz = pin.pos.z / dx;

        // --- Интерполяция U (MAC face по X) ---
        // базовые индексы: iU ∈ [0..W], jU ∈ [0..H-1], kU ∈ [0..D-1]
        int iU = floorf(rx);
        int jU = floorf(ry - 0.5f);
        int kU = floorf(rz - 0.5f);
        // зажимаем в допустимый диапазон
        iU = min(max(iU, 0),     W);
        jU = min(max(jU, 0),     H - 1);
        kU = min(max(kU, 0),     D - 1);
        // дробные части внутри “ячейки” U (координаты относительно ячейки)
        float fxU = rx - iU;
        float fyU = (ry - 0.5f) - jU;
        float fzU = (rz - 0.5f) - kU;
        fxU = fminf(fmaxf(fxU, 0.0f), 1.0f);
        fyU = fminf(fmaxf(fyU, 0.0f), 1.0f);
        fzU = fminf(fmaxf(fzU, 0.0f), 1.0f);

        // адресация 8 вершин массива u (размер (W+1) × H × D):
        // idx_u(i,j,k) = i + j*(W+1) + k*(W+1)*H
        int baseU = jU * (W + 1) + kU * (W + 1) * H;
        float u000 = u[ iU     + baseU ];
        float u100 = u[(iU + 1) + baseU ];
        float u010 = u[ iU     + (jU + 1) * (W + 1) + kU * (W + 1) * H ];
        float u110 = u[(iU + 1) + (jU + 1) * (W + 1) + kU * (W + 1) * H ];
        float u001 = u[ iU     + jU * (W + 1) + (kU + 1) * (W + 1) * H ];
        float u101 = u[(iU + 1) + jU * (W + 1) + (kU + 1) * (W + 1) * H ];
        float u011 = u[ iU     + (jU + 1) * (W + 1) + (kU + 1) * (W + 1) * H ];
        float u111 = u[(iU + 1) + (jU + 1) * (W + 1) + (kU + 1) * (W + 1) * H ];
        float uPIC = trilinearInterpolation(
                fxU, fyU, fzU,
                u000, u100, u001, u101,
                u010, u110, u011, u111
        );

        float du000 = du[ iU     + baseU ];
        float du100 = du[(iU + 1) + baseU ];
        float du010 = du[ iU     + (jU + 1) * (W + 1) + kU * (W + 1) * H ];
        float du110 = du[(iU + 1) + (jU + 1) * (W + 1) + kU * (W + 1) * H ];
        float du001 = du[ iU     + jU * (W + 1) + (kU + 1) * (W + 1) * H ];
        float du101 = du[(iU + 1) + jU * (W + 1) + (kU + 1) * (W + 1) * H ];
        float du011 = du[ iU     + (jU + 1) * (W + 1) + (kU + 1) * (W + 1) * H ];
        float du111 = du[(iU + 1) + (jU + 1) * (W + 1) + (kU + 1) * (W + 1) * H ];
        float duFLIP = trilinearInterpolation(
                fxU, fyU, fzU,
                du000, du100, du001, du101,
                du010, du110, du011, du111
        );

        // --- Интерполяция V (MAC face по Y) ---
        // iV ∈ [0..W-1], jV ∈ [0..H], kV ∈ [0..D-1]
        int iV = floorf(rx - 0.5f);
        int jV = floorf(ry);
        int kV = floorf(rz - 0.5f);
        iV = min(max(iV, 0),     W - 1);
        jV = min(max(jV, 0),     H);
        kV = min(max(kV, 0),     D - 1);

        float fxV = (rx - 0.5f) - iV;
        float fyV = ry - jV;
        float fzV = (rz - 0.5f) - kV;
        fxV = fminf(fmaxf(fxV, 0.0f), 1.0f);
        fyV = fminf(fmaxf(fyV, 0.0f), 1.0f);
        fzV = fminf(fmaxf(fzV, 0.0f), 1.0f);

        // idx_v(i,j,k) = i + j*W + k*(W*(H+1))
        int baseV = jV * W + kV * (W * (H + 1));
        float v000 = v[ iV     + baseV ];
        float v100 = v[(iV + 1) + baseV ];
        float v010 = v[ iV     + (jV + 1) * W + kV * (W * (H + 1)) ];
        float v110 = v[(iV + 1) + (jV + 1) * W + kV * (W * (H + 1)) ];
        float v001 = v[ iV     + jV * W + (kV + 1) * (W * (H + 1)) ];
        float v101 = v[(iV + 1) + jV * W + (kV + 1) * (W * (H + 1)) ];
        float v011 = v[ iV     + (jV + 1) * W + (kV + 1) * (W * (H + 1)) ];
        float v111 = v[(iV + 1) + (jV + 1) * W + (kV + 1) * (W * (H + 1)) ];
        float vPIC = trilinearInterpolation(
                fxV, fyV, fzV,
                v000, v100, v001, v101,
                v010, v110, v011, v111
        );

        float dv000 = dv[ iV     + baseV ];
        float dv100 = dv[(iV + 1) + baseV ];
        float dv010 = dv[ iV     + (jV + 1) * W + kV * (W * (H + 1)) ];
        float dv110 = dv[(iV + 1) + (jV + 1) * W + kV * (W * (H + 1)) ];
        float dv001 = dv[ iV     + jV * W + (kV + 1) * (W * (H + 1)) ];
        float dv101 = dv[(iV + 1) + jV * W + (kV + 1) * (W * (H + 1)) ];
        float dv011 = dv[ iV     + (jV + 1) * W + (kV + 1) * (W * (H + 1)) ];
        float dv111 = dv[(iV + 1) + (jV + 1) * W + (kV + 1) * (W * (H + 1)) ];
        float dvFLIP = trilinearInterpolation(
                fxV, fyV, fzV,
                dv000, dv100, dv001, dv101,
                dv010, dv110, dv011, dv111
        );

        // --- Интерполяция W (MAC face по Z) ---
        // iW ∈ [0..W-1], jW ∈ [0..H-1], kW ∈ [0..D]
        int iW = floorf(rx - 0.5f);
        int jW = floorf(ry - 0.5f);
        int kW = floorf(rz);
        iW = min(max(iW, 0),     W - 1);
        jW = min(max(jW, 0),     H - 1);
        kW = min(max(kW, 0),     D);

        float fxW = (rx - 0.5f) - iW;
        float fyW = (ry - 0.5f) - jW;
        float fzW = rz - kW;
        fxW = fminf(fmaxf(fxW, 0.0f), 1.0f);
        fyW = fminf(fmaxf(fyW, 0.0f), 1.0f);
        fzW = fminf(fmaxf(fzW, 0.0f), 1.0f);

        // idx_w(i,j,k) = i + j*W + k*(W*H)
        int baseW = jW * W + kW * (W * H);
        float w000 = w[ iW     + baseW ];
        float w100 = w[(iW + 1) + baseW ];
        float w010 = w[ iW     + (jW + 1) * W + kW * (W * H) ];
        float w110 = w[(iW + 1) + (jW + 1) * W + kW * (W * H) ];
        float w001 = w[ iW     + jW * W + (kW + 1) * (W * H) ];
        float w101 = w[(iW + 1) + jW * W + (kW + 1) * (W * H) ];
        float w011 = w[ iW     + (jW + 1) * W + (kW + 1) * (W * H) ];
        float w111 = w[(iW + 1) + (jW + 1) * W + (kW + 1) * (W * H) ];
        float wPIC = trilinearInterpolation(
                fxW, fyW, fzW,
                w000, w100, w001, w101,
                w010, w110, w011, w111
        );

        float dw000 = dw[ iW     + baseW ];
        float dw100 = dw[(iW + 1) + baseW ];
        float dw010 = dw[ iW     + (jW + 1) * W + kW * (W * H) ];
        float dw110 = dw[(iW + 1) + (jW + 1) * W + kW * (W * H) ];
        float dw001 = dw[ iW     + jW * W + (kW + 1) * (W * H) ];
        float dw101 = dw[(iW + 1) + jW * W + (kW + 1) * (W * H) ];
        float dw011 = dw[ iW     + (jW + 1) * W + (kW + 1) * (W * H) ];
        float dw111 = dw[(iW + 1) + (jW + 1) * W + (kW + 1) * (W * H) ];
        float dwFLIP = trilinearInterpolation(
                fxW, fyW, fzW,
                dw000, dw100, dw001, dw101,
                dw010, dw110, dw011, dw111
        );

        // 2) Форма FLIP: pin.vel + delta-скорость, а PIC – это просто uPIC,vPIC,wPIC
        float newU = alpha * uPIC + (1.0f - alpha) * (pin.vel.x + duFLIP);
        float newV = alpha * vPIC + (1.0f - alpha) * (pin.vel.y + dvFLIP);
        float newW = alpha * wPIC + (1.0f - alpha) * (pin.vel.z + dwFLIP);

        pout.vel = make_float3(newU, newV, newW);

        return pout;
    }
};

// ----------------------------------
void FluidSolver3D::gridToParticles(float alpha)
{
    // 1) размеры MAC-решёток
    int Nu = (gridWidth + 1) * gridHeight * gridDepth;
    int Nv = gridWidth * (gridHeight + 1) * gridDepth;
    int Nw = gridWidth * gridHeight * (gridDepth + 1);

    // 2) вычисляем дельты (new – old) в отдельные device_vector
    thrust::device_vector<float> du(Nu);
    thrust::device_vector<float> dv(Nv);
    thrust::device_vector<float> dw(Nw);

    thrust::transform(
            u.device_data.begin(), u.device_data.end(),
            uSaved.device_data.begin(),
            du.begin(),
            thrust::minus<float>()
    );
    thrust::transform(
            v.device_data.begin(), v.device_data.end(),
            vSaved.device_data.begin(),
            dv.begin(),
            thrust::minus<float>()
    );
    thrust::transform(
            w.device_data.begin(), w.device_data.end(),
            wSaved.device_data.begin(),
            dw.begin(),
            thrust::minus<float>()
    );

    // 3) получаем raw-указатели
    const float* pu  = u.device_ptr();
    const float* pv  = v.device_ptr();
    const float* pw  = w.device_ptr();
    const float* pdu = thrust::raw_pointer_cast(du.data());
    const float* pdv = thrust::raw_pointer_cast(dv.data());
    const float* pdw = thrust::raw_pointer_cast(dw.data());

    // 4) запускаем один transform по всем частицам
    thrust::transform(
            d_particles.begin(),
            d_particles.end(),
            d_particles.begin(),
            GridToParticleFunctor(
                    gridWidth, gridHeight, gridDepth,
                    dx, alpha,
                    pu, pv, pw,
                    pdu, pdv, pdw
            )
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "G2P error: " << cudaGetErrorString(err) << std::endl;
    }
}

__device__ inline bool isCellValid(int x, int y, int z, int W, int H, int D) {
    return x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
}

__device__ inline int idx3d(int x , int y, int z, int W, int H){
    return x + y * W + z * W * H;
}

// ----------------------------------
// 1) Исправленная функция интерполяции MAC-скоростей (без FLIP-дельт)
__device__
float3 interpVelDevice3D(const float* u, const float* v, const float* w,
                         int W, int H, int D, float dx, float3 pos)
{
    // Преобразуем позицию в “ячейковые” координаты
    float rx = pos.x / dx;
    float ry = pos.y / dx;
    float rz = pos.z / dx;

    // --- Интерполяция U (MAC-грань по X) ---
    // iU ∈ [0..W], jU ∈ [0..H-1], kU ∈ [0..D-1]
    int iU = floorf(rx);
    int jU = floorf(ry - 0.5f);
    int kU = floorf(rz - 0.5f);
    iU = min(max(iU, 0),     W);
    jU = min(max(jU, 0),     H - 1);
    kU = min(max(kU, 0),     D - 1);

    float fxU = rx - iU;
    float fyU = (ry - 0.5f) - jU;
    float fzU = (rz - 0.5f) - kU;
    fxU = fminf(fmaxf(fxU, 0.0f), 1.0f);
    fyU = fminf(fmaxf(fyU, 0.0f), 1.0f);
    fzU = fminf(fmaxf(fzU, 0.0f), 1.0f);

    // Линейный индекс в массиве u: idx_u(i,j,k) = i + j*(W+1) + k*(W+1)*H
    int baseU = jU * (W + 1) + kU * (W + 1) * H;
    float u000 = u[ iU     + baseU ];
    float u100 = u[(iU + 1) + baseU ];
    float u010 = u[ iU     + (jU + 1)*(W + 1) + kU*(W + 1)*H ];
    float u110 = u[(iU + 1) + (jU + 1)*(W + 1) + kU*(W + 1)*H ];
    float u001 = u[ iU     + jU*(W + 1) + (kU + 1)*(W + 1)*H ];
    float u101 = u[(iU + 1) + jU*(W + 1) + (kU + 1)*(W + 1)*H ];
    float u011 = u[ iU     + (jU + 1)*(W + 1) + (kU + 1)*(W + 1)*H ];
    float u111 = u[(iU + 1) + (jU + 1)*(W + 1) + (kU + 1)*(W + 1)*H ];
    float uInterp = trilinearInterpolation(
            fxU, fyU, fzU,
            u000, u100, u001, u101,
            u010, u110, u011, u111
    );

    // --- Интерполяция V (MAC-грань по Y) ---
    // iV ∈ [0..W-1], jV ∈ [0..H], kV ∈ [0..D-1]
    int iV = floorf(rx - 0.5f);
    int jV = floorf(ry);
    int kV = floorf(rz - 0.5f);
    iV = min(max(iV, 0),     W - 1);
    jV = min(max(jV, 0),     H);
    kV = min(max(kV, 0),     D - 1);

    float fxV = (rx - 0.5f) - iV;
    float fyV = ry - jV;
    float fzV = (rz - 0.5f) - kV;
    fxV = fminf(fmaxf(fxV, 0.0f), 1.0f);
    fyV = fminf(fmaxf(fyV, 0.0f), 1.0f);
    fzV = fminf(fmaxf(fzV, 0.0f), 1.0f);

    // idx_v(i,j,k) = i + j*W + k*(W*(H+1))
    int baseV = jV * W + kV * (W * (H + 1));
    float v000 = v[ iV     + baseV ];
    float v100 = v[(iV + 1) + baseV ];
    float v010 = v[ iV     + (jV + 1)*W + kV*(W*(H + 1)) ];
    float v110 = v[(iV + 1) + (jV + 1)*W + kV*(W*(H + 1)) ];
    float v001 = v[ iV     + jV*W + (kV + 1)*(W*(H + 1)) ];
    float v101 = v[(iV + 1) + jV*W + (kV + 1)*(W*(H + 1)) ];
    float v011 = v[ iV     + (jV + 1)*W + (kV + 1)*(W*(H + 1)) ];
    float v111 = v[(iV + 1) + (jV + 1)*W + (kV + 1)*(W*(H + 1)) ];
    float vInterp = trilinearInterpolation(
            fxV, fyV, fzV,
            v000, v100, v001, v101,
            v010, v110, v011, v111
    );

    // --- Интерполяция W (MAC-грань по Z) ---
    // iW ∈ [0..W-1], jW ∈ [0..H-1], kW ∈ [0..D]
    int iW = floorf(rx - 0.5f);
    int jW = floorf(ry - 0.5f);
    int kW = floorf(rz);
    iW = min(max(iW, 0),     W - 1);
    jW = min(max(jW, 0),     H - 1);
    kW = min(max(kW, 0),     D);

    float fxW = (rx - 0.5f) - iW;
    float fyW = (ry - 0.5f) - jW;
    float fzW = rz - kW;
    fxW = fminf(fmaxf(fxW, 0.0f), 1.0f);
    fyW = fminf(fmaxf(fyW, 0.0f), 1.0f);
    fzW = fminf(fmaxf(fzW, 0.0f), 1.0f);

    // idx_w(i,j,k) = i + j*W + k*(W*H)
    int baseW = jW * W + kW * (W * H);
    float w000 = w[ iW     + baseW ];
    float w100 = w[(iW + 1) + baseW ];
    float w010 = w[ iW     + (jW + 1)*W + kW*(W*H) ];
    float w110 = w[(iW + 1) + (jW + 1)*W + kW*(W*H) ];
    float w001 = w[ iW     + jW*W + (kW + 1)*(W*H) ];
    float w101 = w[(iW + 1) + jW*W + (kW + 1)*(W*H) ];
    float w011 = w[ iW     + (jW + 1)*W + (kW + 1)*(W*H) ];
    float w111 = w[(iW + 1) + (jW + 1)*W + (kW + 1)*(W*H) ];
    float wInterp = trilinearInterpolation(
            fxW, fyW, fzW,
            w000, w100, w001, w101,
            w010, w110, w011, w111
    );

    return make_float3(uInterp, vInterp, wInterp);
}

// ----------------------------------
// 2) Исправленный функтор явной адвекции (Runge-Kutta / Heun не нужен — используем адаптивный Эйлер)
__device__
bool projectParticleDevice3D(Utility::Particle3D &particle,
                             const int* labels,
                             int W, int H, int D, float dx)
{
    // 26 соседей
    const int off[34][3] = {
            { 1, 0, 0}, {-1, 0, 0},
            { 0, 1, 0}, { 0,-1, 0},
            { 0, 0, 1}, { 0, 0,-1},
            {1, 1, 0}, {-1, 1, 0},
            {1, -1, 0}, {-1, -1},
            {1, 0, 1}, {-1, 0, 1},
            {1, 0, -1}, {-1, 0, -1},
            {0, 1, 1}, {0, -1, 1},
            {0, 1, -1}, {0, -1, -1},
            {1, 1, 1}, {-1, 1, 1},
            {1, -1, 1}, {1, 1, -1},
            {-1, -1, 1}, {-1, 1, -1},
            {1, -1, -1}, {-1, -1, -1},
            {-2, -1, 0},{0,-1,-2},
            {-2,-2,0}, {0,-2,-2},
            {-2,-2,-2},{-2,0,-2},
            {2,-1,0},   {0,-1,2}
    };

    // Текущая клетка
    int cx = int(floorf(particle.pos.x / dx));
    int cy = int(floorf(particle.pos.y / dx));
    int cz = int(floorf(particle.pos.z / dx));

    float3 bestPos = particle.pos;
    float  bestD   = 1e10f;
    bool   found   = false;
    int foundNeigInd = 0;
    //ищем наименьшее расстояние до твёрдой ячейки, не нашли, тогда до ближайшей воздушной
    // Сначала пытаемся найти соседнюю клетку со статусом FLUID, потом AIR
    for (int pass = 0; pass < 1; ++pass) {
        int wanted = (pass == 0 ? Utility::FLUID : Utility::AIR);
        for (int n = 0; n < 34; ++n) {
            int nx = cx + off[n][0];
            int ny = cy + off[n][1];
            int nz = cz + off[n][2];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D) continue;
            int idx = nx + ny*W + nz*W*H;
            if (labels[idx] != wanted) continue;

            float3 cellC = make_float3(
                    (nx + 0.5f) * dx,
                    (ny + 0.5f) * dx,
                    (nz + 0.5f) * dx
            );  //координаты центра ячейки
            float d = (cellC.x - particle.pos.x) * (cellC.x - particle.pos.x)
                      + (cellC.y - particle.pos.y) * (cellC.y - particle.pos.y)
                      + (cellC.z - particle.pos.z) * (cellC.z - particle.pos.z); //квадрат расстояния от положения частицы до центра ячейки
            if (d < bestD) {
                bestD   = d;
                bestPos = cellC;
                found   = true;
                foundNeigInd = n;
            }
        }
        if (found) break;
    }
    if (!found) return false;

    particle.pos = particle.pos + make_float3(static_cast<float>(off[foundNeigInd][0]),static_cast<float>(off[foundNeigInd][1]),static_cast<float>(off[foundNeigInd][2]))* 0.5f * dx;
//    // Переносим частицу на центр найденной соседней клетки
//    particle.pos = bestPos;
//
//    // во избежание накопления частиц в центрах ячеек:
//    thrust::default_random_engine randEng;
//    thrust::uniform_real_distribution<float> uniDist(-1.0f, 1.0f);
//    randEng.discard(foundNeigInd);
//    particle.pos.x += uniDist(randEng) * 0.05f * dx;
//    particle.pos.y += uniDist(randEng) * 0.05f * dx;
//    particle.pos.z += uniDist(randEng) * 0.05f * dx;
    return true;
}

struct AdvectParticlesFunctor {
    float dt, dx, C;
    int W, H, D;
    const float* u;
    const float* v;
    const float* w;
    const int*   labels;

    __host__ __device__
    AdvectParticlesFunctor(float _dt, float _dx, float _C,
                           int _W, int _H, int _D,
                           const float* _u, const float* _v, const float* _w,
                           const int*   _labels)
            : dt(_dt), dx(_dx), C(_C),
              W(_W), H(_H), D(_D),
              u(_u), v(_v), w(_w),
              labels(_labels) {}

    __device__
    Utility::Particle3D operator()(const Utility::Particle3D& pin) const {
        Utility::Particle3D particle = pin;
        float subT = 0.0f; //локальный отсчёт времени (глобальный шаг dt дробим на шаги dT)
        bool  finished = false;
        int   iter = 0;

        while (!finished && iter++ < 1000) {
            // 1) Интерполируем скорость из MAC-поля
            float3 vel = interpVelDevice3D(u, v, w, W, H, D, dx, particle.pos);

            // 2) Рассчитываем шаг dT по CFL-критерию
            //" It has been suggested[FF01] that an appropriate strategy is to limit dT so that the furthest a particle trajectory is traced is five grid cell widths:
            float speed = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z) +1e-37f + sqrtf(5.0f * dx * 9.8f);
            float dT = (C * dx) / speed; //шаг по времени находим из критерия Куранта
            if (subT + dT >= dt) {
                dT = dt - subT;
                finished = true;
            } else if (subT + 2*dT >= dt) {
                dT *= 0.5f; // делим пополам, чтобы не выйти за dt
            }

            // 3) Явный Эйлер (возможно, стоит поменять на RK3, как это советует R. Bridson...)
//            particle.pos.x += vel.x * dT;
//            particle.pos.y += vel.y * dT;
//            particle.pos.z += vel.z * dT;

            //3) RK2
//            float3 midPos = particle.pos + vel * dT * 0.5f;
//            float3 midVel = interpVelDevice3D(u, v, w, W, H, D, dx, midPos);
//            particle.pos = particle.pos + midVel * dT;
            //3)RK3
            float3 k1 = vel;
            float3 k2 = interpVelDevice3D(u, v, w, W, H, D, dx, particle.pos + k1 * 0.5f*dT);
            float3 k3 = interpVelDevice3D(u, v, w, W, H, D, dx, particle.pos + k2 * 0.75f*dT);
            particle.pos = particle.pos + (2.0f / 9.0f)*dT * k1 + (3.0f / 9.0f)*dT * k2 + (4.0f / 9.0f)*dT * k3;

            subT += dT;

            particle.pos.x = fmaxf(particle.pos.x, 0.0f);
            particle.pos.x = fminf(particle.pos.x, (W-1)*dx);
            particle.pos.y = fmaxf(particle.pos.y, 0.0f);
            particle.pos.y = fminf(particle.pos.y, (H-1)*dx);
            particle.pos.z = fmaxf(particle.pos.z, 0.0f);
            particle.pos.z = fminf(particle.pos.z, (D-1)*dx);

            // 4) Если частица попала в SOLID-клетку, пытаемся спроецировать её в соседнюю
            int cx = int(floorf(particle.pos.x / dx));
            int cy = int(floorf(particle.pos.y / dx));
            int cz = int(floorf(particle.pos.z / dx));
            if (cx >= 0 && cx < W && cy >= 0 && cy < H && cz >= 0 && cz < D) {
                int idx = cx + cy*W + cz*W*H;
                if (labels[idx] == Utility::SOLID || labels[idx] == Utility::BODY) {//проверка на попадание в твёрдую границу
                    if (!projectParticleDevice3D(particle, labels, W, H, D, dx)){ //насильно отбрасываем в FLUID (приоритетнее) или в AIR ячейку
                        projectParticleDevice3D(particle, labels, W, H, D, dx);
                        break;
                    }

                }
            }

            // 5) Проверяем выход за нижние границы и NaN
            if (particle.pos.x < 0.0f || particle.pos.y < 0.0f || particle.pos.z < 0.0f ||
                isnan(particle.pos.x)  || isnan(particle.pos.y)  || isnan(particle.pos.z)) {
                //возможно, стоит придумать, как обрабатывать такой случай
                break;
            }
        }

        return particle;
    }
};

// ----------------------------------
// 3) Обновлённый метод FluidSolver3D::advectParticles
void FluidSolver3D::advectParticles(float C)
{
    const float* pu = u.device_ptr();
    const float* pv = v.device_ptr();
    const float* pw = w.device_ptr();
    const int*   pl = labels.device_ptr();

    thrust::transform(
            d_particles.begin(),
            d_particles.end(),
            d_particles.begin(),
            AdvectParticlesFunctor(
                    dt, dx, C,
                    gridWidth, gridHeight, gridDepth,
                    pu, pv, pw,
                    pl
            )
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "advectParticles() error: " << cudaGetErrorString(err) << std::endl;
    }
}


struct RHSCalculator3D {
    const int* labels;
    const float* u, *v, *w;
    float scale;
    int W, H, D;
    float* rhs_temp;
    float dx;              // размер ячейки
    float3 vel_com;        // линейная скорость центра масс
    float3 omega;          // угловая скорость
    float3 com;            // центр масс тела

    __device__ void operator()(int idx) const {
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        if (labels[idx] != Utility::FLUID) return;

        float div =
                u[(i+1) + j*(W+1) + k*(W+1)*H] - u[i + j*(W+1) + k*(W+1)*H] +
                v[i + (j+1)*W + k*W*(H+1)] - v[i + j*W + k*W*(H+1)] +
                w[i + j*W + (k+1)*W*H] - w[i + j*W + k*W*H];

        float rhs_val = -scale * div;

        // Вспомогательная функция для вычисления скорости тела в точке
        auto get_body_velocity = [&](float3 pos) -> float3 {
            float3 r = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
            return make_float3(
                    vel_com.x + omega.y * r.z - omega.z * r.y,
                    vel_com.y + omega.z * r.x - omega.x * r.z,
                    vel_com.z + omega.x * r.y - omega.y * r.x
            );
        };

        // Обработка границ по X
        if (i-1 >= 0) {
            int label_left = labels[idx-1];
            if (label_left == Utility::SOLID || label_left == Utility::BODY) {
                float3 face_pos = make_float3(i*dx, (j+0.5f)*dx, (k+0.5f)*dx);
                float usolid = (label_left == Utility::BODY) ?
                               get_body_velocity(face_pos).x : 0.0f;
                rhs_val -= scale * (u[i + j*(W+1) + k*(W+1)*H] - usolid);
            }
        }

        if (i < W) {
            int label_right = labels[idx+1];
            if (label_right == Utility::SOLID || label_right == Utility::BODY) {
                float3 face_pos = make_float3((i+1)*dx, (j+0.5f)*dx, (k+0.5f)*dx);
                float usolid = (label_right == Utility::BODY) ?
                               get_body_velocity(face_pos).x : 0.0f;
                rhs_val += scale * (u[(i+1) + j*(W+1) + k*(W+1)*H] - usolid);
            }
        }

        // Обработка границ по Y
        if (j-1 >= 0) {
            int label_bottom = labels[idx-W];
            if (label_bottom == Utility::SOLID || label_bottom == Utility::BODY) {
                float3 face_pos = make_float3((i+0.5f)*dx, j*dx, (k+0.5f)*dx);
                float vsolid = (label_bottom == Utility::BODY) ?
                               get_body_velocity(face_pos).y : 0.0f;
                rhs_val -= scale * (v[i + j*W + k*W*(H+1)] - vsolid);
            }
        }

        if (j < H) {
            int label_top = labels[idx+W];
            if (label_top == Utility::SOLID || label_top == Utility::BODY) {
                float3 face_pos = make_float3((i+0.5f)*dx, (j+1)*dx, (k+0.5f)*dx);
                float vsolid = (label_top == Utility::BODY) ?
                               get_body_velocity(face_pos).y : 0.0f;
                rhs_val += scale * (v[i + (j+1)*W + k*W*(H+1)] - vsolid);
            }
        }

        // Обработка границ по Z
        if (k-1 >= 0) {
            int label_back = labels[idx - W*H];
            if (label_back == Utility::SOLID || label_back == Utility::BODY) {
                float3 face_pos = make_float3((i+0.5f)*dx, (j+0.5f)*dx, k*dx);
                float wsolid = (label_back == Utility::BODY) ?
                               get_body_velocity(face_pos).z : 0.0f;
                rhs_val -= scale * (w[i + j*W + k*W*H] - wsolid);
            }
        }

        if (k < D) {
            int label_front = labels[idx + W*H];
            if (label_front == Utility::SOLID || label_front == Utility::BODY) {
                float3 face_pos = make_float3((i+0.5f)*dx, (j+0.5f)*dx, (k+1)*dx);
                float wsolid = (label_front == Utility::BODY) ?
                               get_body_velocity(face_pos).z : 0.0f;
                rhs_val += scale * (w[i + j*W + (k+1)*W*H] - wsolid);
            }
        }

        // Solid boundaries (old ver.)
//        if (i-1 >= 0 && labels[idx-1] == Utility::SOLID)
//            rhs_val -= scale * (u[i + j*(W+1) + k*(W+1)*H] - 0.0f); //change 0.0f to solid boundary vel
//        if (i < W && labels[idx+1] == Utility::SOLID)
//            rhs_val += scale * (u[(i+1) + j*(W+1) + k*(W+1)*H] - 0.0f);
//        if (j-1 >= 0 && labels[idx-W] == Utility::SOLID)
//            rhs_val -= scale * (v[i + j*W + k*W*(H+1)] - 0.0f);
//        if (j < H && labels[idx+W] == Utility::SOLID)
//            rhs_val += scale * (v[i + (j+1)*W + k*W*(H+1)] - 0.0f);
//        if (k-1 >= 0 && labels[idx-W*H] == Utility::SOLID)
//            rhs_val -= scale * (w[i + j*W + k*W*H] - 0.0f);
//        if (k < D && labels[idx+W*H] == Utility::SOLID)
//            rhs_val += scale * (w[i + j*W + (k+1)*W*H] - 0.0f);
        rhs_temp[idx] = rhs_val;

    }
};

// Копируем только FLUID ячейки
struct CopyFluidRHSFunctor {
    const int* fluidNumbers;
    const float* rhs_temp;

    __device__ float operator()(int idx) const {
        int fnum = fluidNumbers[idx];
        return (fnum >= 0) ? rhs_temp[idx] : 0.0f;
    }
};


struct FluidCellPredicate {
    const int* labels;
    const int FLUID;

    __device__
    bool operator()(int idx) const {
        return labels[idx] == FLUID;
    }
};


struct IsSelected {
    const int* flags; // Указатель на данные вектора меток

    IsSelected(const int* flags_ptr) : flags(flags_ptr) {}

    __host__ __device__
    bool operator()(const int idx) const {
        return flags[idx] == 1; // Возвращает true, если метка равна 1
    }
};

// Функтор-трансформатор (просто возвращает значение)
struct ValueTransformer {
    __host__ __device__
    float operator()(float val) const {
        return val;
    }
};

void FluidSolver3D::constructRHS(thrust::device_vector<float>& rhs, const thrust::device_vector<int>& fluidNumbers, const thrust::device_vector<int>& fluidFlags) {

    float scale = (FLUID_DENSITY * dx) / dt;
    //std::cout << "scale = " << scale << std::endl;
    thrust::device_vector<float> rhs_temp(w_x_h_x_d, 0.0f);

    thrust::for_each_n(
            thrust::device,
            thrust::counting_iterator<int>(0),
            w_x_h_x_d,
            RHSCalculator3D{
                    labels.device_ptr(),
                    u.device_ptr(),
                    v.device_ptr(),
                    w.device_ptr(),
                    scale,
                    gridWidth, gridHeight, gridDepth,
                    thrust::raw_pointer_cast(rhs_temp.data()),
                    dx,
                    body.vel,
                    body.omega,
                    body.pos
            }
    );

//    std::cout << "rhs_temp:"<<std::endl;
//    thrust::host_vector<float> rhs_temp_h = rhs_temp;
//    for(int j = 0; j < gridHeight; ++j){
//        for(int i  =0 ; i < gridWidth ; ++i){
//            std::cout << rhs_temp_h[i + j * gridWidth] << ", ";
//        }
//        std::cout << std::endl;
//    }


    const int result_size = thrust::count(fluidFlags.begin(), fluidFlags.end(), 1);
    rhs.resize(result_size);

    const int* flags_ptr = thrust::raw_pointer_cast(fluidFlags.data());
    thrust::copy_if(
            thrust::device,
            rhs_temp.begin(),
            rhs_temp.end(),
            thrust::counting_iterator<size_t>(0),
            rhs.begin(),
            IsSelected( flags_ptr)
    );

    //Вывод результата
//    thrust::host_vector<float> rhs_h = rhs;
//    //rhs = thrust::device_vector<float>{4905, 4905, 4905, -4905, -4905, -4905};
//    std::cout << "Copied values: ";
//    for (float val : rhs_h) {
//        std::cout << val << " ";
//    }
//    std::cout << std::endl;
//    std::cout << "----" << std::endl;
//    thrust::transform(
//            thrust::device,
//            thrust::counting_iterator<int>(0),
//            thrust::counting_iterator<int>(w_x_h_x_d),
//            rhs.begin(),
//            CopyFluidRHSFunctor{
//                    thrust::raw_pointer_cast(fluidNumbers.data()),
//                    thrust::raw_pointer_cast(rhs_temp.data())
//            }
//    );

    /* ДО (пример)
        Индексы:    [0]     [1]     [2]     [3]
        rhs_temp:  [1.0]  [2.0]  [3.0]  [4.0]
        labels:    [SOLID] [FLUID] [AIR] [FLUID]
     * */
    /* После (пример)
        rhs: [2.0] [4.0]  // Только FLUID-ячейки
        fluidCellsAmount = 2
     * */

}

struct MatrixBuilder3D {
    int W, H, D;
    const int* labels;        // метки ячеек
    const int* fluidNumbers;   // mapping global idx → local idx (или −1)
    int*       nnz_per_row;    // выход: сколько ненулей у строки “row”

    __device__ void operator()(int idx) const {
        // 1) перевод idx → (i,j,k)
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        // 2) работаем только для FLUID-ячейки
        if (labels[idx] != Utility::FLUID) return;

        int row = fluidNumbers[idx];   // локальный номер этой ячейки
        int count = 1;                 // учитываем диагональный элемент

        // 3) 6 соседей по x,y,z
        const int off[6][3] = {
                { 1,  0,  0}, {-1,  0,  0},
                { 0,  1,  0}, { 0, -1,  0},
                { 0,  0,  1}, { 0,  0, -1}
        };

        for (int n = 0; n < 6; ++n) {
            int ni = i + off[n][0];
            int nj = j + off[n][1];
            int nk = k + off[n][2];
            // проверяем границы
            if (ni < 0 || ni >= W ||
                nj < 0 || nj >= H ||
                nk < 0 || nk >= D) continue;

            int nidx = ni + nj * W + nk * (W * H);

            //  -- если сосед FLUID и fluidNumbers[nidx] > row,
            //     значит мы храним только “верхний треугольник” (симметрично)
            if (labels[nidx] == Utility::FLUID) {
                int nrow = fluidNumbers[nidx];
                if (nrow > row) {
                    count++;
                }
            }
            //  -- если сосед не SOLID (то есть FLUID или AIR),
            //     но мы всё равно добавляем вклад в диагональ,
            //     но не добавляем off-diagonal
            //     (поскольку AIR не даёт off-diagonal,
            //      а SOLID вообще не считается).
            //    Это справедливо, потому что
            //    для AIR → только диагональный вклад,
            //    а off-diagonal (сосед) не появляется в CSR.
        }

        nnz_per_row[row] = count;
    }
};

struct MatrixFiller3D {
    int W, H, D;
    const int* labels;
    const int* fluidNumbers;
    const int* csr_offsets;
    float* csr_values;
    int* csr_columns;
    float scale;

    __device__ void operator()(int idx) const {
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        if (labels[idx] != Utility::FLUID) return;

        int row = fluidNumbers[idx];
        int pos = csr_offsets[row];
        float diagVal = 0.0f;

        // Диагональный элемент
        csr_values[pos] = diagVal;
        csr_columns[pos] = row;
        pos++;

        // Обработка соседей
        const int offsets[6][3] = {
                {1,0,0}, {-1,0,0},
                {0,1,0}, {0,-1,0},
                {0,0,1}, {0,0,-1}
        };

        for (int n = 0; n < 6; ++n) {
            int ni = i + offsets[n][0];
            int nj = j + offsets[n][1];
            int nk = k + offsets[n][2];

            if (ni >= 0 && ni < W && nj >= 0 && nj < H && nk >= 0 && nk < D) {
                int nidx = ni + nj * W + nk * W * H;

                if (labels[nidx] == Utility::FLUID) {
                    int col = fluidNumbers[nidx];

                    // Только верхний треугольник
                    if (col > row) {
                        csr_values[pos] = -scale;
                        csr_columns[pos] = col;
                        pos++;
                    }
                    diagVal += scale;
                }
                else if (labels[nidx] == Utility::AIR) {
                    diagVal += scale;
                }
            }
        }

        // Обновляем диагональ
        csr_values[csr_offsets[row]] = diagVal;
    }
};

void FluidSolver3D::constructA(
        thrust::device_vector<float>& csr_values,
        thrust::device_vector<int>& csr_columns,
        thrust::device_vector<int>& csr_offsets,
        thrust::device_vector<int> fluidNumbers
) {
    thrust::device_vector<int> nnz_per_row(fluidCellsAmount, 0);

    // Фаза 1: Подсчет ненулевых элементов
    thrust::for_each_n(
            thrust::device,
            thrust::counting_iterator<int>(0),
            gridWidth * gridHeight * gridDepth,
            MatrixBuilder3D{
                    gridWidth, gridHeight, gridDepth,
                    thrust::raw_pointer_cast(labels.device_ptr()),
                    thrust::raw_pointer_cast(fluidNumbers.data()),
                    thrust::raw_pointer_cast(nnz_per_row.data())
            }
    );

    // Строим смещения
    thrust::exclusive_scan(
            thrust::device,
            nnz_per_row.begin(), nnz_per_row.end(),
            csr_offsets.begin()
    );

    // Общее количество ненулевых элементов
    int total_nnz = thrust::reduce(
            thrust::device,
            nnz_per_row.begin(), nnz_per_row.end()
    );

    csr_values.resize(total_nnz);
    csr_columns.resize(total_nnz);

    // Фаза 2: Заполнение матрицы
    thrust::for_each_n(
            thrust::device,
            thrust::counting_iterator<int>(0),
            gridWidth * gridHeight * gridDepth,
            MatrixFiller3D{
                    gridWidth, gridHeight, gridDepth,
                    thrust::raw_pointer_cast(labels.device_ptr()),
                    thrust::raw_pointer_cast(fluidNumbers.data()),
                    thrust::raw_pointer_cast(csr_offsets.data()),
                    thrust::raw_pointer_cast(csr_values.data()),
                    thrust::raw_pointer_cast(csr_columns.data()),
                    1.0f
            }
    );

    csr_offsets.back() = csr_values.size();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ConstructA error: " << cudaGetErrorString(err) << std::endl;
    }
}


struct GlobalToLocal {
    const int* flags;       // Указатель на вектор флагов
    const int* old_to_new;  // Указатель на вектор преобразования индексов
    const float* local_vals; // Указатель на локальные значения

    GlobalToLocal(const int* f, const int* m, const float* l)
            : flags(f), old_to_new(m), local_vals(l) {}

    __host__ __device__
    float operator()(int global_idx) const {
        if (flags[global_idx] == 1) {
            int local_idx = old_to_new[global_idx];
            return local_vals[local_idx];
        }
        return 0.0f; // Значение по умолчанию
    }
};

// Функтор для пометки не-FLUID ячеек (flags[idx]==0) значением −1 в fluidNumbers_d
struct MarkNonFluidFunctor {
    const int* flags;      // указатель на массив flags
    int*       fluidNums;  // указатель на массив fluidNumbers_d

    MarkNonFluidFunctor(const int* _flags, int* _fluidNums)
            : flags(_flags), fluidNums(_fluidNums) {}

    __host__ __device__
    void operator()(int idx) const {
        if (flags[idx] == 0) {
            fluidNums[idx] = -1;
        }
    }
};


int FluidSolver3D::pressureSolve() {

    //w_x_h_x_d = gridWidth * gridHeight * gridDepth;

    // новая нумерация
    thrust::device_vector<int> fluidNumbers_d(w_x_h_x_d, -1);
    thrust::sequence(thrust::device, fluidNumbers_d.begin(), fluidNumbers_d.end()); // последовательность индексов от 0
    thrust::device_vector<int> flags(w_x_h_x_d, 0);

    /*ЧТО ХОТИМ СДЕЛАТЬ НИЖЕ: ВВЕСТИ НОВУЮ НУМЕРАЦИЮ.
     * Ячейка	Метка	flags	fluidNumbers_d
       (0,0,0)	FLUID	 1	     0
       (1,0,0)	SOLID	 0	     1
       (2,0,0)	FLUID	 1	     1
     * */
    thrust::transform(
            thrust::device,
            labels.device_ptr(),
            labels.device_ptr() + w_x_h_x_d,
            flags.begin(),
            FluidFlagFunctor()
    );

//    std::cout << "flags:" << std::endl;
//    thrust::host_vector<float> flags_h = flags;
//    for(int j = 0; j < gridHeight; ++j){
//        for(int i = 0; i < gridWidth; ++i){
//            std::cout << flags_h[i + j*gridWidth] << ", ";
//        }
//        std::cout << std::endl;
//    }


    // с помощью префиксной суммы (не включая текущий жлемент, exclusive) получаем индексы жидких ячеек в новой нумерации (флаги нужны для получения таких сумм)
    thrust::exclusive_scan(
            thrust::device,
            flags.begin(), flags.end(),
            fluidNumbers_d.begin()
    );
    // После этого нужно “записать -1” для тех клеток, где flags[idx]==0:
//    thrust::for_each(
//            thrust::device,
//            thrust::make_counting_iterator<int>(0),
//            thrust::make_counting_iterator<int>(w_x_h_x_d),
//            MarkNonFluidFunctor(
//                    thrust::raw_pointer_cast(flags.data()),
//                    thrust::raw_pointer_cast(fluidNumbers_d.data())
//            )
//    );
    //std::cout << "last number = " << fluidNumbers_d[w_x_h_x_d-1] << std::endl;
    //fluidCellsAmount = fluidNumbers_d[w_x_h_x_d-1];
    //  Подсчет количества жидких ячеек
    fluidCellsAmount = thrust::reduce(
            thrust::device,
            flags.begin(),
            flags.end(),
            0,
            thrust::plus<int>()
    );
//    std::cout << "fluidCelssAmount = " << fluidCellsAmount << std::endl;
    if (fluidCellsAmount == 0) {
        std::cerr << "No fluid cells found!" << std::endl;
        return -1;
    }

    //  Построение правой части (RHS)
    thrust::device_vector<float> rhs_d;
    constructRHS(rhs_d, fluidNumbers_d, flags);

//    std::cout << "fluidNumbers:" << std::endl;
//    thrust::host_vector<float> fluidNumbers = fluidNumbers_d;
//    for(int j = 0; j < gridHeight; ++j){
//        for(int i = 0; i < gridWidth; ++i){
//            std::cout << fluidNumbers[i + j*gridWidth] << ", ";
//        }
//        std::cout << std::endl;
//    }
//
//    std::cout << "rhs for new system:" << std::endl;
//    thrust::host_vector<float> rhs = rhs_d;
//    for(int j = 0; j < gridHeight; ++j){
//        for(int i = 0; i < gridWidth; ++i){
//            if(i + j*gridWidth < rhs.size())
//                std::cout << rhs[i + j*gridWidth] << ", ";
//        }
//        std::cout << std::endl;
//    }

    //  Построение матрицы A в формате CSR
    thrust::device_vector<float> csr_values;
    thrust::device_vector<int> csr_columns;
    thrust::device_vector<int> csr_offsets(fluidCellsAmount + 1, 0);

    constructA(csr_values, csr_columns, csr_offsets, fluidNumbers_d);

    if (csr_values.empty()) {
        std::cerr << "Matrix construction failed!" << std::endl;
        return -2;
    }

    //cudaStream_t stream = NULL;
    //cudaStreamCreate(&stream);
    //cudssHandle_t handle;
    //cudssStatus_t status = cudssCreate(&handle);
    //cudssSetStream(handle, stream);

    //cudssConfig_t solverConfig;
    //cudssData_t solverData;
    //cudssConfigCreate(&solverConfig);
    //cudssDataCreate(handle, &solverData);

//    if (status != CUDSS_STATUS_SUCCESS) {
//        std::cerr << "cuDSS init failed: " << status << std::endl;
//        return -3;
//    }
    cudssStatus_t status;
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;// Symmetric Positive Definite
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;// Upper triangular stored
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    int nnz = csr_values.size();

    status = cudssMatrixCreateCsr(
            &A,
            fluidCellsAmount, fluidCellsAmount, nnz,
            thrust::raw_pointer_cast(csr_offsets.data()),
            NULL,
            thrust::raw_pointer_cast(csr_columns.data()),
            thrust::raw_pointer_cast(csr_values.data()),
            CUDA_R_32I, CUDA_R_32F,
            mtype, mview, base
    );

    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "Matrix creation failed: " << status << std::endl;
        cudssDestroy(handle);
        return -4;
    }

//    thrust::host_vector<float> csr_vals_h = csr_values;
//    thrust::host_vector<float> csr_cols_h = csr_columns;
//    thrust::host_vector<float> csr_offs_h = csr_offsets;

//    for(int i = 0; i < csr_vals_h.size(); ++i){
//        std::cout << csr_vals_h[i] << ", ";
//    }
//    std::cout << "\n------\n";
//    for(int i = 0; i < csr_cols_h.size(); ++i){
//        std::cout << csr_cols_h[i] << ", ";
//    }
//    std::cout << "\n------\n";
//    for(int i = 0; i < csr_offs_h.size(); ++i){
//        std::cout << csr_offs_h[i] << ", ";
//    }
//    std::cout << "\n------\n";
//    std::cin.get();

    // решение Системы линейных алгебраических уравнений с разреженной матрицей
    thrust::device_vector<float> solution(fluidCellsAmount);

    cudssMatrix_t x, b;

    status = cudssMatrixCreateDn(
            &b, fluidCellsAmount, 1, fluidCellsAmount,
            thrust::raw_pointer_cast(rhs_d.data()),
            CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR
    );

    status = cudssMatrixCreateDn(
            &x, fluidCellsAmount, 1, fluidCellsAmount,
            thrust::raw_pointer_cast(solution.data()),
            CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR
    );

    // Решение системы
    // Анализ (async func)
    status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS,
                          solverConfig, solverData, A, x, b);
    cudaStreamSynchronize(stream);
    // Факторизация (async func)
    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,
                          solverConfig, solverData, A, x, b);
    cudaStreamSynchronize(stream);
    // Решение (async func)
    status = cudssExecute(handle, CUDSS_PHASE_SOLVE,
                          solverConfig, solverData, A, x, b);
    cudaStreamSynchronize(stream);

    status = cudssMatrixDestroy(A);
    status = cudssMatrixDestroy(b);
    status = cudssMatrixDestroy(x);
    //cudssDataDestroy(handle, solverData);
    //cudssConfigDestroy(solverConfig);
    //cudssDestroy(handle);


//    std::cout << "----solution local---" << std::endl;
//    thrust::host_vector<float> sol_h = solution;
//    for(int k = 0; k < sol_h.size(); ++k){
//        std::cout << sol_h[k] << ", ";
//    }
//    std::cout << std::endl;

    //  Копирование решения в сетку давления
    thrust::device_vector<float> p_temp(w_x_h_x_d, 0.0f);
//    thrust::transform(
//            thrust::device,
//            fluidNumbers_d.begin(), fluidNumbers_d.end(),
//            solution.begin(),
//            p_temp.begin(),
//            CopySolutionFunctor()
//    );
    GlobalToLocal transformer(
            thrust::raw_pointer_cast(flags.data()),
            thrust::raw_pointer_cast(fluidNumbers_d.data()),
            thrust::raw_pointer_cast(solution.data())
    );
    thrust::transform(
            thrust::make_counting_iterator(0),      // Итератор глобальных индексов: 0,1,2,...
            thrust::make_counting_iterator(w_x_h_x_d), // Конец индексов
            p_temp.begin(),                   // Выходной итератор
            transformer                             // Функтор преобразования
    );

    thrust::copy(p_temp.begin(), p_temp.end(), p.device_data.begin());


//    std::cout << "----pressure 3d---" << std::endl;
//    p.host_data = p.device_data;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << p.host_data[i + j*gridWidth + k * gridWidth*gridHeight] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }


    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "PressureSolve() error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}

// ----------------------------------
// Утилитарная функция для преобразования из 3D-индекса (i,j,k) в линейный (cell-centered)
__device__ __host__
inline int cellIdx(int i, int j, int k, int W, int H) {
    // Ячейки размером W × H × D
    return i + j * W + k * (W * H);
}

struct UFunctor {
    float*       u;          // массив U-скоростей (face-centered по X), размер (W+1)*H*D
    const float* p;          // массив давлений на центрах ячеек, размер W*H*D
    const int*   labels;     // метки ячеек (AIR, FLUID, SOLID), размер W*H*D
    float        scale;      // = dt / (ρ * dx)
    int          W, H, D;    // размеры сетки (ячейки): W × H × D
    float        VEL_UNKNOWN;
    float        dx;         // размер ячейки
    float3       vel_com;    // скорость центра масс тела
    float3       omega;      // угловая скорость тела
    float3       com;        // центр масс тела

    UFunctor(float*       u_,
             const float* p_,
             const int*   labels_,
             float        scale_,
             int          W_,
             int          H_,
             int          D_,
             float        vel_unknown,
             float        dx_,
             float3       vel_com_,
             float3       omega_,
             float3       com_)
            : u(u_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown),
              dx(dx_), vel_com(vel_com_), omega(omega_), com(com_)  {}

    __host__ __device__
    void operator()(int idx) const {
        // Получаем трёхмерные индексы (iU, jU, kU) для u:
        // где iU ∈ [0..W], jU ∈ [0..H-1], kU ∈ [0..D-1]
        int sliceSize = (W + 1) * H;        // размер “плоскости” (iU + jU*(W+1)) для каждого kU
        int kU = idx / sliceSize;
        int rem = idx % sliceSize;
        int jU = rem / (W + 1);
        int iU = rem % (W + 1);
        int leftIdx  = cellIdx(iU - 1, jU, kU, W, H);
        int rightIdx = cellIdx(iU, jU, kU, W, H);

        //float usolid = 0.0f; // пока что границы неподвижны
        float invScale = 1.0f / scale;
        float p0 = 0.0f;
        float p1 = 0.0f;

        // Вычисляем координаты центра грани
        float3 face_pos = make_float3(
                iU * dx,
                (jU + 0.5f) * dx,
                (kU + 0.5f) * dx
        );

        // Вычисляем скорость тела в точке грани
        float3 r = make_float3(
                face_pos.x - com.x,
                face_pos.y - com.y,
                face_pos.z - com.z
        );

        float3 v_body = make_float3(
                vel_com.x + omega.y * r.z - omega.z * r.y,
                vel_com.y + omega.z * r.x - omega.x * r.z,
                vel_com.z + omega.x * r.y - omega.y * r.x
        );

        // Обработка граничных условий
        if ((labels[leftIdx] == Utility::FLUID || labels[leftIdx] == Utility::AIR) &&
            (labels[rightIdx] == Utility::FLUID || labels[rightIdx] == Utility::AIR)) {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }
        else if (labels[leftIdx] == Utility::SOLID || labels[leftIdx] == Utility::BODY) {
            float usolid = (labels[leftIdx] == Utility::BODY) ? v_body.x : 0.0f;
            p0 = p[rightIdx] - invScale * (u[idx] - usolid);
            p1 = p[rightIdx];
        }
        else if (labels[rightIdx] == Utility::SOLID || labels[rightIdx] == Utility::BODY) {
            float usolid = (labels[rightIdx] == Utility::BODY) ? v_body.x : 0.0f;
            p0 = p[leftIdx];
            p1 = p[leftIdx] + invScale * (u[idx] - usolid);
        }
        else {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }

        u[idx] = u[idx] - scale * (p1 - p0);
    }
};

// ----------------------------------
// Функтор для коррекции V-скоростей (MAC-грань по Y)
// vSize = W × (H+1) × D
struct VFunctor {
    float*       v;
    const float* p;
    const int*   labels;
    float        scale;
    int          W, H, D;
    float        VEL_UNKNOWN;
    float        dx;         // размер ячейки
    float3       vel_com;    // скорость центра масс тела
    float3       omega;      // угловая скорость тела
    float3       com;        // центр масс тела

    VFunctor(float*       v_,
             const float* p_,
             const int*   labels_,
             float        scale_,
             int          W_,
             int          H_,
             int          D_,
             float        vel_unknown,
             float        dx_,
             float3       vel_com_,
             float3       omega_,
             float3       com_)
            : v(v_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown),
              dx(dx_), vel_com(vel_com_), omega(omega_), com(com_) {}

    __host__ __device__
    void operator()(int idx) const {
        // iV ∈ [0..W-1], jV ∈ [0..H], kV ∈ [0..D-1]
        int sliceSize = W * (H + 1);
        int kV = idx / sliceSize;
        int rem = idx % sliceSize;
        int jV = rem / W;
        int iV = rem % W;
        int leftIdx  = cellIdx(iV, jV - 1, kV, W, H);
        int rightIdx = cellIdx(iV, jV, kV, W, H);

        //float vsolid = 0.0f; // пока что границы неподвижны
        float invScale = 1.0f / scale;
        float p0 = 0.0f;
        float p1 = 0.0f;

        float3 face_pos = make_float3(
                (iV + 0.5f) * dx,
                jV * dx,
                (kV + 0.5f) * dx
        );
        float3 r = make_float3(
                face_pos.x - com.x,
                face_pos.y - com.y,
                face_pos.z - com.z
        );
        float3 v_body = make_float3(
                vel_com.x + omega.y * r.z - omega.z * r.y,
                vel_com.y + omega.z * r.x - omega.x * r.z,
                vel_com.z + omega.x * r.y - omega.y * r.x
        );

        // Обработка граничных условий
        if ((labels[leftIdx] == Utility::FLUID || labels[leftIdx] == Utility::AIR) &&
            (labels[rightIdx] == Utility::FLUID || labels[rightIdx] == Utility::AIR)) {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }
        else if (labels[leftIdx] == Utility::SOLID || labels[leftIdx] == Utility::BODY) {
            float vsolid = (labels[leftIdx] == Utility::BODY) ? v_body.y : 0.0f;
            p0 = p[rightIdx] - invScale * (v[idx] - vsolid);
            p1 = p[rightIdx];
        }
        else if (labels[rightIdx] == Utility::SOLID || labels[rightIdx] == Utility::BODY) {
            float vsolid = (labels[rightIdx] == Utility::BODY) ? v_body.y : 0.0f;
            p0 = p[leftIdx];
            p1 = p[leftIdx] + invScale * (v[idx] - vsolid);
        }
        else {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }
        v[idx] = v[idx] - scale * (p1 - p0);
    }
};

// ----------------------------------
// Функтор для коррекции W-скоростей (MAC-грань по Z)
// wSize = W × H × (D+1)
struct WFunctor {
    float*       w;
    const float* p;
    const int*   labels;
    float        scale;
    int          W, H, D;
    float        VEL_UNKNOWN;
    float        dx;         // размер ячейки
    float3       vel_com;    // скорость центра масс тела
    float3       omega;      // угловая скорость тела
    float3       com;        // центр масс тела

    WFunctor(float*       w_,
             const float* p_,
             const int*   labels_,
             float        scale_,
             int          W_,
             int          H_,
             int          D_,
             float        vel_unknown,
             float        dx_,
             float3       vel_com_,
             float3       omega_,
             float3       com_)
            : w(w_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown),
              dx(dx_), vel_com(vel_com_), omega(omega_), com(com_) {}

    __host__ __device__
    void operator()(int idx) const {
        // iW ∈ [0..W-1], jW ∈ [0..H-1], kW ∈ [0..D]
        int sliceSize = W * H;
        int kW = idx / sliceSize;
        int rem = idx % sliceSize;
        int jW = rem / W;
        int iW = rem % W;
        int leftIdx  = cellIdx(iW, jW, kW - 1, W, H);
        int rightIdx = cellIdx(iW, jW, kW, W, H);

        //float wsolid = 0.0f; // пока что границы неподвижны
        float invScale = 1.0f / scale;
        float p0 = 0.0f;
        float p1 = 0.0f;

        float3 face_pos = make_float3(
                (iW + 0.5f) * dx,
                (jW + 0.5f) * dx,
                kW * dx
        );
        float3 r = make_float3(
                face_pos.x - com.x,
                face_pos.y - com.y,
                face_pos.z - com.z
        );
        float3 v_body = make_float3(
                vel_com.x + omega.y * r.z - omega.z * r.y,
                vel_com.y + omega.z * r.x - omega.x * r.z,
                vel_com.z + omega.x * r.y - omega.y * r.x
        );

        // Обработка граничных условий
        if ((labels[leftIdx] == Utility::FLUID || labels[leftIdx] == Utility::AIR) &&
            (labels[rightIdx] == Utility::FLUID || labels[rightIdx] == Utility::AIR)) {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }
        else if (labels[leftIdx] == Utility::SOLID || labels[leftIdx] == Utility::BODY) {
            float wsolid = (labels[leftIdx] == Utility::BODY) ? v_body.z : 0.0f;
            p0 = p[rightIdx] - invScale * (w[idx] - wsolid);
            p1 = p[rightIdx];
        }
        else if (labels[rightIdx] == Utility::SOLID || labels[rightIdx] == Utility::BODY) {
            float wsolid = (labels[rightIdx] == Utility::BODY) ? v_body.z : 0.0f;
            p0 = p[leftIdx];
            p1 = p[leftIdx] + invScale * (w[idx] - wsolid);
        }
        else {
            p0 = p[leftIdx];
            p1 = p[rightIdx];
        }

        w[idx] = w[idx] - scale * (p1 - p0);

    }
};

// ----------------------------------
// 4) Обновлённый метод FluidSolver3D::applyPressure()
void FluidSolver3D::applyPressure() {
    float scale        = dt / (FLUID_DENSITY * dx);
    float vel_unknown  = static_cast<float>(VEL_UNKNOWN);

//    std::cout << "----pressure 3d---" << std::endl;
//    p.host_data = p.device_data;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << p.host_data[i + j*gridWidth + k * gridWidth*gridHeight] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout <<"\n"<< std::endl;
//    }

    // --- Коррекция U-скоростей ---
    int u_size = (gridWidth + 1) * gridHeight * gridDepth;
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(u_size),
            UFunctor(
                    u.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth,
                    gridHeight,
                    gridDepth,
                    vel_unknown,
                    dx,
                    body.vel,
                    body.omega,
                    body.pos
            )
    );

//        u.copy_to_host();
//        std::cout << "----u 3d (after pressure apply)---" << std::endl;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth+1; ++i){
//                std::cout << u.host_data[i + j*(gridWidth+1) + k * (gridWidth+1)*gridHeight] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }


    // --- Коррекция V-скоростей ---
    int v_size = gridWidth * (gridHeight + 1) * gridDepth;
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(v_size),
            VFunctor(
                    v.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth,
                    gridHeight,
                    gridDepth,
                    vel_unknown,
                    dx,
                    body.vel,
                    body.omega,
                    body.pos
            )
    );

//        v.copy_to_host();
//    std::cout << "----v 3d (after pressure apply)---" << std::endl;
//    for(int k = 0; k < gridDepth; ++k){
//        for(int j = 0; j < gridHeight+1; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << v.host_data[i + j*(gridWidth) + k * (gridWidth)*(gridHeight+1)] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }

    // --- Коррекция W-скоростей ---
    int w_size = gridWidth * gridHeight * (gridDepth + 1);
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(w_size),
            WFunctor(
                    w.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth,
                    gridHeight,
                    gridDepth,
                    vel_unknown,
                    dx,
                    body.vel,
                    body.omega,
                    body.pos
            )
    );

//        w.copy_to_host();
//    std::cout << "----w 3d (after pressure apply)---" << std::endl;
//    for(int k = 0; k < gridDepth+1; ++k){
//        for(int j = 0; j < gridHeight; ++j){
//            for(int i = 0; i < gridWidth; ++i){
//                std::cout << w.host_data[i + j*(gridWidth) + k * (gridWidth)*(gridHeight)] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }

    // Проверка ошибок CUDA
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "applyPressure() error: " << cudaGetErrorString(err) << std::endl;
    }
}

//####################################################################
// RIGID BODY FUNCS
//#############################################
// 1) Функтор для накопления локальных реакций по U‑узлам
struct AccumulateBodyForcesU {
    int W, H, D;
    float dx, dt, rho;
    float3 cm;             // центр масс тела
    const int* labels;     // метки ячеек (W×H×D)
    const float* uStar;    // предварительная скорость U⋆
    const float* uNew;     // скорректированная скорость uⁿ⁺¹
    float3* forceAcc;      // аккумулятор силы
    float3* torqueAcc;     // аккумулятор момента

    __device__
    void operator()(int idx) const {
        int i = idx % (W+1);
        int tmp = idx / (W+1);
        int j = tmp % H;
        int k = tmp / H;

        // Определяем соседние ячейки
        int cellL = (i > 0) ? (i-1) + j*W + k*W*H : -1;
        int cellR = (i < W) ? i + j*W + k*W*H : -1;

        // Проверяем, находится ли узел на границе с телом
        bool leftIsFluid = (cellL >= 0) && (labels[cellL] == Utility::FLUID);
        bool rightIsBody = (cellR >= 0) && (labels[cellR] == Utility::BODY);
        bool leftIsBody = (cellL >= 0) && (labels[cellL] == Utility::BODY);
        bool rightIsFluid = (cellR >= 0) && (labels[cellR] == Utility::FLUID);

        // Только границы жидкость-тело
        if (!((leftIsFluid && rightIsBody) || (leftIsBody && rightIsFluid)))
            return;

        // Направление силы (нормаль к грани)
        float3 normal;
        float sign = 1.0f;
        if (leftIsFluid && rightIsBody) {
            normal = make_float3(1.0f, 0.0f, 0.0f); // Нормаль вправо
        } else {
            normal = make_float3(-1.0f, 0.0f, 0.0f); // Нормаль влево
            sign = -1.0f;
        }

        // Координата грани
        float3 X = make_float3(i*dx, (j+0.5f)*dx, (k+0.5f)*dx);
        float3 r = X - cm;  // Вектор от центра масс к точке приложения силы

        // Разница скоростей
        float deltaU = uNew[idx] - uStar[idx];

        // Площадь грани
        float area = dx * dx;

        // Сила реакции (F = -ρ * A * Δu)
        float3 Fi = normal * (-rho * area * deltaU / dt);

        // Момент: τ = r × F
        float3 torque = Utility::cross(r, Fi);

        // Атомарное добавление
        atomicAdd(&forceAcc->x, Fi.x);
        atomicAdd(&forceAcc->y, Fi.y);
        atomicAdd(&forceAcc->z, Fi.z);

        atomicAdd(&torqueAcc->x, torque.x);
        atomicAdd(&torqueAcc->y, torque.y);
        atomicAdd(&torqueAcc->z, torque.z);
    }
};

// ------------------------------------------
// Аналогичные функторы для V‑узлов и W‑узлов:
//   AccumulateBodyForcesV  собирает Fi = (0, −m*Δv, 0); позицию (i+0.5,j,k+0.5)
struct AccumulateBodyForcesV {
    int W, H, D;
    float dx, dt, rho;
    float3 cm;             // центр масс тела
    const int* labels;     // метки ячеек (W×H×D)
    const float* vStar;    // предварительная скорость V⋆
    const float* vNew;     // скорректированная скорость vⁿ⁺¹
    float3* forceAcc;      // аккумулятор силы
    float3* torqueAcc;     // аккумулятор момента

    __device__
    void operator()(int idx) const {
        int i = idx % W;
        int tmp = idx / W;
        int j = tmp % (H+1);
        int k = tmp / (H+1);

        // Определяем соседние ячейки
        int cellB = (j > 0) ? i + (j-1)*W + k*W*H : -1;
        int cellT = (j < H) ? i + j*W + k*W*H : -1;

        // Проверяем, находится ли узел на границе с телом
        bool bottomIsFluid = (cellB >= 0) && (labels[cellB] == Utility::FLUID);
        bool topIsBody = (cellT >= 0) && (labels[cellT] == Utility::BODY);
        bool bottomIsBody = (cellB >= 0) && (labels[cellB] == Utility::BODY);
        bool topIsFluid = (cellT >= 0) && (labels[cellT] == Utility::FLUID);

        // Только границы жидкость-тело
        if (!((bottomIsFluid && topIsBody) || (bottomIsBody && topIsFluid)))
            return;

        // Направление силы (нормаль к грани)
        float3 normal;
        if (bottomIsFluid && topIsBody) {
            normal = make_float3(0.0f, 1.0f, 0.0f); // Нормаль вверх
        } else {
            normal = make_float3(0.0f, -1.0f, 0.0f); // Нормаль вниз
        }

        // Координата грани
        float3 X = make_float3((i+0.5f)*dx, j*dx, (k+0.5f)*dx);
        float3 r = X - cm;  // Вектор от центра масс к точке приложения силы

        // Разница скоростей
        float deltaV = vNew[idx] - vStar[idx];

        // Площадь грани
        float area = dx * dx;

        // Сила реакции (F = -ρ * A * Δv)
        float3 Fi = normal * (-rho * area * deltaV / dt);

        // Момент: τ = r × F
        float3 torque = Utility::cross(r, Fi);

        // Атомарное добавление
        atomicAdd(&forceAcc->x, Fi.x);
        atomicAdd(&forceAcc->y, Fi.y);
        atomicAdd(&forceAcc->z, Fi.z);

        atomicAdd(&torqueAcc->x, torque.x);
        atomicAdd(&torqueAcc->y, torque.y);
        atomicAdd(&torqueAcc->z, torque.z);
    }
};

//   AccumulateBodyForcesW  собирает Fi = (0,0, −m*Δw); позицию (i+0.5,j+0.5,k)
struct AccumulateBodyForcesW {
    int W, H, D;
    float dx, dt, rho;
    float3 cm;             // центр масс тела
    const int* labels;     // метки ячеек (W×H×D)
    const float* wStar;    // предварительная скорость W⋆
    const float* wNew;     // скорректированная скорость wⁿ⁺¹
    float3* forceAcc;      // аккумулятор силы
    float3* torqueAcc;     // аккумулятор момента

    __device__
    void operator()(int idx) const {
        int i = idx % W;
        int tmp = idx / W;
        int j = tmp % H;
        int k = tmp / H;

        // Определяем соседние ячейки
        int cellBack = (k > 0) ? i + j*W + (k-1)*W*H : -1;
        int cellFront = (k < D) ? i + j*W + k*W*H : -1;

        // Проверяем, находится ли узел на границе с телом
        bool backIsFluid = (cellBack >= 0) && (labels[cellBack] == Utility::FLUID);
        bool frontIsBody = (cellFront >= 0) && (labels[cellFront] == Utility::BODY);
        bool backIsBody = (cellBack >= 0) && (labels[cellBack] == Utility::BODY);
        bool frontIsFluid = (cellFront >= 0) && (labels[cellFront] == Utility::FLUID);

        // Только границы жидкость-тело
        if (!((backIsFluid && frontIsBody) || (backIsBody && frontIsFluid)))
            return;

        // Направление силы (нормаль к грани)
        float3 normal;
        if (backIsFluid && frontIsBody) {
            normal = make_float3(0.0f, 0.0f, 1.0f); // Нормаль вперед
        } else {
            normal = make_float3(0.0f, 0.0f, -1.0f); // Нормаль назад
        }

        // Координата грани
        float3 X = make_float3((i+0.5f)*dx, (j+0.5f)*dx, k*dx);
        float3 r = X - cm;  // Вектор от центра масс к точке приложения силы

        // Разница скоростей
        float deltaW = wNew[idx] - wStar[idx];

        // Площадь грани
        float area = dx * dx;

        // Сила реакции (F = -ρ * A * Δw)
        float3 Fi = normal * (-rho * area * deltaW / dt);

        // Момент: τ = r × F
        float3 torque = Utility::cross(r, Fi);

        // Атомарное добавление
        atomicAdd(&forceAcc->x, Fi.x);
        atomicAdd(&forceAcc->y, Fi.y);
        atomicAdd(&forceAcc->z, Fi.z);

        atomicAdd(&torqueAcc->x, torque.x);
        atomicAdd(&torqueAcc->y, torque.y);
        atomicAdd(&torqueAcc->z, torque.z);
    }
};

struct PressureForceCalculator {
    int W, H, D;
    float dx;
    float3 cm;             // центр масс тела
    const int* labels;     // метки ячеек (W×H×D)
    const float* pressure; // давление в ячейках
    float3* forceAcc;      // аккумулятор силы
    float3* torqueAcc;     // аккумулятор момента

    __device__
    void operator()(int idx) const {
        // Преобразуем линейный индекс в 3D координаты
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        // Работаем только с ячейками тела
        if (labels[idx] != Utility::BODY) return;

        // Площадь грани
        float area = dx * dx;

        // Смещения для соседей (право, лево, верх, низ, перед, зад)
        int di[6] = {1, -1, 0, 0, 0, 0};
        int dj[6] = {0, 0, 1, -1, 0, 0};
        int dk[6] = {0, 0, 0, 0, 1, -1};

        // Нормали для каждой грани
        float3 normals[6] = {
                {1.0f, 0.0f, 0.0f},  // right
                {-1.0f, 0.0f, 0.0f}, // left
                {0.0f, 1.0f, 0.0f},  // top
                {0.0f, -1.0f, 0.0f}, // bottom
                {0.0f, 0.0f, 1.0f},  // front
                {0.0f, 0.0f, -1.0f}  // back
        };

        // Центры граней относительно центра ячейки
        float3 face_offsets[6] = {
                {0.5f, 0.0f, 0.0f},  // right
                {-0.5f, 0.0f, 0.0f}, // left
                {0.0f, 0.5f, 0.0f},  // top
                {0.0f, -0.5f, 0.0f}, // bottom
                {0.0f, 0.0f, 0.5f},  // front
                {0.0f, 0.0f, -0.5f}  // back
        };

        for (int dir = 0; dir < 6; dir++) {
            int ni = i + di[dir];
            int nj = j + dj[dir];
            int nk = k + dk[dir];

            // Пропускаем, если сосед выходит за границы
            if (ni < 0 || ni >= W || nj < 0 || nj >= H || nk < 0 || nk >= D)
                continue;

            int nidx = ni + nj * W + nk * W * H;

            // Работаем только с границами тело-жидкость
            if (labels[nidx] != Utility::FLUID)
                continue;

            // Вычисляем центр грани в мировых координатах
            float3 cell_center = make_float3(
                    (i + 0.5f) * dx,
                    (j + 0.5f) * dx,
                    (k + 0.5f) * dx
            );

            float3 face_center = cell_center + face_offsets[dir] * dx;

            // Вектор от центра масс к центру грани
            float3 r = face_center - cm;

            // Сила давления (F = -p * A * n)
            float p = pressure[nidx];
            float3 F = normals[dir] * (-p * area);

            // Момент (τ = r × F)
            float3 torque = Utility::cross(r, F);

            // Атомарное добавление
            atomicAdd(&forceAcc->x, F.x);
            atomicAdd(&forceAcc->y, F.y);
            atomicAdd(&forceAcc->z, F.z);

            atomicAdd(&torqueAcc->x, torque.x);
            atomicAdd(&torqueAcc->y, torque.y);
            atomicAdd(&torqueAcc->z, torque.z);
        }
    }
};
void FluidSolver3D::updateBody() {
    int Usize = (gridWidth+1)*gridHeight*gridDepth;
    int Vsize = gridWidth*(gridHeight+1)*gridDepth;
    int Wsize = gridWidth*gridHeight*(gridDepth+1);

    // 1) Сохраняем u⋆ до applyPressure
    thrust::device_vector<float> uStar(uSaved.device_data);
    thrust::device_vector<float> vStar(vSaved.device_data);
    thrust::device_vector<float> wStar(wSaved.device_data);


    // 2) Заведём аккумуляторы (на device)
    thrust::device_vector<float3> forceAcc(1, make_float3(0.0f,0.0f,0.0f));
    thrust::device_vector<float3> torqueAcc(1, make_float3(0.0f,0.0f,0.0f));

    //float inv_dx3 = FLUID_DENSITY * dx*dx*dx;  // m_i = ρ·dx³

//    // 3) Параллельно проходим по U‑узлам
//    thrust::for_each_n(
//            thrust::device,
//            thrust::make_counting_iterator<int>(0),
//            Usize,
//            AccumulateBodyForcesU{
//                    gridWidth, gridHeight, gridDepth,
//                    dx, dt, FLUID_DENSITY,
//                    body.pos,
////                    body.vel,
////                    body.omega,
//                    thrust::raw_pointer_cast(labels.device_data.data()),
//                    thrust::raw_pointer_cast(uStar.data()),
//                    thrust::raw_pointer_cast(u.device_data.data()),
//                    thrust::raw_pointer_cast(forceAcc.data()),
//                    thrust::raw_pointer_cast(torqueAcc.data())
//            }
//    );
//
//    cudaDeviceSynchronize();
//
//    thrust::for_each_n(
//            thrust::device,
//            thrust::make_counting_iterator<int>(0),
//            Vsize,
//            AccumulateBodyForcesV{
//                    gridWidth, gridHeight, gridDepth,
//                    dx, dt, FLUID_DENSITY,
//                    body.pos,
//                    thrust::raw_pointer_cast(labels.device_data.data()),
//                    thrust::raw_pointer_cast(vStar.data()),
//                    thrust::raw_pointer_cast(v.device_data.data()),
//                    thrust::raw_pointer_cast(forceAcc.data()),
//                    thrust::raw_pointer_cast(torqueAcc.data())
//            }
//    );
//
//    cudaDeviceSynchronize();
//
//    thrust::for_each_n(
//            thrust::device,
//            thrust::make_counting_iterator<int>(0),
//            Wsize,
//            AccumulateBodyForcesW{
//                    gridWidth, gridHeight, gridDepth,
//                    dx, dt, FLUID_DENSITY,
//                    body.pos,
////                    body.vel,
////                    body.omega,
//                    thrust::raw_pointer_cast(labels.device_data.data()),
//                    thrust::raw_pointer_cast(wStar.data()),
//                    thrust::raw_pointer_cast(w.device_data.data()),
//                    thrust::raw_pointer_cast(forceAcc.data()),
//                    thrust::raw_pointer_cast(torqueAcc.data())
//            }
//    );

    // Создание функтора
        PressureForceCalculator calculator = {
                gridWidth, gridHeight, gridDepth, dx,
                body.pos,
                labels.device_ptr(), p.device_ptr(),
                thrust::raw_pointer_cast(forceAcc.data()),
                thrust::raw_pointer_cast( torqueAcc.data())
        };

    // Запуск для всех ячеек
        thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(w_x_h_x_d),
                calculator
        );

    cudaDeviceSynchronize();

    // 5) Скачиваем на host и интегрируем
    float3 totalF  = forceAcc[0];
    float3 totalt = torqueAcc[0];

    body.force = body.mass * GRAVITY;
    //body.force  = body.force + totalF;
    //body.torque = body.torque + totalt;
    applyBuoyancy();

     std::cout << " Force: " << body.force.x << ", " << body.force.y << ", " << body.force.z
              << std::endl;

    // 6) Интегрируем тело
    body.integrate(dt);

    //handleBodyWallCollisions();

}

void FluidSolver3D::handleBodyWallCollisions() {
    // 1) Проверяем совпадение dx и sdf_cell_size
    assert(fabs(dx - body.sdf_cell_size) < 1e-6);

    // 2) Радиус тела
    float r = 0.5f * fmaxf(fmaxf(body.size.x, body.size.y), body.size.z);

    // 3) Центр ячейки контейнера (в world) для SOLID‑границы
    float half = 0.5f * dx;
    float3 minWall = make_float3(half, half, half);
    float3 maxWall = make_float3(gridWidth*dx - half,
                                  gridHeight*dx - half,
                                  gridDepth*dx - half);

    // локальная функция для коррекции вдоль одной оси:
    auto collideAxis = [&](float &p, float &v, float pMin, float pMax, float3 &pos, float3 &vel, const float3 &n) {
        // dist to nearest wall plane
        float distMin = p - pMin;
        float distMax = pMax - p;
        float dist   = fminf(distMin, distMax);
        float penetration = r - dist;
        if (penetration > 0.0f) {
            // 4) Корректируем позицию: «выдавливаем» тело наружу
            p = p + penetration * (distMin < distMax ? +1.0f : -1.0f);
            // 5) Нормальную скорость обнуляем
            float vn = vel *  n;
            if (vn < 0.0f) {
                vel = vel -  vn * n;
            }
        }
    };

    float3 &P = body.pos;
    float3 &V = body.vel;

    // X‑оси
    collideAxis(P.x, V.x, minWall.x + r, maxWall.x - r,
                body.pos, body.vel, make_float3(1,0,0));
    // Y‑оси
    collideAxis(P.y, V.y, minWall.y + r, maxWall.y - r,
                body.pos, body.vel, make_float3(0,1,0));
    // Z‑оси
    collideAxis(P.z, V.z, minWall.z + r, maxWall.z - r,
                body.pos, body.vel, make_float3(0,0,1));

    // 6) Сдвигаем SDF‑origin
    body.sdf_origin = body.pos - body.size * 0.5f;
}

// Функция для ограничения значения в диапазоне [min_val, max_val]
template<typename T>
__host__ __device__ T clamp(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Функция для вычисления расстояния между двумя точками в 3D
__host__ __device__ float distance(const float3& a, const float3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}
void FluidSolver3D::applyBuoyancy() {
    // 1) Копируем необходимые данные на хост
    body.sdf_data.copy_to_host();
    labels.host_data = labels.device_data;

    // 2) Параметры SDF-сетки тела
    int sw = body.sdf_data.width();
    int sh = body.sdf_data.height();
    int sd = body.sdf_data.depth();
    float cs = body.sdf_cell_size;
    float3 origin = body.sdf_origin;

    // 3) Создаем grid для быстрого поиска частиц в окрестностях
    float search_radius = 2.0f * dx;  // радиус поиска частиц
    std::vector<std::vector<int>> grid(w_x_h_x_d);  // 3D grid для частиц
    float3 domain_min = make_float3(0.0f,0.0f,0.0f);
    for(int idx = 0; idx < h_particles.size(); ++idx) {
        const auto& p = h_particles[idx];
        int i = clamp(static_cast<int>((p.pos.x - domain_min.x) / dx), 0, gridWidth-1);
        int j = clamp(static_cast<int>((p.pos.y - domain_min.y) / dx), 0, gridHeight-1);
        int k = clamp(static_cast<int>((p.pos.z - domain_min.z) / dx), 0, gridDepth-1);
        grid[i + j*gridWidth + k*gridWidth*gridHeight].push_back(idx);
    }

    // 4) Перебираем воксели тела, считаем погруженные с учетом реального уровня жидкости
    float totalVolume = 0.0f;
    float submergedVolume = 0.0f;

    for (int k = 0; k < sd; ++k) {
        for (int j = 0; j < sh; ++j) {
            for (int i = 0; i < sw; ++i) {
                int sdfIdx = i + j*sw + k*sw*sh;
                float sdf_val = body.sdf_data.host_data[sdfIdx];

                // Пропускаем внешние воксели (SDF > 0)
                if (sdf_val > 0.0f) continue;

                // Мировые координаты центра вокселя
                float3 voxel_center = origin + make_float3(
                        (i + 0.5f) * cs,
                        (j + 0.5f) * cs,
                        (k + 0.5f) * cs
                );

                // Объем вокселя (с учетом SDF для частично заполненных)
                float voxel_vol = cs*cs*cs;

                // Для частично заполненных вокселей корректируем объем
                if (sdf_val > -cs) {
                    float fraction = 0.5f - 0.5f * sdf_val / cs;
                    voxel_vol *= fraction;
                }

                totalVolume += voxel_vol;

                // Проверяем, находится ли воксель в жидкости
                int ii = clamp(static_cast<int>((voxel_center.x - domain_min.x) / dx), 0, gridWidth-1);
                int jj = clamp(static_cast<int>((voxel_center.y - domain_min.y) / dx), 0, gridHeight-1);
                int kk = clamp(static_cast<int>((voxel_center.z - domain_min.z) / dx), 0, gridDepth-1);
                int gridIdx = ii + jj*gridWidth + kk*gridWidth*gridHeight;

                // Вариант 1: Проверка по сетке жидкости
                bool isSubmerged = false;
                if (labels.host_data[gridIdx] == Utility::FLUID) {
                    isSubmerged = true;
                }
                    // Вариант 2: Проверка по близости к частицам (более точный)
                else {
                    for (int pidx : grid[gridIdx]) {
                        const auto& p = h_particles[pidx];
                        if (distance(p.pos, voxel_center) < search_radius) {
                            isSubmerged = true;
                            break;
                        }
                    }
                }

                if (isSubmerged) {
                    submergedVolume += voxel_vol;
                }
            }
        }
    }

    // 5) Архимедова сила: F_arch = ρ_fluid · V_submerged · g
    float3 F_arch = -FLUID_DENSITY * submergedVolume * GRAVITY;

    // 6) Добавляем демпфирование для стабилизации
    float damping = 0.1f;
    body.force = body.force + F_arch - damping * body.vel * body.mass;

    // Отладочная информация
//    std::cout << "Submerged volume: " << submergedVolume
//              << " / " << totalVolume
//              << " (" << (submergedVolume/totalVolume)*100 << "%)"
//              << " Force: " << F_arch.x << ", " << F_arch.y << ", " << F_arch.z
//              << std::endl;
}

//##############################
// general loop funcs##########
__host__ void FluidSolver3D::frameStep(){
    labelGrid();
    updateBody();
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

    advectParticles(0.1);

    
    //computeBodyForces();


}

__host__ void FluidSolver3D::run(int max_steps) {
//    csv_file.open("OutputData/csvRes.csv");
//    csv_file << "frame,centerX1,centerX2,centerX3,angleX1,angleX2,angleX3\n";
    d_particles = h_particles;
    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    for(int i = 0; i <= max_steps*iterPerFrame; ++i){
        frameStep();
        if(i%iterPerFrame == 0){
            h_particles = d_particles;
            switch(outputFormat){
                case PLY:
                    Utility::save3dParticlesToPLY(h_particles, outputTemplate + std::to_string(i/iterPerFrame ) + ".ply");
                    break;
                case OFF:
                    Utility::save3dParticlesToOFF(h_particles, outputTemplate + std::to_string(i/iterPerFrame ) + ".off");
                    break;
                default:
                    Utility::save3dParticlesToPLY(h_particles, outputTemplate + std::to_string(i/iterPerFrame ) + ".ply");
            }
            std::cout << "frame = " << i / iterPerFrame   << "; numParticles = " << h_particles.size()<<std::endl;


            // Генерируем и сохраняем поверхность тела
            body.generateSurfacePoints(0.5*dx);  // Плотность точек
            body.exportToPLY("OutputData/body_" + std::to_string(i/iterPerFrame) + ".ply");

//            saveLabelsToPLY("OutputData/labels_" + std::to_string(i/iterPerFrame) + ".ply");
//
//            float3 angles = Utility::quaternion_to_ship_angles(body.orientation);
//             csv_file << i/iterPerFrame << ","
//                 << body.pos.x << ","
//                  << body.pos.y << ","
//                  << body.pos.z << ","
//                  << angles.x << ","   // Крен (roll)
//                  << angles.y << ","   // Тангаж (pitch)
//                 << angles.z << "\n"; // Рыскание (yaw)   // << std::setprecision(precision)
//
//            // Обеспечиваем немедленную запись на диск
//            csv_file.flush();
        }

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    std::cout << "elapsed time = " << elapsedTime / 1000.0f << std::endl;
}






//????????????????
float FluidSolver3D::interpolatePressure(const float3& pos) {
    // Переводим мировые координаты в сеточные
    float rx = pos.x / dx;
    float ry = pos.y / dx;
    float rz = pos.z / dx;

    int i = static_cast<int>(floorf(rx));
    int j = static_cast<int>(floorf(ry));
    int k = static_cast<int>(floorf(rz));

    // Зажимаем индексы в пределах сетки
    i = std::clamp(i, 0, gridWidth - 1);
    j = std::clamp(j, 0, gridHeight - 1);
    k = std::clamp(k, 0, gridDepth - 1);

    // Дробные части
    float fx = rx - i;
    float fy = ry - j;
    float fz = rz - k;

    // Трилинейная интерполяция
    return trilinearInterpolation(
            fx, fy, fz,
            p(i, j, k),     p(i+1, j, k),     p(i, j, k+1),   p(i+1, j, k+1),
            p(i, j+1, k),   p(i+1, j+1, k),   p(i, j+1, k+1), p(i+1, j+1, k+1)
    );
}


void FluidSolver3D::saveLabelsToPLY(const std::string& filename) {
    // Копируем метки с устройства на хост
    thrust::host_vector<int> labels_h = labels.device_data;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    // Считаем количество не-AIR ячеек и не-SOlid
    size_t count = 0;
    for (int k = 0; k < gridDepth; k++) {
        for (int j = 0; j < gridHeight; j++) {
            for (int i = 0; i < gridWidth; i++) {
                int idx = i + j * gridWidth + k * gridWidth * gridHeight;
                if (labels_h[idx] != Utility::AIR && labels_h[idx] != Utility::SOLID && labels_h[idx] != Utility::FLUID) count++;
            }
        }
    }
    
    // Записываем заголовок PLY
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << count << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Записываем данные
    for (int k = 0; k < gridDepth; k++) {
        for (int j = 0; j < gridHeight; j++) {
            for (int i = 0; i < gridWidth; i++) {
                int idx = i + j * gridWidth + k * gridWidth * gridHeight;
                int label = labels_h[idx];
                
                if (label == Utility::AIR || label == Utility::SOLID|| label == Utility::FLUID) continue;
                
                // Центр ячейки
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;
                float z = (k + 0.5f) * dx;
                
                // Цвет в зависимости от типа ячейки
                unsigned char r = 0, g = 0, b = 0;
                switch (label) {
                    case Utility::FLUID: 
                        r = 0; g = 0; b = 255;  // Синий
                        break;
                    case Utility::SOLID: 
                        r = 128; g = 128; b = 128;  // Серый
                        break;
                    case Utility::BODY: 
                        r = 255; g = 0; b = 0;  // Красный
                        break;
                    default: 
                        r = 255; g = 255; b = 255;  // Белый (не должно быть)
                }
                
                file << x << " " << y << " " << z << " "
                     << static_cast<int>(r) << " "
                     << static_cast<int>(g) << " "
                     << static_cast<int>(b) << "\n";
            }
        }
    }
    
    file.close();
    std::cout << "Saved labels to: " << filename << std::endl;
}