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
    w_x_h_x_d = gridWidth * gridHeight * gridDepth;
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
    Utility::save3dParticlesToPLY(h_particles, "InputData/particles_-1.ply");
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
        Utility::Particle3D particle = particles[pidx];
        float3 pos = particle.pos;
        float3 vel = particle.vel;

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

struct ApplyScalarForce
{
    float dt, a, vel_unknown;

    ApplyScalarForce(float _dt, float _a, float _vel_unknown)
        : dt(_dt), a(_a), vel_unknown(_vel_unknown) {}

    __host__ __device__
    float operator()(const float& x) const {
        return (x == vel_unknown) ? x : x + dt * a;
    }
};

void FluidSolver3D::applyForces(){

    thrust::transform(
        u.device_data.begin(), u.device_data.end(),         // вход
        u.device_data.begin(),                  // выход 
        ApplyScalarForce(dt, GRAVITY.x, VEL_UNKNOWN)
    );

    thrust::transform(
        v.device_data.begin(), v.device_data.end(),
        v.device_data.begin(),
        ApplyScalarForce(dt, GRAVITY.y, VEL_UNKNOWN)
    );

    thrust::transform(
        w.device_data.begin(), w.device_data.end(),
        w.device_data.begin(),
        ApplyScalarForce(dt, GRAVITY.z, VEL_UNKNOWN)
    );
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

// ----------------------------------
// 2) Функтор переноса скоростей с сетки на частицу
struct GridToParticleFunctor
{
    int W,H,D;
    float3 origin;
    float  dx, alpha;

    // raw-пойнтеры на device_data[]
    const float *u, *v, *w;
    const float *du,*dv,*dw;

    GridToParticleFunctor(int _W,int _H,int _D,
                            float3 _orig,float _dx,float _alpha,
                            const float* _u,const float* _v,const float* _w,
                            const float* _du,const float* _dv,const float* _dw)
        : W(_W), H(_H), D(_D),
        origin(_orig), dx(_dx), alpha(_alpha),
        u(_u), v(_v), w(_w),
        du(_du), dv(_dv), dw(_dw)
    {}

    __device__
    Utility::Particle3D operator()(const Utility::Particle3D& pin) const
    {
        Utility::Particle3D pout = pin;

        // локальные координаты в ячейках
        float rel_x = (pin.pos.x - origin.x) * (1.0f / dx);
        float rel_y = (pin.pos.y - origin.y) * (1.0f / dx);
        float rel_z = (pin.pos.z - origin.z) * (1.0f / dx);

        int i = floorf(rel_x);
        int j = floorf(rel_y);
        int k = floorf(rel_z);

        if (i < 0 || i >= W || j < 0 || j >= H || k < 0 || k >= D)
            return pout;

        float fx = rel_x - i;
        float fy = rel_y - j;
        float fz = rel_z - k;

        // PIC: исходные u,v,w
        float uc  = trilerp(u,  i, j, k, fx,fy,fz, (W+1), (W+1)*H);
        float vc  = trilerp(v,  i, j, k, fx,fy,fz,  W,     W*(H+1));
        float wc  = trilerp(w,  i, j, k, fx,fy,fz,  W,     W*H    );

        // FLIP: дельты
        float duc = trilerp(du, i, j, k, fx,fy,fz, (W+1), (W+1)*H);
        float dvc = trilerp(dv, i, j, k, fx,fy,fz,  W,     W*(H+1));
        float dwc = trilerp(dw, i, j, k, fx,fy,fz,  W,     W*H    );

        // микс PIC/FLIP
        pout.vel.x = alpha * uc + (pin.vel.x + duc) * (1.0f - alpha);
        pout.vel.y = alpha * vc + (pin.vel.y + dvc) * (1.0f - alpha);
        pout.vel.z = alpha * wc + (pin.vel.z + dwc) * (1.0f - alpha);

        return pout;
    }
};

// ----------------------------------
// 3) Обновляем метод FluidSolver3D::gridToParticles
void FluidSolver3D::gridToParticles(float alpha)
{
    // размеры MAC-решёток
    int Nu = (gridWidth+1)*gridHeight*gridDepth;
    int Nv = gridWidth*(gridHeight+1)*gridDepth;
    int Nw = gridWidth*gridHeight*(gridDepth+1);

    // 3.1) вычисляем дельты в конструкторах device_vector
    thrust::device_vector<float> du(Nu);
    thrust::device_vector<float> dv(Nv);
    thrust::device_vector<float> dw(Nw);

    // вместо zip_iterator:
    thrust::transform(
        u.device_data.begin(),                   // new
        u.device_data.end(),
        uSaved.device_data.begin(),              // old
        du.begin(),                              // out
        thrust::minus<float>()                   // new - old
    );

    thrust::transform(
        v.device_data.begin(),
        v.device_data.end(),
        vSaved.device_data.begin(),
        dv.begin(),
        thrust::minus<float>()
    );

    thrust::transform(
        w.device_data.begin(),
        w.device_data.end(),
        wSaved.device_data.begin(),
        dw.begin(),
        thrust::minus<float>()
    );

    // 3.2) raw-пойнтеры из Grid3D
    const float* pu  = u.device_ptr();
    const float* pv  = v.device_ptr();
    const float* pw  = w.device_ptr();
    const float* pdu = du.data().get();  // device_vector<float>::data().get()
    const float* pdv = dv.data().get();
    const float* pdw = dw.data().get();

    // 3.3) один transform по частицам
    thrust::transform(
        d_particles.begin(),
        d_particles.end(),
        d_particles.begin(),
        GridToParticleFunctor(
            gridWidth, gridHeight, gridDepth,
            make_float3(0,0,0),  // origin
            dx, alpha,
            pu, pv, pw,
            pdu, pdv, pdw
        )
    );
}

__device__ inline bool isCellValid(int x, int y, int z, int W, int H, int D) {
    return x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D;
}

__device__ inline int idx3d(int x , int y, int z, int W, int H){
    return x + y * W + z * W * H;
}

__device__
float3 interpVelDevice3D(const float* uGrid, const float* vGrid, const float* wGrid,
                            int W,int H,int D, float dx, float3 pos){
    // переводим в "ячейковые" координаты
    float rx = pos.x / dx;
    float ry = pos.y / dx;
    float rz = pos.z / dx;
    int i = min(max(int(floorf(rx)), 0), W);
    int j = min(max(int(floorf(ry)), 0), H);
    int k = min(max(int(floorf(rz)), 0), D);
    float fx = rx - i;
    float fy = ry - j;
    float fz = rz - k;

    // strides
    int s_i = 1;
    int s_j = (W+1);

    float ux = trilerp(uGrid, i, j, k, fx, fy, fz, s_i, s_j);
    float vy = trilerp(vGrid, i, j, k, fx, fy, fz, s_i, s_j);
    float wz = trilerp(wGrid, i, j, k, fx, fy, fz, s_i, s_j);

    return make_float3(ux, vy, wz);
}

__device__
bool projectParticleDevice3D(Utility::Particle3D &particle,
                            const int* labels,
                            int W,int H,int D, float dx)
{
    // 26 соседей в 3D
    const int off[26][3] = {
        {-1,-1,-1},{-1,-1, 0},{-1,-1, 1},{-1, 0,-1},{-1, 0, 0},{-1, 0, 1},{-1, 1,-1},{-1, 1, 0},{-1, 1, 1},
        { 0,-1,-1},{ 0,-1, 0},{ 0,-1, 1},{ 0, 0,-1},            { 0, 0, 1},{ 0, 1,-1},{ 0, 1, 0},{ 0, 1, 1},
        { 1,-1,-1},{ 1,-1, 0},{ 1,-1, 1},{ 1, 0,-1},{ 1, 0, 0},{ 1, 0, 1},{ 1, 1,-1},{ 1, 1, 0},{ 1, 1, 1}
    };

    // текущая ячейка
    int cx = int(floorf(particle.pos.x/dx));
    int cy = int(floorf(particle.pos.y/dx));
    int cz = int(floorf(particle.pos.z/dx));

    float3 bestPos = particle.pos;
    float  bestD   = 1e10f;
    bool   found   = false;

    // сначала ищем SOLID, потом AIR
    for(int pass=0;pass<2;++pass){
        int want = (pass==0 ? Utility::SOLID : Utility::AIR);
        for(int n=0;n<26;++n){
            int nx = cx+off[n][0],
                ny = cy+off[n][1],
                nz = cz+off[n][2];
            if(!isCellValid(nx,ny,nz,W,H,D)) continue;
            int idx = nx + ny*W + nz*W*H;
            if(labels[idx] != want) continue;

            float3 posC = make_float3(
                (nx+0.5f)*dx,
                (ny+0.5f)*dx,
                (nz+0.5f)*dx
            );
            float d = sqrtf((posC.x-particle.pos.x)*(posC.x-particle.pos.x)
                        + (posC.y-particle.pos.y)*(posC.y-particle.pos.y)
                        + (posC.z-particle.pos.z)*(posC.z-particle.pos.z));
            if(d < bestD){
                bestD = d;
                bestPos = posC;
                found = true;
            }
        }
        if(found) break;
    }
    if(!found) return false;

    // сдвигаем на 1.0*(bestPos - p.pos)
    particle.pos.x = bestPos.x;
    particle.pos.y = bestPos.y;
    particle.pos.z = bestPos.z;
    return true;
}

struct AdvectParticlesFunctor {
    float dt, dx, C;
    int W, H, D;
    const float* u;
    const float* v;
    const float* w;
    const int* labels;

    __host__ AdvectParticlesFunctor(float _dt,float _dx,float _C,
                            int _W,int _H,int _D,
                            const float* _u,const float* _v,const float* _w,
                            const int*   _labels)
        : dt(_dt), dx(_dx), C(_C),
        W(_W), H(_H), D(_D),
        u(_u), v(_v), w(_w),
        labels(_labels) {}

    __device__ Utility::Particle3D operator()(const Utility::Particle3D& pin) const {
        Utility::Particle3D particle = pin;
        float subT = 0.0f;
        bool finished = false;
        int iter = 0;
        while (!finished && iter++ < 100) {
            // 1) Интерполируем скорость
            float3 vel = interpVelDevice3D(u,v,w, W,H,D, dx, particle.pos);

            // 2) Считаем dT
            float speed = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z) + 1e-37f;
            float dT = (C * dx) / speed;
            if (subT + dT >= dt) {
                dT = dt - subT;
                finished = true;
            } else if (subT + 2*dT >= dt) {
                dT *= 0.5f;
            }

            // 3) Явный Эйлер
            particle.pos.x += vel.x * dT;
            particle.pos.y += vel.y * dT;
            particle.pos.z += vel.z * dT;
            subT += dT;

            // 4) Границы и NaN
            if (particle.pos.x < 0 || particle.pos.y < 0 || particle.pos.z < 0 ||
                isnan(particle.pos.x) || isnan(particle.pos.y) || isnan(particle.pos.z)) {
                break;
            }

            // 5) Попал в SOLID?
            int cx = int(floorf(particle.pos.x/dx));
            int cy = int(floorf(particle.pos.y/dx));
            int cz = int(floorf(particle.pos.z/dx));
            if (isCellValid(cx,cy,cz,W,H,D) &&
                labels[idx3d(cx,cy,cz,W,H)] == Utility::SOLID)
            {
                if (!projectParticleDevice3D(particle, labels, W,H,D, dx))
                    break;
            }
        }
        return particle;
    }
};

void FluidSolver3D::advectParticles(float C){

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
}


struct RHSCalculator3D {
    const int* labels;
    const float* u, *v, *w;
    float scale;
    int W, H, D;
    float* rhs_temp;

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

        // Solid boundaries
        if (i-1 >= 0 && labels[idx-1] == Utility::SOLID)
            rhs_val -= scale * (u[i + j*(W+1) + k*(W+1)*H] - 0.0f);
        if (i < W && labels[idx+1] == Utility::SOLID)
            rhs_val += scale * (u[(i+1) + j*(W+1) + k*(W+1)*H] - 0.0f);
        if (j-1 >= 0 && labels[idx-W] == Utility::SOLID)
            rhs_val -= scale * (v[i + j*W + k*W*(H+1)] - 0.0f);
        if (j < H && labels[idx+W] == Utility::SOLID)
            rhs_val += scale * (v[i + (j+1)*W + k*W*(H+1)] - 0.0f);
        if (k-1 >= 0 && labels[idx-W*H] == Utility::SOLID)
            rhs_val -= scale * (w[i + j*W + k*W*H] - 0.0f);
        if (k < D && labels[idx+W*H] == Utility::SOLID)
            rhs_val += scale * (w[i + j*W + (k+1)*W*H] - 0.0f);

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

struct FluidFlagFunctor {
    __host__ __device__
    int operator()(int label) const {
        return label == Utility::FLUID ? 1 : 0;
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
                    thrust::raw_pointer_cast(rhs_temp.data())
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

    // Вывод результата
//    thrust::host_vector<float> rhs_h = rhs;
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
    const int* labels;
    const int* fluidNumbers;
    int* nnz_per_row;

    __device__ void operator()(int idx) const {
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        if (labels[idx] != Utility::FLUID) return;

        int row = fluidNumbers[idx];
        int count = 1; // Диагональный элемент

        // 6 направлений соседей
        const int offsets[6][3] = {
                {1,0,0}, {-1,0,0},
                {0,1,0}, {0,-1,0},
                {0,0,1}, {0,0,-1}
        };

        for (int n = 0; n < 6; n++) {
            int ni = i + offsets[n][0];
            int nj = j + offsets[n][1];
            int nk = k + offsets[n][2];

            if (ni >= 0 && ni < W && nj >= 0 && nj < H && nk >= 0 && nk < D) {
                int nidx = ni + nj * W + nk * W * H;
                if (labels[nidx] == Utility::FLUID && fluidNumbers[nidx] > row) {
                    count++;
                }
            }
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

        for (int n = 0; n < 6; n++) {
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
                else if (labels[nidx] != Utility::SOLID) {
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
//            std::cout << rhs[i + j*gridWidth] << ", ";
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

    cudaStream_t stream = NULL;
    cudaStreamCreate(&stream);
    cudssHandle_t handle;
    cudssStatus_t status = cudssCreate(&handle);
    cudssSetStream(handle, stream);

    cudssConfig_t solverConfig;
    cudssData_t solverData;
    cudssConfigCreate(&solverConfig);
    cudssDataCreate(handle, &solverData);

    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "cuDSS init failed: " << status << std::endl;
        return -3;
    }

    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;// Symmetric Positive Definite
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;// Upper triangular stored
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    int nnz = csr_values.size();

    status = cudssMatrixCreateCsr(
            &A,
            fluidCellsAmount, fluidCellsAmount, nnz,
            thrust::raw_pointer_cast(csr_offsets.data()),
            nullptr,
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
//
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


    // Анализ
    status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS,
                          solverConfig, solverData, A, x, b);

    // Факторизация
    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,
                          solverConfig, solverData, A, x, b);

    // Решение
    status = cudssExecute(handle, CUDSS_PHASE_SOLVE,
                          solverConfig, solverData, A, x, b);

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


    status = cudssMatrixDestroy(A);
    status = cudssMatrixDestroy(b);
    status = cudssMatrixDestroy(x);
    cudssDataDestroy(handle, solverData);
    cudssConfigDestroy(solverConfig);
    cudssDestroy(handle);
    cudaStreamSynchronize(stream);

    return 0;
}

struct UFunctor {
    float* u;
    const float* p;
    const int* labels;
    float scale;
    int W, H, D;
    float VEL_UNKNOWN;

    UFunctor(float* u_, const float* p_, const int* labels_, float scale_,
             int W_, int H_, int D_, float vel_unknown)
            : u(u_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown) {}

    __host__ __device__
    void operator()(int idx) const {
        int k = idx / ((W+1) * H);
        int j = (idx % ((W+1) * H)) / (W+1);
        int i = (idx % ((W+1) * H)) % (W+1);

        if (i > 0 && i < W) {
            int left = (i-1) + j*W + k*W*H;
            int right = i + j*W + k*W*H;

            if(labels[left] == Utility::FLUID || labels[right] == Utility::FLUID) {
                if(labels[left] == Utility::SOLID || labels[right] == Utility::SOLID) {
                    u[idx] = 0.0f;
                } else {
                    u[idx] -= scale * (p[right] - p[left]);
                }
            } else {
                u[idx] = VEL_UNKNOWN;
            }
        }
    }
};

struct VFunctor {
    float* v;
    const float* p;
    const int* labels;
    float scale;
    int W, H, D;
    float VEL_UNKNOWN;

    VFunctor(float* v_, const float* p_, const int* labels_, float scale_,
             int W_, int H_, int D_, float vel_unknown)
            : v(v_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown) {}

    __host__ __device__
    void operator()(int idx) const {
        int k = idx / (W * (H+1));
        int j = (idx % (W * (H+1))) / W;
        int i = (idx % (W * (H+1))) % W;

        if (j > 0 && j < H) {
            int left = i + (j-1)*W + k*W*H;
            int right = i + j*W + k*W*H;

            if(labels[left] == Utility::FLUID || labels[right] == Utility::FLUID) {
                if(labels[left] == Utility::SOLID || labels[right] == Utility::SOLID) {
                    v[idx] = 0.0f;
                } else {
                    v[idx] -= scale * (p[right] - p[left]);
                }
            } else {
                v[idx] = VEL_UNKNOWN;
            }
        }
    }
};

struct WFunctor {
    float* w;
    const float* p;
    const int* labels;
    float scale;
    int W, H, D;
    float VEL_UNKNOWN;

    WFunctor(float* w_, const float* p_, const int* labels_, float scale_,
             int W_, int H_, int D_, float vel_unknown)
            : w(w_), p(p_), labels(labels_), scale(scale_),
              W(W_), H(H_), D(D_), VEL_UNKNOWN(vel_unknown) {}

    __host__ __device__
    void operator()(int idx) const {
        int k = idx / (W * H);
        int j = (idx % (W * H)) / W;
        int i = (idx % (W * H)) % W;

        if (k > 0 && k < D) {
            int left = i + j*W + (k-1)*W*H;
            int right = i + j*W + k*W*H;

            if(labels[left] == Utility::FLUID || labels[right] == Utility::FLUID) {
                if(labels[left] == Utility::SOLID || labels[right] == Utility::SOLID) {
                    w[idx] = 0.0f;
                } else {
                    w[idx] -= scale * (p[right] - p[left]);
                }
            } else {
                w[idx] = VEL_UNKNOWN;
            }
        }
    }
};

void FluidSolver3D::applyPressure() {
    float scale = dt / (FLUID_DENSITY * dx);
    float vel_unknown = (float)VEL_UNKNOWN; // Предполагается, что VEL_UNKNOWN определен

    // Обработка u-компоненты
    int u_size = (gridWidth+1) * gridHeight * gridDepth;
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(u_size),
            UFunctor(
                    u.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth, gridHeight, gridDepth,
                    vel_unknown
            )
    );

    // Обработка v-компоненты
    int v_size = gridWidth * (gridHeight+1) * gridDepth;
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(v_size),
            VFunctor(
                    v.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth, gridHeight, gridDepth,
                    vel_unknown
            )
    );

    // Обработка w-компоненты
    int w_size = gridWidth * gridHeight * (gridDepth+1);
    thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(w_size),
            WFunctor(
                    w.device_ptr(),
                    p.device_ptr(),
                    labels.device_ptr(),
                    scale,
                    gridWidth, gridHeight, gridDepth,
                    vel_unknown
            )
    );
}

__host__ void FluidSolver3D::frameStep(){
    labelGrid();

    //particles velocities to grid
    particlesToGrid();

    //saving a copy of the current grid velocities (for FLIP)
    saveVelocities();

    //applying body forces on grid (e.g. gravity force)
    applyForces();
    cudaDeviceSynchronize();
    pressureSolve();
    cudaDeviceSynchronize();
    //applyPressure();

    //grid velocities to particles
    gridToParticles(PIC_WEIGHT);

    advectParticles(ADVECT_MAX);

}

__host__ void FluidSolver3D::run(int max_steps) {
    d_particles = h_particles;
    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    for(int i = 0; i < max_steps; ++i){
        frameStep();
        if(i%10 == 0){
            h_particles = d_particles;
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
