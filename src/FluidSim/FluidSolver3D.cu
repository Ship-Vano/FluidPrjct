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
bool projectParticleDevice3D(Utility::Particle3D &p,
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
    int cx = int(floorf(p.pos.x/dx));
    int cy = int(floorf(p.pos.y/dx));
    int cz = int(floorf(p.pos.z/dx));

    float3 bestPos = p.pos;
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
            float d = sqrtf((posC.x-p.pos.x)*(posC.x-p.pos.x)
                        + (posC.y-p.pos.y)*(posC.y-p.pos.y)
                        + (posC.z-p.pos.z)*(posC.z-p.pos.z));
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
    p.pos.x = bestPos.x;
    p.pos.y = bestPos.y;
    p.pos.z = bestPos.z;
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
        Utility::Particle3D p = pin;
        float subT = 0.0f;
        bool finished = false;

        while (!finished) {
            // 1) Интерполируем скорость
            float3 vel = interpVelDevice3D(u,v,w, W,H,D, dx, p.pos);

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
            p.pos.x += vel.x * dT;
            p.pos.y += vel.y * dT;
            p.pos.z += vel.z * dT;
            subT += dT;

            // 4) Границы и NaN
            if (p.pos.x < 0 || p.pos.y < 0 || p.pos.z < 0 ||
                isnan(p.pos.x) || isnan(p.pos.y) || isnan(p.pos.z)) {
                break;
            }

            // 5) Попал в SOLID?
            int cx = int(floorf(p.pos.x/dx));
            int cy = int(floorf(p.pos.y/dx));
            int cz = int(floorf(p.pos.z/dx));
            if (isCellValid(cx,cy,cz,W,H,D) &&
                labels[idx3d(cx,cy,cz,W,H)] == Utility::SOLID)
            {
                if (!projectParticleDevice3D(p, labels, W,H,D, dx*0.25f))
                    break;
            }
        }
        return p;
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

    __host__ __device__
    float operator()(int idx) const {
        int i = idx % W;
        int j = (idx / W) % H;
        int k = idx / (W * H);

        if (labels[idx] != Utility::FLUID) return 0.0f;

        float div =
                u[i+1 + j*(W+1) + k*(W+1)*H] - u[i + j*(W+1) + k*(W+1)*H] +
                v[i + (j+1)*W + k*W*(H+1)] - v[i + j*W + k*W*(H+1)] +
                w[i + j*W + (k+1)*W*H] - w[i + j*W + k*W*H];

        float rhs_val = -scale * div;

        // Solid boundary conditions
        if (i-1 >= 0 && labels[idx - 1] == Utility::SOLID)
            rhs_val -= scale * (u[i + j*(W+1) + k*(W+1)*H] - 0.0f);
        if (i+1 < W && labels[idx + 1] == Utility::SOLID)
            rhs_val += scale * (u[i+1 + j*(W+1) + k*(W+1)*H] - 0.0f);

        if (j-1 >= 0 && labels[idx - W] == Utility::SOLID)
            rhs_val -= scale * (v[i + j*W + k*W*(H+1)] - 0.0f);
        if (j+1 < H && labels[idx + W] == Utility::SOLID)
            rhs_val += scale * (v[i + (j+1)*W + k*W*(H+1)] - 0.0f);

        if (k-1 >= 0 && labels[idx - W*H] == Utility::SOLID)
            rhs_val -= scale * (w[i + j*W + k*W*H] - 0.0f);
        if (k+1 < D && labels[idx + W*H] == Utility::SOLID)
            rhs_val += scale * (w[i + j*W + (k+1)*W*H] - 0.0f);

        return rhs_val;
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


void FluidSolver3D::constructRHS(thrust::device_vector<float>& rhs) {

    float scale = (FLUID_DENSITY * dx) / dt;
    int totalCells = gridWidth * gridHeight * gridDepth;

    thrust::device_vector<float> rhs_temp(totalCells);

    thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(totalCells),
            rhs_temp.begin(),
            RHSCalculator3D{
                    thrust::raw_pointer_cast(labels.device_ptr()),
                    thrust::raw_pointer_cast(u.device_ptr()),
                    thrust::raw_pointer_cast(v.device_ptr()),
                    thrust::raw_pointer_cast(w.device_ptr()),
                    scale,
                    gridWidth, gridHeight, gridDepth
            }
    );

    /* ДО (пример)
        Индексы:    [0]     [1]     [2]     [3]
        rhs_temp:  [1.0]  [2.0]  [3.0]  [4.0]
        labels:    [SOLID] [FLUID] [AIR] [FLUID]
     * */
    /* После (пример)
        rhs: [2.0] [4.0]  // Только FLUID-ячейки
        fluidCellsAmount = 2
     * */

    //  Подсчитываем количество жидких ячеекisFLuid isFluid;

    fluidCellsAmount = thrust::transform_reduce(
            thrust::device,
            labels.device_data.begin(),
            labels.device_data.end(),
            FluidFlagFunctor(),  // Преобразует метку в 1/0
            0,                       // Начальное значение суммы
            thrust::plus<int>()      // Операция сложения
    );

    rhs.resize(fluidCellsAmount);

    // Compact to fluid cells only
    auto new_end = thrust::copy_if(
            rhs_temp.begin(), rhs_temp.end(),
            thrust::counting_iterator<int>(0),
            rhs.begin(),
            FluidCellPredicate{
                    thrust::raw_pointer_cast(labels.device_ptr()),
                    Utility::FLUID
            }
    );
    //fluidCellsAmount = thrust::distance(rhs.begin(), new_end);
}



int FluidSolver3D::pressureSolve() {
    // новая нумерация
    thrust::device_vector<int> fluidNumbers_d(gridWidth * gridHeight * gridDepth, -1);
    thrust::sequence(thrust::device, fluidNumbers_d.begin(), fluidNumbers_d.end());
    thrust::device_vector<int> flags(labels.size());

    /*ЧТО ХОТИМ СДЕЛАТЬ НИЖЕ: ВВЕСТИ НОВУЮ НУМЕРАЦИЮ.
     * Ячейка	Метка	flags	fluidNumbers_d
       (0,0,0)	FLUID	 1	     0
       (1,0,0)	SOLID	 0	     1
       (2,0,0)	FLUID	 1	     1
     * */
    thrust::transform(
            thrust::device,
            labels.device_ptr(),
            labels.device_ptr() + labels.size(),
            flags.begin(),
            FluidFlagFunctor()
    );

    thrust::exclusive_scan(
            thrust::device,
            flags.begin(), flags.end(),
            fluidNumbers_d.begin()
    );

    // Construct RHS and matrix
    thrust::device_vector<float> rhs_d(fluidCellsAmount);
    thrust::device_vector<float> csr_values;
    thrust::device_vector<int> csr_columns;
    thrust::device_vector<int> csr_offsets(fluidCellsAmount+1);

    constructRHS(rhs_d);
//    constructA(csr_values, csr_columns, csr_offsets);
//
//    // Solve with cuDSS (similar to 2D version)
//    // ... [cuDSS setup and solve] ...
//
//    // Copy pressure back to grid
//    thrust::copy(x_solution.begin(), x_solution.end(), p.device_ptr());
    return 1;
}

void FluidSolver3D::applyPressure() {

}

__host__ void FluidSolver3D::frameStep(){
    labelGrid();

    //particles velocities to grid
    particlesToGrid();

    //saving a copy of the current grid velocities (for FLIP)
    saveVelocities();

    //applying body forces on grid (e.g. gravity force)
    applyForces();

    //grid velocities to particles
    gridToParticles(PIC_WEIGHT);

    advectParticles(ADVECT_MAX);

    pressureSolve();
//    applyPressure();


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
