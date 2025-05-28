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


void FluidSolver3D::init(const std::string& fileName) {
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
    //labelGrid();
    //frameStep();
}

void FluidSolver3D::seedParticles(int particlesPerCell){
    // Инициализация генератора (один раз вне функции!)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> subCellDist(0, 7);
    static std::uniform_real_distribution<> jitterDist(-0.24f, 0.24f);

    // Сначала подсчитываем общее количество частиц
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

    blocksForParticles = (h_particles.size() + threadsPerBlock- 1) / threadsPerBlock;
}
