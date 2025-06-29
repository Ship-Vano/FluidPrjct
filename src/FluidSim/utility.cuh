#ifndef UTILITY_H
#define UTILITY_H
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cusparse.h>
#include "cudss.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/iterator/permutation_iterator.h>


#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <iomanip>

__host__ __device__ float3 operator+(const float3& a, const float3& b);

__host__ __device__ float3 operator-(const float3& a, const float3& b);

__host__ __device__ float3 operator*(const float3& a, float b);

__host__ __device__ float3 operator*(float b, const float3& a);

__host__ __device__ float operator*(const float3& a, const float3& b);

__host__ __device__ float3 operator/(const float3&a, const float&b);

__host__ __device__ double3 operator+(const double3& a, const double3& b);

__host__ __device__ double3 operator-(const double3& a, const double3& b);

__host__ __device__ double3 operator*(const double3& a, float b);

__host__ __device__ float2 operator+(const float2& a, const float2& b);

__host__ __device__ float2 operator-(const float2& a, const float2& b);

__host__ __device__ float2 operator*(const float2& a, float b);


const int VEL_UNKNOWN = INT_MIN;

enum FileOutputFormat{
    PLY,
    OFF
};

namespace Utility {

    const int SOLID = 0;
    const int FLUID = 1;
    const int AIR = 2;
    const int BODY = 3;

    struct Particle2D {
        float2 pos;
        float2 vel;
        Particle2D(float2 pos, float2 vel): pos(pos), vel(vel){}
    };

    struct Particle3D {
        float3 pos;
        float3 vel;
        Particle3D(float3 pos, float3 vel): pos(pos), vel(vel){}
    };

    float2 getGridCellPosition(float i, float j, float dx);
    float3 getGridCellPosition(float i, float j, float k, float dx);
    int2 getGridCellIndex(float2 pos, float dx);

    void saveParticlesToFile(const std::vector<Particle2D>& particles,
                             const std::string& filename);
    void saveParticlesToPLY(const std::vector<Particle2D>& particles,
                            const std::string& filename);
    void save3dParticlesToPLY(const thrust::host_vector<Particle3D>& particles,
                              const std::string& filename);
    void save3dParticlesToOFF(const thrust::host_vector<Particle3D>& particles,
                              const std::string& filename);

    __device__ int getGridCellIndex_device(float2 pos, float dx, int gridWidth);

    __device__ float2 getGridCellPosition_device(float i, float j, float dx);

    __device__ float bilinearHatKernel(float2 dist, float dx, float dy);

    __device__ float hatFunction(float r);

    __device__ int2 getGridIndicesU(int ind, int gridWidth);

    __device__ int2 getGridIndicesV(int ind, int gridWidth);


    template <typename T>
    class Grid3D {
    protected:
        int m_width, m_height, m_depth;

    public:
        thrust::host_vector<T> host_data;
        thrust::device_vector<T> device_data;

        Grid3D() : m_width(0), m_height(0), m_depth(0) {}

        __host__ void resize(int w, int h, int d) {
            m_width = w;
            m_height = h;
            m_depth = d;
            host_data.resize(w * h * d);
            device_data.resize(w * h * d);
        }

        __host__ void copy_to_device() {
            thrust::copy(host_data.begin(), host_data.end(), device_data.begin());
        }

        __host__ void copy_to_host() {
            thrust::copy(device_data.begin(), device_data.end(), host_data.begin());
        }

        __host__ __device__ int width() const { return m_width; }
        __host__ __device__ int height() const { return m_height; }
        __host__ __device__ int depth() const { return m_depth; }
        __host__ __device__ int size() const { return m_width * m_height * m_depth; }

        // Доступ на хосте
        __host__ T& operator()(int i, int j, int k) {
            return host_data[i + j * m_width + k * (m_width * m_height)];
        }

        // Доступ на устройстве
        __device__ T& dev(int i, int j, int k) {
            return device_data[i + j * m_width + k * (m_width * m_height)];
        }

        // Указатель на device-данные
        __host__ T* device_ptr() {
            return thrust::raw_pointer_cast(device_data.data());
        }
    };

    struct float3x3 {
        float3 col0, col1, col2;

        __device__ float3x3()
                : col0(make_float3(0.0f, 0.0f, 0.0f)), col1(make_float3(0.0f, 0.0f, 0.0f)), col2(make_float3(0.0f, 0.0f, 0.0f)) {}

        __device__ float3x3(float3 c0, float3 c1, float3 c2)
                : col0(c0), col1(c1), col2(c2) {}

        // Умножение матрицы на вектор
        __device__ float3 operator*(const float3& v) const {
            return make_float3(
                    col0 * v,
                    col1 *  v,
                    col2 * v
            );
        }

        // Создание диагональной матрицы
        __device__ static float3x3 diag(float d) {
            return float3x3(
                    make_float3(d, 0, 0),
                    make_float3(0, d, 0),
                    make_float3(0, 0, d)
            );
        }
    };

    struct RigidBody{
        float3 pos; // Центр масс
        float3 vel; // Скорость
        float3 force; // Суммарная сила
        float mass; // Масса тела
        float inertia;       // Момент инерции (скалярное упрощение)
        float inv_inertia;   // Обратный момент инерции
//        float3x3 inertia; // Момент инерции
//        float3x3 inv_inertia; // Обратный момент инерции

        // SDF данные
        float* sdf_data;          // Сырой указатель на данные
        int sdf_dims[3];          // Размеры сетки SDF [width, height, depth]
        float3 sdf_origin;      // Минимальный угол сетки SDF (мировые координаты)
        float sdf_cell_size; // Размер ячейки SDF

        // Размеры тела в мировых координатах
        float3 size;

        thrust::host_vector<float3> surface_points;
        thrust::host_vector<float> sdf_data_host;  // Хост-копия данных SDF

        // Метод для загрузки SDF
        void loadSDF(const std::string& filename, const float3& initial_position) {
            // Освобождаем предыдущие данные
            if (sdf_data) {
                cudaFree(sdf_data);
                sdf_data = nullptr;
            }

            // Открываем файл в текстовом режиме
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open SDF file: " + filename);
            }

            // 1. Читаем размеры сетки
            std::string line;
            std::getline(file, line);
            std::istringstream dims_line(line);
            dims_line >> sdf_dims[0] >> sdf_dims[1] >> sdf_dims[2];
            std::cout << "SDF dimensions: "
                      << sdf_dims[0] << "x"
                      << sdf_dims[1] << "x"
                      << sdf_dims[2] << std::endl;

            // 2. Читаем начало координат SDF
            std::getline(file, line);
            std::istringstream origin_line(line);
            origin_line >> sdf_origin.x >> sdf_origin.y >> sdf_origin.z;
            std::cout << "Original SDF origin: ("
                      << sdf_origin.x << ", "
                      << sdf_origin.y << ", "
                      << sdf_origin.z << ")" << std::endl;

            // 3. Читаем размер ячейки
            std::getline(file, line);
            sdf_cell_size = std::stof(line);
            std::cout << "SDF cell size: " << sdf_cell_size << std::endl;

            // 4. Рассчитываем размеры тела
            size = make_float3(
                    sdf_dims[0] * sdf_cell_size,
                    sdf_dims[1] * sdf_cell_size,
                    sdf_dims[2] * sdf_cell_size
            );

            // 5. Устанавливаем положение центра масс
            pos = initial_position;

            // 6. Корректируем начало SDF под новое положение
            sdf_origin = pos - size / 2.0f;
            std::cout << "Adjusted SDF origin: ("
                      << sdf_origin.x << ", "
                      << sdf_origin.y << ", "
                      << sdf_origin.z << ")" << std::endl;

            // 7. Читаем данные SDF
            size_t data_size = sdf_dims[0] * sdf_dims[1] * sdf_dims[2];
            std::vector<float> h_sdf;
            h_sdf.reserve(data_size);

            float value;
            while (file >> value) {
                h_sdf.push_back(value);
            }

            // Проверяем количество считанных значений
            if (h_sdf.size() != data_size) {
                std::ostringstream msg;
                msg << "Incorrect number of SDF values: expected "
                    << data_size << ", got " << h_sdf.size();
                throw std::runtime_error(msg.str());
            }

            // 8. Выводим первые 5 значений для проверки
            std::cout << "First 5 SDF values: ";
            for (int i = 0; i < 5 && i < h_sdf.size(); i++) {
                std::cout << h_sdf[i] << " ";
            }
            std::cout << std::endl;

            // 9. Выделяем память на GPU и копируем данные
            cudaError_t err = cudaMalloc(&sdf_data, data_size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed: " +
                                         std::string(cudaGetErrorString(err)));
            }

            err = cudaMemcpy(sdf_data, h_sdf.data(),
                             data_size * sizeof(float),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(sdf_data);
                sdf_data = nullptr;
                throw std::runtime_error("cudaMemcpy failed: " +
                                         std::string(cudaGetErrorString(err)));
            }

            std::cout << "SDF loaded successfully. Total values: "
                      << data_size << std::endl;
        }

        // Освобождение памяти
        void freeSDF() {
            if (sdf_data) {
                cudaFree(sdf_data);
                sdf_data = nullptr;
            }
        }

        // Функция для копирования SDF данных на хост
        void copySDFToHost() {
            size_t data_size = sdf_dims[0] * sdf_dims[1] * sdf_dims[2];
            sdf_data_host.resize(data_size);

            cudaError_t err = cudaMemcpy(
                    sdf_data_host.data(),
                    sdf_data,
                    data_size * sizeof(float),
                    cudaMemcpyDeviceToHost
            );

            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to copy SDF data to host: " +
                                         std::string(cudaGetErrorString(err)));
            }
        }

        // Генерация точек поверхности с использованием хост-данных
        void generateSurfacePoints(float density) {
            surface_points.clear();
            const float threshold = 0.0f;

            // Убедимся, что данные на хосте актуальны
            if (sdf_data_host.size() != static_cast<size_t>(sdf_dims[0] * sdf_dims[1] * sdf_dims[2])) {
                copySDFToHost();
            }

            for (int k = 0; k < sdf_dims[2]; k++) {
                for (int j = 0; j < sdf_dims[1]; j++) {
                    for (int i = 0; i < sdf_dims[0]; i++) {
                        int idx = i + j*sdf_dims[0] + k*sdf_dims[0]*sdf_dims[1];
                        if (fabs(sdf_data_host[idx]) < density) {
                            float3 pos = sdf_origin + make_float3(
                                    i * sdf_cell_size,
                                    j * sdf_cell_size,
                                    k * sdf_cell_size
                            );
                            surface_points.push_back(pos);
                        }
                    }
                }
            }
        }

        // Экспорт поверхности в формате OBJ
        void exportToOBJ(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open OBJ file: " + filename);
            }

            file << "# Rigid Body Surface\n";
            file << "# Vertices: " << surface_points.size() << "\n\n";

            // Записываем вершины
            for (const auto& p : surface_points) {
                file << "v " << p.x << " " << p.y << " " << p.z << "\n";
            }

            // Для полноценного меша нужно добавить грани,
            // но для облака точек этого достаточно
            file.close();
        }
    };

    __device__ bool contains(const RigidBody& body, float3 world_pos);



}

#endif //UTILITY_H