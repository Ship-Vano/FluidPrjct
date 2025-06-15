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

__device__ float3 operator+(const float3& a, const float3& b);

__device__ float3 operator-(const float3& a, const float3& b);

__device__ float3 operator*(const float3& a, float b);

__device__ float3 operator*(float b, const float3& a);

__device__ float operator*(const float3& a, const float3& b);

__device__ float3 operator/(const float3&a, const float&b);

__device__ double3 operator+(const double3& a, const double3& b);

__device__ double3 operator-(const double3& a, const double3& b);

__device__ double3 operator*(const double3& a, float b);

__device__ float2 operator+(const float2& a, const float2& b);

__device__ float2 operator-(const float2& a, const float2& b);

__device__ float2 operator*(const float2& a, float b);


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
        float3 pos;
        float3 vel;
        float3 force;
        float mass;
//        float3x3 inertia;
//        float3x3 inv_inertia;

        // SDF данные
        float* sdf_data;          // Сырой указатель на данные
        int sdf_dims[3];          // [width, height, depth]
        float3 sdf_origin;
        float sdf_cell_size;

        // Метод для загрузки SDF
        void loadSDF(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            file.read(reinterpret_cast<char*>(sdf_dims), 3 * sizeof(int));
            file.read(reinterpret_cast<char*>(&sdf_origin), 3 * sizeof(float));
            file.read(reinterpret_cast<char*>(&sdf_cell_size), sizeof(float));

            size_t size = sdf_dims[0] * sdf_dims[1] * sdf_dims[2];
            thrust::host_vector<float> h_sdf(size);
            file.read(reinterpret_cast<char*>(h_sdf.data()), size * sizeof(float));

            // Выделяем память на устройстве
            cudaMalloc(&sdf_data, size * sizeof(float));
            cudaMemcpy(sdf_data, h_sdf.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Освобождение памяти
        void freeSDF() {
            if (sdf_data) {
                cudaFree(sdf_data);
                sdf_data = nullptr;
            }
        }
    };

    __device__ bool contains(const RigidBody& body, float3 world_pos);



}

#endif //UTILITY_H