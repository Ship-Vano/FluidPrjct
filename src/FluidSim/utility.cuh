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

__device__ double3 operator+(const double3& a, const double3& b);

__device__ double3 operator-(const double3& a, const double3& b);

__device__ double3 operator*(const double3& a, float b);

__device__ float2 operator+(const float2& a, const float2& b);

__device__ float2 operator-(const float2& a, const float2& b);

__device__ float2 operator*(const float2& a, float b);


const int VEL_UNKNOWN = 0.0f;

namespace Utility {

    const int SOLID = 0;
    const int FLUID = 1;
    const int AIR = 2;

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





}

#endif //UTILITY_H