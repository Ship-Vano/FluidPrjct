#ifndef UTILITY_H
#define UTILITY_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusparse.h>
#include "cudss.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>


#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <iomanip>
#include <stdio.h>

__device__ float3 operator+(const float3& a, const float3& b);

__device__ float3 operator-(const float3& a, const float3& b);

__device__ float3 operator*(const float3& a, float b);

__device__ double3 operator+(const double3& a, const double3& b);

__device__ double3 operator-(const double3& a, const double3& b);

__device__ double3 operator*(const double3& a, float b);

__device__ float2 operator+(const float2& a, const float2& b);

__device__ float2 operator-(const float2& a, const float2& b);

__device__ float2 operator*(const float2& a, float b);


const int VEL_UNKNOWN = INT_MIN;

namespace Utility {

    const int SOLID = 0;
    const int FLUID = 1;
    const int AIR = 2;

    struct Particle2D {
        float2 pos;
        float2 vel;
        Particle2D(float2 pos, float2 vel): pos(pos), vel(vel){}
    };

    float2 getGridCellPosition(float i, float j, float dx);
    int2 getGridCellIndex(float2 pos, float dx);

    void saveParticlesToFile(const std::vector<Particle2D>& particles,
                             const std::string& filename);
    void saveParticlesToPLY(const std::vector<Particle2D>& particles,
                            const std::string& filename);


    __device__ int getGridCellIndex_device(float2 pos, float dx, int gridWidth);

    __device__ float2 getGridCellPosition_device(float i, float j, float dx);

    __device__ float bilinearHatKernel(float2 dist, float dx, float dy);

    __device__ float hatFunction(float r);

    __device__ int2 getGridIndicesU(int ind, int gridWidth);

    __device__ int2 getGridIndicesV(int ind, int gridWidth);


    // matrixes and solvers
    struct CSRMatrix {
        int rows;           // Число строк (равно количеству fluid-ячеек)
        int cols;           // Число столбцов (равно количеству fluid-ячеек)
        int nnz;            // Количество ненулевых элементов
        double* values;     // Массив значений (на GPU)
        int* row_ptr;       // Массив указателей на строки (на GPU)
        int* col_ind;       // Массив индексов столбцов (на GPU)
    };

}
#endif //UTILITY_H