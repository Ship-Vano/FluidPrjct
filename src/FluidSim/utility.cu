#include "utility.cuh"

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double3 operator*(const double3& a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__device__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}



namespace Utility {
    float2 getGridCellPosition(float i, float j, float dx) {
        return make_float2((i +0.5f)* dx, (j + 0.5f) * dx);
    }

    void saveParticlesToFile(const std::vector<Particle2D>& particles,
                             const std::string& filename){
        std::ofstream file(filename, std::ios::binary);

        // Просто записываем x и y как два отдельных float
        for (const auto& p : particles) {
        float x = p.pos.x;
        float y = p.pos.y;
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
            }
    }
    void saveParticlesToPLY(const std::vector<Particle2D>& particles,
                            const std::string& filename) {
        std::ofstream file(filename);

        file << "ply\n"
             << "format ascii 1.0\n"
             << "element vertex " << particles.size() << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "end_header\n";

        for (const auto p: particles) {
            file << p.pos.x << " " << p.pos.y << " " << 0.0 << "\n";
        }
    }

    int2 getGridCellIndex(float2 pos, float dx){
        return make_int2((int)(pos.x / dx), (int)(pos.y/dx));
    }

    __device__ int getGridCellIndex_device(float2 pos, float dx, int gridHeight){
        return (int)(pos.x / dx) * gridHeight + (int)(pos.y / dx);
    }


    __device__ float2 getGridCellPosition_device(float i, float j, float dx){
        return float2{(i+0.5f)*dx, (j+0.5f)*dx};
    }

    __device__ float bilinearHatKernel(float2 dist, float dx, float dy){
        return hatFunction(dist.x / dx) * hatFunction(dist.y / dy);
    }

    __device__ float hatFunction(float r){
        float rAbs = fabs(r);
        if(rAbs-1.0f <= 1e-8){ //if(rAbs <= 1.0)
            return 1.0f - rAbs;
        } else{
            return 0.0f;
        }
    }

    __device__ int2 getGridIndicesU(int ind, int gridHeight) {
        return {ind / gridHeight, ind % gridHeight}; // i, j для u-компоненты
    }

    __device__ int2 getGridIndicesV(int ind, int gridHeight) {
        return {ind / (gridHeight + 1), ind % (gridHeight + 1)}; // i, j для v-компоненты
    }


}