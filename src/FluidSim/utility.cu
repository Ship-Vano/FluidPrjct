#include "utility.cuh"

__host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator*(float b, const float3& a){
    return a * b;
}

__host__ __device__ float operator*(const float3& a, const float3& b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ float3 operator/(const float3&a, const float&b){
    return a * (1.0f/b);
}

__host__ __device__ double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ double3 operator-(const double3& a, const double3& b) {
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

    float3 getGridCellPosition(float i, float j, float k, float dx) {
        return make_float3((i +0.5f)* dx, (j + 0.5f) * dx, (k + 0.5f) * dx);
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

    void save3dParticlesToPLY(const thrust::host_vector<Particle3D>& particles,
                              const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        file << "ply\n"
             << "format ascii 1.0\n"
             << "element vertex " << particles.size() << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "end_header\n";

        for (const auto p: particles) {
            file << p.pos.x << " " << p.pos.y << " " << p.pos.z << "\n";
        }
    }

    void save3dParticlesToOFF(const thrust::host_vector<Particle3D>& particles,
                              const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Заголовок формата OFF
        file << "OFF\n";
        file << particles.size() << " 0 0\n"; // Вершины, грани (0), рёбра (0)

        // Запись координат частиц
        for (const auto& p : particles) {
            file << p.pos.x << " " << p.pos.y << " " << p.pos.z << "\n";
        }
    }

    int2 getGridCellIndex(float2 pos, float dx){
        return make_int2((int)(pos.x / dx), (int)(pos.y/dx));
    }

    __device__ int getGridCellIndex_device(float2 pos, float dx, int gridWidth){
        return (int)(pos.x / dx)  + (int)(pos.y / dx) * gridWidth;
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

    __device__ int2 getGridIndicesU(int ind, int gridWidth) {
        return {ind % (gridWidth+1), ind / (gridWidth+1)}; // i, j для u-компоненты
    }

    __device__ int2 getGridIndicesV(int ind, int gridWidth) {
        return {ind % (gridWidth), ind / (gridWidth)}; // i, j для v-компоненты
    }

    __device__ bool contains(float* sdf_data, float3 sdf_origin, float3 world_pos, float sdf_cell_size, int sdf_w, int sdf_h, int sdf_d) {
        float3 local_pos = (world_pos - sdf_origin) / sdf_cell_size;
        int i = static_cast<int>(local_pos.x);
        int j = static_cast<int>(local_pos.y);
        int k = static_cast<int>(local_pos.z);

        if (i < 0 || i >= sdf_w ||
            j < 0 || j >= sdf_h ||
            k < 0 || k >= sdf_d) return false;

        int idx = i + j * sdf_w + k * sdf_w * sdf_h;
        return sdf_data[idx] <= 0.0f;
    }

    __device__ float3 cross(const float3& a, const float3& b){
        return make_float3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
        );
    }

    __device__ float sampleBody(float3 bodyVel, float3 bodyOmega, float3 bodyCM, float3 facePos, float3 normal){
        float3 vel = bodyVel + cross(bodyOmega, facePos - bodyCM);
        return vel * normal;
    }

}