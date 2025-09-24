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

    __device__ bool contains(float* sdf_data, float3 sdf_origin, float3 body_pos, float3 local_com, float* rotation_matrix, float3 world_pos, float sdf_cell_size, int sdf_w, int sdf_h, int sdf_d){
        // 1. Преобразование в локальные координаты тела
        float3 p_rel = world_pos - body_pos;

        // 2. Обратное вращение: p_local_centered = R^T * p_rel
        float3 p_local_centered;
        p_local_centered.x = rotation_matrix[0] * p_rel.x
                           + rotation_matrix[3] * p_rel.y
                           + rotation_matrix[6] * p_rel.z;

        p_local_centered.y = rotation_matrix[1] * p_rel.x
                           + rotation_matrix[4] * p_rel.y
                           + rotation_matrix[7] * p_rel.z;

        p_local_centered.z = rotation_matrix[2] * p_rel.x
                           + rotation_matrix[5] * p_rel.y
                           + rotation_matrix[8] * p_rel.z;

        // 3. Переход в SDF-систему: добавляем локальный центр масс
        float3 p_local = p_local_centered + local_com;

        // 4. Преобразование в координаты SDF сетки
        float3 sdf_coord = {
            (p_local.x - sdf_origin.x) / sdf_cell_size,
            (p_local.y - sdf_origin.y) / sdf_cell_size,
            (p_local.z - sdf_origin.z) / sdf_cell_size
        };
        // 5. Проверка границ сетки
        if (sdf_coord.x < 0 || sdf_coord.x >= sdf_w ||
            sdf_coord.y < 0 || sdf_coord.y >= sdf_h ||
            sdf_coord.z < 0 || sdf_coord.z >= sdf_d)
        {
            return false;
        }

        // 6. Определение индексов
        int i = min(static_cast<int>(sdf_coord.x), sdf_w - 1);
        int j = min(static_cast<int>(sdf_coord.y), sdf_h - 1);
        int k = min(static_cast<int>(sdf_coord.z), sdf_d - 1);

        // 7. Проверка значения SDF
        int idx = i + j * sdf_w + k * sdf_w * sdf_h;
        return sdf_data[idx] <= 0.0f;
    }

    __host__ __device__ float3 cross(const float3& a, const float3& b){
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

    __host__ __device__ float4 multiply_quaternions(const float4& a, const float4& b) {
        return {
                a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
        };
    }

    // Нормализация кватерниона
    __host__ __device__ void normalize_quaternion(float4& q) {
        float len = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        if (len > 1e-6f) {
            q.x /= len;
            q.y /= len;
            q.z /= len;
            q.w /= len;
        }
    }

    // Преобразование кватерниона в корабельные углы
    __host__ __device__ float3 quaternion_to_ship_angles(const float4& q) {
        // Крен (roll, φ) - вращение вокруг оси X
        // Тангаж (pitch, θ) - вращение вокруг оси Y
        // Рыскание (yaw, ψ) - вращение вокруг оси Z

        // Используем формулу для преобразования кватерниона в углы Эйлера (Z-Y-X)
        float roll, pitch, yaw;

        // Рыскание (yaw) - ψ
        float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
        float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
        yaw = atan2(siny_cosp, cosy_cosp);

        // Тангаж (pitch) - θ
        float sinp = 2.0f * (q.w * q.y - q.z * q.x);
        if (fabs(sinp) >= 1.0f) {
            // Используем 90 градусов, если значение выходит за пределы
            pitch = copysign(M_PI / 2.0f, sinp);
        } else {
            pitch = asin(sinp);
        }

        // Крен (roll) - φ
        float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
        float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
        roll = atan2(sinr_cosp, cosr_cosp);

        return make_float3(roll, pitch, yaw);
    }

}