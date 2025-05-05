#include "test.cuh"

__device__ float3 calc(){
    float3 vec1{1.0f, 2.0f, 3.0f};
    float3 vec2{1.0f, 2.0f, 3.0f};
    float3 vec3 = vec1 + vec2;
    return vec3;
}

__global__ void run_calc_kernel(float3* result) {
    *result = calc();  // Вызов device-функции внутри kernel
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}


 void out() {
     float3* d_result;  // Указатель на память GPU
     cudaError_t err = cudaMalloc(&d_result, sizeof(float3));
     checkCudaError(err, "cudaMalloc failed");

     // Запуск kernel (1 блок, 1 поток)
     run_calc_kernel<<<1, 1>>>(d_result);

     float3 vec;
     cudaMemcpy(&vec, d_result, sizeof(float3), cudaMemcpyDeviceToHost);

     std::cout << "Vec: (" << vec.x << ", " << vec.y << ", " << vec.z << ")" << std::endl;
     cudaFree(d_result);

    std::cout << "Hello, World!" << std::endl;


     // 1. Инициализация cuSPARSE
     cusparseHandle_t handle;
     cusparseCreate(&handle);

     // 2. Данные матрицы (CSR-формат)
     int n = 3;  // Размер матрицы 3x3
     float h_values[] = {1.0f, 2.0f, 3.0f};  // Ненулевые элементы
     int h_col_ind[] = {0, 1, 2};            // Столбцы
     int h_row_ptr[] = {0, 1, 2, 3};         // Индексы строк

     // 3. Копируем данные на GPU
     float* d_values;
     int* d_col_ind, *d_row_ptr;
     cudaMalloc(&d_values, 3 * sizeof(float));
     cudaMalloc(&d_col_ind, 3 * sizeof(int));
     cudaMalloc(&d_row_ptr, 4 * sizeof(int));

     cudaMemcpy(d_values, h_values, 3 * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_col_ind, h_col_ind, 3 * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_row_ptr, h_row_ptr, 4 * sizeof(int), cudaMemcpyHostToDevice);

     // 4. Создаем дескриптор матрицы
     cusparseSpMatDescr_t matA;
     cusparseCreateCsr(&matA,
                       n, n, 3,  // rows, cols, nnz
                       d_row_ptr, d_col_ind, d_values,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

     // 5. Создаем дескрипторы векторов
     float* d_x, *d_y;
     cudaMalloc(&d_x, n * sizeof(float));
     cudaMalloc(&d_y, n * sizeof(float));

     float h_x[] = {1.0f, 1.0f, 1.0f};  // Входной вектор
     cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

     cusparseDnVecDescr_t vecX, vecY;
     cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F);
     cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_32F);

     // 6. Настройка операции SpMV
     float alpha = 1.0f, beta = 0.0f;
     size_t bufferSize;  // Объявляем переменную для размера буфера
     cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matA, vecX, &beta, vecY,
                             CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

     void* buffer;
     cudaMalloc(&buffer, bufferSize);

     // 7. Умножение матрицы на вектор: y = alpha * A * x + beta * y
     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, matA, vecX, &beta, vecY,
                  CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

     // 8. Копируем результат на CPU
     float h_y[3];
     cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
     printf("Result: %f, %f, %f\n", h_y[0], h_y[1], h_y[2]);

     // 9. Освобождаем ресурсы
     cusparseDestroySpMat(matA);
     cusparseDestroyDnVec(vecX);
     cusparseDestroyDnVec(vecY);
     cusparseDestroy(handle);
     cudaFree(d_values);
     cudaFree(d_col_ind);
     cudaFree(d_row_ptr);
     cudaFree(d_x);
     cudaFree(d_y);
     cudaFree(buffer);
}

