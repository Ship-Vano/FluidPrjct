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
         // Generate 32M random numbers serially.
         thrust::default_random_engine rng(1337);
         thrust::uniform_int_distribution<int> dist;
         thrust::host_vector<int> h_vec(32 << 20);
         thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

    // Transfer data to the device.
         thrust::device_vector<int> d_vec = h_vec;

    // Sort data on the device.
         thrust::sort(d_vec.begin(), d_vec.end());

    // Transfer data back to host.
         thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

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

     /*
      * cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Пример симметричной матрицы 3x3 (хранится только верхняя часть)
    // A = [1 2 3
    //      2 4 5
    //      3 5 6]
    // Верхняя часть в CSR:
    int n = 3; // размер матрицы
    int nnz = 6; // ненулевых элементов в верхней части

    // CSR-данные (только верхняя треугольная часть)
    float h_values[] = {1, 2, 3, 4, 5, 6};
    int h_col_ind[] = {0, 1, 2, 1, 2, 2};
    int h_row_ptr[] = {0, 3, 5, 6};

    // Копируем данные на GPU
    float* d_values;
    int* d_col_ind;
    int* d_row_ptr;
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_col_ind, nnz * sizeof(int));
    cudaMalloc(&d_row_ptr, (n+1) * sizeof(int));

    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);

    // Создаем дескриптор матрицы
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
                     n, n, nnz,
                     d_row_ptr,
                     d_col_ind,
                     d_values,
                     CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO,
                     CUDA_R_32F);

    // Устанавливаем свойства матрицы
    cusparseSetMatType(matA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(matA, CUSPARSE_FILL_MODE_UPPER); // храним верхнюю часть

    // Теперь можно использовать matA в операциях cuSPARSE
    // Например, для умножения матрицы на вектор:
    // cusparseSpMV(handle, op, alpha, matA, vecX, beta, vecY, ...)

    // Очистка
    cusparseDestroySpMat(matA);
    cudaFree(d_values);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);
    cusparseDestroy(handle);
      *
      * */
}


#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("CALL FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("CALL FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);
#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(b_values_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d); \
    } while(0);



int cuDSStest(){
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix \n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;
    int nnz = 8;
    int nrhs = 1;

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (double*)malloc(nnz * sizeof(double));
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    b_values_h = (double*)malloc(nrhs * n * sizeof(double));

    if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for A and b */
    int i = 0;
    csr_offsets_h[i++] = 0;
    csr_offsets_h[i++] = 2;
    csr_offsets_h[i++] = 4;
    csr_offsets_h[i++] = 6;
    csr_offsets_h[i++] = 7;
    csr_offsets_h[i++] = 8;

    i = 0;
    csr_columns_h[i++] = 0; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 1; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 2; csr_columns_h[i++] = 4;
    csr_columns_h[i++] = 3;
    csr_columns_h[i++] = 4;

    i = 0;
    csr_values_h[i++] = 4.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0; csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 5.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 2.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                                   cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                                              base), status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                                      A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                                      A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                                   cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i+1));
        if (fabs(x_values_h[i] - (i + 1)) > 2.e-15)
            passed = 0;
    }

    /* Release the data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed)
        printf("Example PASSED\n");
    else
        printf("Example FAILED\n");

    return 0;
}

