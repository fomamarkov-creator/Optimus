#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

extern "C" {
    // Импорт нашего "правильного" ядра
    void run_vcore_resonance(const float* X, const float* Q, float* V, int n) {
        float *d_X, *d_Q, *d_V;
        
        // 1. Аллокация памяти без утечек
        cudaMalloc(&d_X, n * sizeof(float));
        cudaMalloc(&d_Q, n * n * sizeof(float));
        cudaMalloc(&d_V, n * sizeof(float));

        // 2. Перенос данных
        cudaMemcpy(d_X, X, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, Q, n * n * sizeof(float), cudaMemcpyHostToDevice);

        // 3. Конфигурация запуска (144 гармоники / потоки)
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        
        // Здесь вызывается ядро из vcore_kernel.cu
        // vcore_optimize<<<blocks, threads>>>(d_X, d_Q, d_V, n);

        cudaDeviceSynchronize();
        cudaMemcpy(V, d_V, n * sizeof(float), cudaMemcpyDeviceToHost);

        // 4. Чистка
        cudaFree(d_X); cudaFree(d_Q); cudaFree(d_V);
    }
}
