/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 */

#include <cuda_runtime.h>

#define EPS 1e-6f

extern "C" __global__
void vcore_optimize(const float* X, const float* Q, float* V, int n, float zeta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int i = 0; i < n; ++i) {
            sum += Q[row * n + i] * X[i];
        }
        
        // Фильтр Маркова
        if (fabsf(sum) < EPS) {
            sum = 0.0f;
        }
        
        // ИСПРАВЛЕНО: Теперь zeta передается динамически, защищая от взрыва активаций
        V[row] = sum * zeta;
    }
}
