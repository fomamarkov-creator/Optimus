/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
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
        
        // Фильтр Маркова (устранение статики)
        if (fabsf(sum) < EPS) {
            sum = 0.0f;
        }
        
        // Резонансное ввинчивание с динамическим коэффициентом
        V[row] = sum * zeta;
    }
}
