#include <cuda_runtime.h>
#include <math.h>

#define ZETA 1.024f
#define EPS 1e-6f

extern "C" __global__
void vcore_optimize(const float* X, const float* Q, float* V, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        // Линейная селекция
        for (int i = 0; i < n; ++i) {
            sum += Q[row * n + i] * X[i];
        }
        
        // Фильтр Маркова (устранение статики)
        if (fabsf(sum) < EPS) sum = 0.0f;
        
        // Резонансное ввинчивание
        V[row] = sum * ZETA;
    }
}
