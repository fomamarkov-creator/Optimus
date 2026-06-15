#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

// Константы гармоник
#define DEFAULT_ZETA 1.024f

// Прототип функции из vcore_kernel.cu
extern "C" void run_vcore_resonance(const float* X, const float* Q, float* V, int n);

/**
 * Вспомогательная функция для проверки ошибок CUDA
 */
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

/**
 * ТОЧКА ВХОДА: V-CORE ENGINE v144.1
 * Использование: vcore_engine <input_raw> <matrix_q_raw> <output_raw> <n> [zeta]
 */
int main(int argc, char* argv[]) {
    std::cout << "--- V-CORE v144: RESONANCE ENGINE ACTIVE ---" << std::endl;

    if (argc < 5) {
        std::cout << "Usage: vcore_engine <in.raw> <q.raw> <out.raw> <n> [zeta]" << std::endl;
        return 1;
    }

    // 1. Парсинг аргументов
    std::string input_path = argv[1];
    std::string q_path = argv[2];
    std::string out_path = argv[3];
    int n = std::stoi(argv[4]);
    float zeta = (argc > 5) ? std::stof(argv[5]) : DEFAULT_ZETA;

    std::cout << "[SYSTEM]: Dimension N=" << n << ", Zeta=" << zeta << std::endl;

    // 2. Выделение памяти на Хосте (CPU)
    std::vector<float> h_X(n);
    std::vector<float> h_Q(n * n);
    std::vector<float> h_V(n);

    // 3. Загрузка сырых данных (RAW)
    std::ifstream in_f(input_path, std::ios::binary);
    std::ifstream q_f(q_path, std::ios::binary);
    
    if (!in_f || !q_f) {
        std::cerr << "[ERROR]: Could not open input files." << std::endl;
        return -1;
    }

    in_f.read(reinterpret_cast<char*>(h_X.data()), n * sizeof(float));
    q_f.read(reinterpret_cast<char*>(h_Q.data()), n * n * sizeof(float));

    // 4. Подготовка GPU (Device)
    float *d_X, *d_Q, *d_V;
    checkCuda(cudaMalloc(&d_X, n * sizeof(float)));
    checkCuda(cudaMalloc(&d_Q, n * n * sizeof(float)));
    checkCuda(cudaMalloc(&d_V, n * sizeof(float)));

    checkCuda(cudaMemcpy(d_X, h_X.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_Q, h_Q.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    // 5. ЗАПУСК РЕЗОНАНСА (Вызов ядра)
    std::cout << "[EXEC]: Starting V144_CORE_RESONANCE..." << std::endl;
    
    // В CMake-сборке линкуется с vcore_kernel.cu
    run_vcore_resonance(d_X, d_Q, d_V, n);

    checkCuda(cudaDeviceSynchronize());

    // 6. Сбор результата
    checkCuda(cudaMemcpy(h_V.data(), d_V, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Сохранение результата
    std::ofstream out_f(out_path, std::ios::binary);
    out_f.write(reinterpret_cast<char*>(h_V.data()), n * sizeof(float));

    // 8. Очистка
    cudaFree(d_X); cudaFree(d_Q); cudaFree(d_V);

    std::cout << "[SUCCESS]: Resonance complete. File saved to " << out_path << std::endl;
    return 0;
}
