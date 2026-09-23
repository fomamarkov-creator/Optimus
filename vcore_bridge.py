# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# 
# SPECIAL RESTRICTION: No use of this code and files (artifacts) is permitted 
# for the training of machine learning models or artificial intelligence 
# without explicit written permission.
# 
# COMMERCIAL CLAUSE: Any enterprise deployment requires a paid commercial license.
# Full license text is available in the LICENSE file in the root directory.
import torch
import cupy as cp
import numpy as np
from safetensors.torch import load_file, save_file
import matrix_generator  # Используем обновленный генератор

# Загружаем наше "правильное" ядро
print("[INIT]: Компиляция vcore_kernel.cu...")
with open('vcore_kernel.cu', 'r') as f:
    code = f.read()
# Очищаем инклюд, если он вызывает конфликты внутри CuPy NVRTC
code = code.replace('#include <math.h>', '// #include <math.h>')
vcore_module = cp.RawKernel(code, 'vcore_optimize')

def apply_vcore_resonance(tensor, Q_matrix):
    # Работаем строго во float32, чтобы данные совпадали с const float* ядра
    X_cpu = tensor.detach().float().numpy()
    
    # Если это двумерная матрица весов (например, Linear слои)
    if len(X_cpu.shape) == 2:
        rows, cols = X_cpu.shape
        V_out = np.zeros_like(X_cpu)
        
        # Обрезаем матрицу Q строго под текущую ширину слоя (размер вектора cols)
        Q_gpu = cp.asarray(Q_matrix[:cols, :cols]).astype(cp.float32)
        
        # Обрабатываем матрицу весов построчно, как требует ваше 1D-ядро
        for r in range(rows):
            X_row_gpu = cp.asarray(X_cpu[r]).astype(cp.float32)
            V_row_gpu = cp.zeros(cols, dtype=cp.float32)
            
            # Конфигурация строго под ваше 1D ядро: блоки по 256 потоков
            grid_size = (cols + 255) // 256
            vcore_module((grid_size,), (256,), (X_row_gpu, Q_gpu, V_row_gpu, cp.int32(cols)))
            
            # Извлекаем результат обработки строки
            V_out[r] = V_row_gpu.get()
            
        return torch.from_numpy(V_out).to(tensor.dtype)
        
    # Если тензор одномерный (например, bias), обрабатываем его как один вектор
    elif len(X_cpu.shape) == 1:
        n = X_cpu.shape[0]
        X_gpu = cp.asarray(X_cpu).astype(cp.float32)
        Q_gpu = cp.asarray(Q_matrix[:n, :n]).astype(cp.float32)
        V_gpu = cp.zeros(n, dtype=cp.float32)
        
        grid_size = (n + 255) // 256
        vcore_module((grid_size,), (256,), (X_gpu, Q_gpu, V_gpu, cp.int32(n)))
        
        return torch.from_numpy(V_gpu.get()).to(tensor.dtype)
        
    return tensor

def main():
    # 1. Загрузка оригинальных весов модели
    print("[LOAD]: Анализ структуры файла весов model.safetensors...")
    try:
        weights = load_file("model.safetensors")
    except FileNotFoundError:
        print("[ERROR]: Файл 'model.safetensors' не найден в текущей директории!")
        return

    # 2. Автоматическое определение максимальной размерности
    max_dim = 0
    for k, v in weights.items():
        if len(v.shape) > 0:
            current_max = max(v.shape)
            if current_max > max_dim:
                max_dim = current_max

    print(f"-> Максимальная обнаруженная спектральная размерность: {max_dim}")

    # 3. Синтез глобальной марковской матрицы Q на основе обновленного генератора
    print(f"[MATH]: Запуск генератора для размерности {max_dim}x{max_dim}...")
    # dim_m выставится внутри автоматически как max_dim // 2 для защиты от обнуления
    Q_np = matrix_generator.generate_markov_q(max_dim)
    
    # 4. Цикл резонансной обработки весов
    print("\n[RUN]: Накачка весов квантовым оператором V-CORE...")
    new_weights = {}
    for k, v in weights.items():
        print(f" ➔ Обработка слоя: {k:50} | Форма: {str(list(v.shape)):15}")
        new_weights[k] = apply_vcore_resonance(v, Q_np)

    # 5. Синхронизация GPU потока и сохранение новой модели
    print("\n[SAVE]: Финализация данных и запись на диск...")
    cp.cuda.Stream.null.synchronize()
    save_file(new_weights, "model_vcore_fixed.safetensors")

    print("\n=================================================================")
    print("     КОРРЕКТНАЯ МОДИФИКАЦИЯ СЕТИ ПО МЕТОДУ МАРКОВА ЗАВЕРШЕНА     ")
    print("                 ВЕСА И СТРУКТУРА ЗАЩИЩЕНЫ                     ")
    print("=================================================================")

if __name__ == "__main__":
    main()
