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
import matrix_generator  # ИМПОРТ ВАШЕГО ГЕНЕРАТОРА

# Загружаем наше "правильное" ядро
with open('vcore_kernel.cu', 'r') as f:
    code = f.read()
code = code.replace('#include <math.h>', '// #include <math.h>')
vcore_module = cp.RawKernel(code, 'vcore_optimize')

def apply_vcore_resonance(tensor, Q_matrix):
    X_cpu = tensor.detach().float().numpy()
    
    # Если это двумерная матрица весов (например, Linear слой)
    if len(X_cpu.shape) == 2:
        rows, cols = X_cpu.shape
        V_out = np.zeros_like(X_cpu)
        
        # Обрезаем матрицу Q строго под текущий размер столбцов слоя (cols)
        Q_gpu = cp.asarray(Q_matrix[:cols, :cols]).astype(cp.float32)
        
        # Обрабатываем матрицу весов построчно
        for r in range(rows):
            X_row_gpu = cp.asarray(X_cpu[r]).astype(cp.float32)
            V_row_gpu = cp.zeros(cols, dtype=cp.float32)
            
            grid_size = (cols + 255) // 256
            vcore_module((grid_size,), (256,), (X_row_gpu, Q_gpu, V_row_gpu, cp.int32(cols)))
            
            V_out[r] = V_row_gpu.get()
            
        return torch.from_numpy(V_out).to(tensor.dtype)
        
    # Если тензор одномерный (bias)
    elif len(X_cpu.shape) == 1:
        n = X_cpu.shape[0]
        X_gpu = cp.asarray(X_cpu).astype(cp.float32)
        Q_gpu = cp.asarray(Q_matrix[:n, :n]).astype(cp.float32)
        V_gpu = cp.zeros(n, dtype=cp.float32)
        
        grid_size = (n + 255) // 256
        vcore_module((grid_size,), (256,), (X_gpu, Q_gpu, V_gpu, cp.int32(n)))
        
        return torch.from_numpy(V_gpu.get()).to(tensor.dtype)
        
    return tensor

# 1. Загрузка весов модели
print("[INIT]: Анализ файла весов model.safetensors...")
weights = load_file("model.safetensors")

# 2. Автоопределение максимальной размерности
max_dim = 0
for k, v in weights.items():
    if len(v.shape) > 0:
        current_max = max(v.shape)
        if current_max > max_dim:
            max_dim = current_max

print(f"-> Максимальная обнаруженная размерность слоя: {max_dim}")

# 3. ИНТЕГРАЦИЯ: Генерация настоящей марковской матрицы Q
print(f"[INIT]: Запуск генератора Маркова для матрицы {max_dim}x{max_dim}...")
Q_np = matrix_generator.generate_markov_q(max_dim, max_dim)
Q = cp.asarray(Q_np).astype(cp.float32)

# 4. Запуск резонансной обработки весов
new_weights = {}
for k, v in weights.items():
    print(f"Обработка резонансом: {k} | Спектр формы: {list(v.shape)}")
    new_weights[k] = apply_vcore_resonance(v, Q)

# 5. Сохранение результатов
cp.cuda.Stream.null.synchronize()
save_file(new_weights, "model_vcore_fixed.safetensors")

print("\n=================================================================")
print("     КОРРЕКТНАЯ МОДИФИКАЦИЯ СЕТИ ПО МЕТОДУ МАРКОВА ЗАВЕРШЕНА     ")
print("=================================================================")
