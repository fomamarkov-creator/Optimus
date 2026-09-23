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
from safetensors.torch import load_file, save_file

# Загружаем наше "правильное" ядро
with open('vcore_kernel.cu', 'r') as f:
    code = f.read()
vcore_module = cp.RawKernel(code, 'vcore_optimize')

def apply_vcore_resonance(tensor, Q_matrix):
    # Проверяем, что тензор двумерный (матрица весов)
    if len(tensor.shape) != 2:
        # Если тензор одномерный (например, bias), возвращаем как есть или обрабатываем иначе
        return tensor.cpu()
        
    size = tensor.shape[0] # Предполагаем квадратную матрицу согласно архитектуре V-Core (size x size)
    
    # Конвертируем тензоры в CuPy массивы (желательно сразу в float16, как в бенчмарке)
    X_gpu = cp.asarray(tensor.numpy()).astype(cp.float16)
    Q_gpu = cp.asarray(Q_matrix).astype(cp.float16)
    V_gpu = cp.zeros_like(X_gpu)
    
    # ИССПРАВЛЕНО: Приведение к 2D-сетке блоков (16х16) в соответствии с vcore_kernel.cu
    grid_x = (size + 15) // 16
    grid_y = (size + 15) // 16
    
    # Запуск "мозга" с корректной геометрией и передачей оригинального size вместо numel()
    vcore_module((grid_x, grid_y), (16, 16), (X_gpu, Q_gpu, V_gpu, cp.int32(size)))
    
    # Синхронизируем поток перед возвратом данных
    cp.cuda.Stream.null.synchronize()
    
    # Преобразуем CuPy массив обратно в PyTorch тензор на CPU для сохранения
    return torch.from_numpy(V_gpu.get())

# Пример работы
weights = load_file("model.safetensors")

# Инициализируем матрицу оператора Q под размерность слоев
Q = cp.eye(1024) 

# Запускаем резонансную обработку весов
new_weights = {}
for k, v in weights.items():
    print(f"Processing layer: {k} | Shape: {v.shape}")
    new_weights[k] = apply_vcore_resonance(v, Q)

save_file(new_weights, "model_vcore_fixed.safetensors")
print("\n[SUCCESS]: Резонансная модификация весов успешно завершена.")
