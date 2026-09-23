# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
import torch
import cupy as cp
import numpy as np
from safetensors.torch import load_file, save_file
import matrix_generator

with open('vcore_kernel.cu', 'r') as f:
    code = f.read().replace('#include <math.h>', '// #include <math.h>')
vcore_module = cp.RawKernel(code, 'vcore_optimize')

def apply_vcore_resonance(tensor, Q_matrix, limit_dim):
    X_cpu = tensor.detach().float().numpy()
    
    # Если это двумерная матрица весов
    if len(X_cpu.shape) == 2:
        rows, cols = X_cpu.shape
        V_out = np.zeros_like(X_cpu)
        
        # ЗАЩИТА: Если слой огромный (например, эмбеддинги 151k), обрабатываем его кусками по limit_dim
        if cols > limit_dim:
            Q_gpu = cp.asarray(Q_matrix).astype(cp.float32) # Используем максимальную доступную Q (limit_dim x limit_dim)
            grid_size = (limit_dim + 255) // 256
            
            for r in range(rows):
                # Нарезаем строку на блоки размером limit_dim
                for c_start in range(0, cols, limit_dim):
                    c_end = min(c_start + limit_dim, cols)
                    current_len = c_end - c_start
                    
                    # Подготавливаем кусок строки
                    chunk = np.zeros(limit_dim, dtype=np.float32)
                    chunk[:current_len] = X_cpu[r, c_start:c_end]
                    
                    X_row_gpu = cp.asarray(chunk)
                    V_row_gpu = cp.zeros(limit_dim, dtype=cp.float32)
                    
                    vcore_module((grid_size,), (256,), (X_row_gpu, Q_gpu, V_row_gpu, cp.int32(limit_dim)))
                    
                    V_out[r, c_start:c_end] = V_row_gpu.get()[:current_len]
            return torch.from_numpy(V_out).to(tensor.dtype)
        
        # Если слой стандартного размера (меньше лимита)
        else:
            Q_gpu = cp.asarray(Q_matrix[:cols, :cols]).astype(cp.float32)
            grid_size = (cols + 255) // 256
            for r in range(rows):
                X_row_gpu = cp.asarray(X_cpu[r]).astype(cp.float32)
                V_row_gpu = cp.zeros(cols, dtype=cp.float32)
                vcore_module((grid_size,), (256,), (X_row_gpu, Q_gpu, V_row_gpu, cp.int32(cols)))
                V_out[r] = V_row_gpu.get()
            return torch.from_numpy(V_out).to(tensor.dtype)
        
    # Если тензор одномерный (bias)
    elif len(X_cpu.shape) == 1:
        n = X_cpu.shape[0]
        if n > limit_dim:
            return tensor # Пропускаем слишком гигантские bias, если они есть
            
        X_gpu = cp.asarray(X_cpu).astype(cp.float32)
        Q_gpu = cp.asarray(Q_matrix[:n, :n]).astype(cp.float32)
        V_gpu = cp.zeros(n, dtype=cp.float32)
        
        grid_size = (n + 255) // 256
        vcore_module((grid_size,), (256,), (X_gpu, Q_gpu, V_gpu, cp.int32(n)))
        return torch.from_numpy(V_gpu.get()).to(tensor.dtype)
        
    return tensor

def main():
    print("[INIT]: Анализ файла весов model.safetensors...")
    try:
        weights = load_file("model.safetensors")
    except FileNotFoundError:
        print("[ERROR]: Файл не найден!")
        return

    # Находим максимальный размер, но ограничиваем его лимитом безопасности VRAM
    max_dim = 0
    for k, v in weights.items():
        if len(v.shape) > 0:
            current_max = max(v.shape)
            if current_max > max_dim:
                max_dim = current_max

    LIMIT_DIM = 4096  # Порог безопасности для защиты видеокарты от OutOfMemory
    target_dim = min(max_dim, LIMIT_DIM)
    print(f"-> Максимальная размерность в модели: {max_dim}. Целевой размер матрицы Q зафиксирован на: {target_dim}")

    print(f"[INIT]: Синтез марковской матрицы Q размера {target_dim}x{target_dim}...")
    Q_np = matrix_generator.generate_markov_q(target_dim)

    new_weights = {}
    for k, v in weights.items():
        print(f"Обработка резонансом: {k} | Спектр формы: {list(v.shape)}")
        new_weights[k] = apply_vcore_resonance(v, Q_np, target_dim)

    cp.cuda.Stream.null.synchronize()
    save_file(new_weights, "model_vcore_fixed.safetensors")
    print("\n=================================================================")
    print("     БЕЗОПАСНАЯ МОДИФИКАЦИЯ СЕТИ ПО МЕТОДУ МАРКОВА ЗАВЕРШЕНА     ")
    print("=================================================================")

if __name__ == "__main__":
    main()
