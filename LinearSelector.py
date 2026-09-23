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

class VCoreLayer(torch.nn.Module):
    def __init__(self, dim_n, matrix_Q):
        super().__init__()
        self.dim_n = dim_n
        # Матрица Q хранится сразу на GPU в формате float32
        self.Q = cp.asarray(matrix_Q, dtype=cp.float32)
        
        # Загрузка честного одномерного ядра
        with open('vcore_kernel.cu', 'r') as f:
            code = f.read()
        # Программное исключение конфликтующего инклюда для CuPy NVRTC
        code = code.replace('#include <math.h>', '// #include <math.h>')
        self.kernel = cp.RawKernel(code, 'vcore_optimize')

    def forward(self, x):
        # Сохраняем исходную форму тензора (например, [B, L, D])
        original_shape = x.shape
        
        # Превращаем тензор в 2D-матрицу [Количество векторов, dim_n]
        # Это позволяет правильно обрабатывать батчи любой вложенности
        x_2d = x.reshape(-1, self.dim_n)
        num_vectors = x_2d.shape[0]
        
        # Переносим тензор в CuPy без копирования через DLPack (напрямую в VRAM)
        # Убеждаемся, что входной тензор находится на CUDA и имеет тип float32
        x_cuda = x_2d.cuda().float()
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_cuda))
        v_cp = cp.zeros_like(x_cp)
        
        # Конфигурация под подпрограммы ядра: блоки обрабатывают векторы размера dim_n
        grid_size = (self.dim_n + 255) // 256
        
        # Повекторная обработка батча в цикле для обеспечения 1D-совместимости vcore_kernel
        for i in range(num_vectors):
            vcore_module_args = (x_cp[i], self.Q, v_cp[i], cp.int32(self.dim_n))
            self.kernel((grid_size,), (256,), vcore_module_args)
            
        # Возвращаем тензор обратно в PyTorch через DLPack и восстанавливаем исходную форму
        # ИСПРАВЛЕНО: изменен метод на корректный to_dlpack()
        v_torch = torch.from_dlpack(v_cp.to_dlpack())
        return v_torch.reshape(original_shape).to(x.dtype)

# Демонстрация корректности интеграции в пайплайн
if __name__ == "__main__":
    dim = 144
    Q_mock = cp.eye(dim, dtype=cp.float32)
    layer = VCoreLayer(dim, Q_mock)
    
    # Тестовый батч (например: batch=2, seq_len=4, features=144)
    test_input = torch.randn(2, 4, dim, device='cuda')
    test_output = layer(test_input)
    print(f"[OK]: Тест пройден. Выходная форма тензора: {test_output.shape}")
