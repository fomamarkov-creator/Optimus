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
        # Матрица Q хранится сразу на GPU
        self.Q = cp.asarray(matrix_Q, dtype=cp.float32)
        
        # Загрузка нашего честного ядра
        with open('vcore_kernel.cu', 'r') as f:
            self.kernel = cp.RawKernel(f.read(), 'vcore_optimize')

    def forward(self, x):
        # Прямая работа с тензорами PyTorch на GPU через DLPack
        shape = x.shape
        x_flat = x.flatten()
        n = x_flat.numel()
        
        # Конвертация без копирования в оперативку
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_flat.cuda()))
        v_cp = cp.zeros_like(x_cp)
        
        # Запуск резонанса
        self.kernel(((n + 255) // 256,), (256,), (x_cp, self.Q, v_cp, n))
        
        # Возврат в PyTorch
        return torch.from_dlpack(v_cp.toDlpack()).reshape(shape)
