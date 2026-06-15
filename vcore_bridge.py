import torch
import cupy as cp
from safetensors.torch import load_file, save_file

# Загружаем наше "правильное" ядро
with open('vcore_kernel.cu', 'r') as f:
    code = f.read()
vcore_module = cp.RawKernel(code, 'vcore_optimize')

def apply_vcore_resonance(tensor, Q_matrix):
    n = tensor.numel()
    X_gpu = cp.asarray(tensor.float())
    Q_gpu = cp.asarray(Q_matrix)
    V_gpu = cp.zeros_like(X_gpu)
    
    # Запуск "мозга"
    vcore_module(((n + 255) // 256,), (256,), (X_gpu, Q_gpu, V_gpu, n))
    
    return torch.as_tensor(V_gpu, device='cpu')

# Пример работы
weights = load_file("model.safetensors")
Q = cp.eye(1024) # Здесь должна быть ваша рассчитанная матрица Q
new_weights = {k: apply_vcore_resonance(v, Q) for k, v in weights.items()}
save_file(new_weights, "model_vcore_fixed.safetensors")
