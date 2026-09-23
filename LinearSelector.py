import torch
import torch.nn as nn
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

# Низкоуровневое ядро линейного селектора по Теореме 3.2
SELECTOR_CUDA_CODE = r'''
extern "C" __global__
void vcore_selector_kernel(const float* __restrict__ X, 
                           float* __restrict__ Y, 
                           int rows, 
                           int cols, 
                           float theta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float val = X[idx];
    
    // Линейный селектор жесткого усечения числового хаоса корпораций
    if (fabsf(val) > theta) {
        Y[idx] = val;
    } else {
        Y[idx] = 0.0f;
    }
}
'''

class LinearSelectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta):
        orig_shape = x.shape
        orig_dtype = x.dtype
        orig_device = x.device
        
        # Выпрямляем тензор в 2D матрицу для CUDA-обработки
        x_flat = x.view(-1, orig_shape[-1]).contiguous()
        
        # АВТОКАСТИНГ: Если веса в BF16/FP16, временно переводим в FP32 для точного ядра
        if x_flat.dtype != torch.float32:
            x_cuda = x_flat.to(device="cuda", dtype=torch.float32)
        else:
            x_cuda = x_flat.to("cuda")
            
        rows, cols = x_cuda.shape
        y_cuda = torch.empty_like(x_cuda)

        # Компилируем и вызываем ядро через высокоскоростной DLPack
        kernel = cp.RawKernel(SELECTOR_CUDA_CODE, 'vcore_selector_kernel')
        
        cp_X = cp.from_dlpack(to_dlpack(x_cuda))
        cp_Y = cp.from_dlpack(to_dlpack(y_cuda))

        block_size = (16, 16)
        grid_x = (cols + block_size[0] - 1) // block_size[0]
        grid_y = (rows + block_size[1] - 1) // block_size[1]

        kernel((grid_x, grid_y), block_size, (cp_X, cp_Y, rows, cols, float(theta)))
        cp.cuda.Device().synchronize()

        # Возвращаем тензор в исходный формат (BF16/FP16) и исходный шейп
        y_flat = y_cuda.to(device=orig_device, dtype=orig_dtype)
        
        ctx.save_for_backward(y_flat)
        return y_flat.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        y_flat, = ctx.saved_tensors
        # Градиенты свободно текут только там, где селектор оставил веса активными
        grad_input = grad_output.clone()
        grad_input[y_flat.view(grad_output.shape) == 0.0f] = 0.0f
        return grad_input, None

class LinearSelector(nn.Module):
    def __init__(self, theta: float = 1e-4):
        super().__init__()
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # В режиме инференса (когда срываем маски) используем стабильный быстрый граф
        if not self.training or not x.requires_grad:
            return torch.where(torch.abs(x) > self.theta, x, torch.zeros_like(x))
        # В режиме обучения подключаем кастомный Autograd-движок CUDA
        return LinearSelectorFunction.apply(x, self.theta)
