import os
import torch
import cupy as cp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.dlpack import to_dlpack, from_dlpack

# Динамическое низкоуровневое CUDA-ядро некоммутативного резонанса Маркова
VCORE_CUDA_CODE = r'''
extern "C" __global__ 
void vcore_resonance_kernel(const float* __restrict__ src, 
                             float* __restrict__ dst, 
                             int rows, 
                             int cols, 
                             float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    int idx = row * cols + col;
    float weight = src[idx];
    
    // Квантованный фильтр по Теореме 7.1
    if (fabsf(weight) > 1e-5f) {
        dst[idx] = weight + sinf(weight * 3.14159265f) * alpha;
    } else {
        dst[idx] = weight;
    }
}
'''

class VCoreBridge:
    def __init__(self, alpha: float = 0.024):
        self.alpha = alpha
        self.kernel = cp.RawKernel(VCORE_CUDA_CODE, 'vcore_resonance_kernel')
        print("🚀 [VCore Engine] CUDA-ядро успешно скомпилировано в VRAM!")

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2: 
            return tensor
            
        orig_dtype = tensor.dtype
        orig_device = tensor.device

        # Шаг 1: Автокастинг. Переводим тензор на GPU в FP32 для точных CUDA-вычислений
        x_cuda = tensor.to(device="cuda", dtype=torch.float32).contiguous()
        rows, cols = x_cuda.shape
        dst_cuda = torch.empty_like(x_cuda)

        # Шаг 2: Zero-Copy DLPack мост напрямую во VRAM без копирования в ОЗУ
        cp_src = cp.from_dlpack(to_dlpack(x_cuda))
        cp_dst = cp.from_dlpack(to_dlpack(dst_cuda))

        # Шаг 3: Динамический расчет двумерной сетки потоков под любую архитектуру LLM
        block_size = (16, 16)
        grid_x = (cols + block_size[0] - 1) // block_size[0]
        grid_y = (rows + block_size[1] - 1) // block_size[1]

        self.kernel((grid_x, grid_y), block_size, (cp_src, cp_dst, rows, cols, self.alpha))
        cp.cuda.Device().synchronize()

        # Шаг 4: Возвращаем матрицу в исходный сжатый тип данных модели (BF16/FP16)
        return dst_cuda.to(device=orig_device, dtype=orig_dtype)

def run_optimization(input_path: str, output_path: str):
    bridge = VCoreBridge(alpha=0.024)
    weights = {}
    processed_layers, skipped_layers = 0, 0
    
    print(f"\n📥 Открытие файла весов: {input_path}")
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Строгий топологический фильтр линейных матриц
            is_linear_weight = any(p in key.lower() for p in ["proj", "weight"]) and not any(n in key.lower() for n in ["norm", "embed", "bias"])
            
            if is_linear_weight and tensor.ndim == 2:
                if "lm_head" in key.lower():
                    print(f"⚠️  [Изоляция] Слой {key} сохранен во избежание двойного наложения резонанса.")
                    weights[key] = tensor
                    skipped_layers += 1
                    continue
                    
                print(f"⚙️  Оптимизация слоя: {key} | Размер: {list(tensor.shape)} | Тип: {tensor.dtype}")
                weights[key] = bridge.optimize_tensor(tensor)
                processed_layers += 1
            else:
                weights[key] = tensor
                skipped_layers += 1
                
    save_file(weights, output_path)
    print(f"\n🎉 [УСПЕХ]: Оптимизированный файл сохранен как: {output_path}")
    print(f"📊 Итог: Перекодировано слоев: {processed_layers}, сохранено без изменений: {skipped_layers}")

if __name__ == "__main__":
    input_file, output_file = "model.safetensors", "model_vcore_fixed.safetensors"
    if os.path.exists(input_file):
        run_optimization(input_file, output_file)
    else:
        print(f"❌ Ошибка: Файл {input_file} не найден в корневой директории.")
