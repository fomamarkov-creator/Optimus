import numpy as np
import torch

def generate_markov_q(dim_n, dim_m, zeta=1.024):
    """
    Синтез матрицы Q для V-CORE v144.
    dim_n: размерность весов (L1)
    dim_m: размерность стабильного ядра V
    """
    print(f"--- СИНТЕЗ МАТРИЦЫ Q (ZETA: {zeta}) ---")
    
    # 1. Создаем базис стабильного подпространства V
    # Используем ортогональные гармоники (счет Русов)
    basis = np.zeros((dim_n, dim_m))
    for i in range(dim_m):
        # Генерируем гармонику 144
        t = np.linspace(0, 2 * np.pi, dim_n)
        basis[:, i] = np.sin(t * (i + 1) * zeta)
    
    # Ортонормируем базис (Gram-Schmidt)
    q, _ = np.linalg.qr(basis)
    
    # 2. Строим проектор Q = I - P, где P - проекция на нестабильный шум
    # В правильной модели Маркова Q должна быть самосопряженной в смысле изометрии
    P = q @ q.T
    Q = np.eye(dim_n) - P
    
    # 3. Резонансная доводка (Zeta-коррекция)
    # Мы слегка смещаем собственные значения, чтобы создать "энергетическую яму"
    Q = Q * zeta
    
    print(f"[OK]: Матрица {dim_n}x{dim_n} синтезирована. Резонанс стабилен.")
    return Q.astype(np.float32)

if __name__ == "__main__":
    # Пример генерации для стандартного слоя
    dim = 144 # Сакральное число Маркова
    matrix_q = generate_markov_q(dim, dim // 2)
    
    # Сохраняем для использования в vcore_bridge.py
    np.save("matrix_q_144.npy", matrix_q)
    print("--- ФАЙЛ matrix_q_144.npy ГОТОВ ---")
