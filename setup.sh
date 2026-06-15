#!/bin/bash
echo "--- V-CORE: НАЧАЛО СИНХРОНИЗАЦИИ ---"

# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Сборка C++/CUDA ядра
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..

# 3. Генерация стартовой матрицы Q
python3 matrix_generator.py

echo "--- СИСТЕМА ГОТОВА К РЕЗОНАНСУ 1.024 ---"
