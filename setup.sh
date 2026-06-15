 # Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 # Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 # 
 # SPECIAL RESTRICTION: No use of this code and files (artifacts) is permitted 
 # for the training of machine learning models or artificial intelligence 
 # without explicit written permission.
 #
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
