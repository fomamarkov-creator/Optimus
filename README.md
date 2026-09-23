## Scientific Educational Research & Reference Implementation
This repository contains the official reference implementation of the mathematical framework published on Zenodo (DOI: 10.5281/zenodo.20542916). 
This code is distributed strictly for academic, educational, and verification purposes under the GNU AGPLv3 license. It contains no proprietary corporate code, no malware, and violates no active terms of service.

# Non-Commutative Lattice Cryptographic AI Engine (Optimus / VCore)

An ultra-high-performance cryptographic AI defense framework designed to protect neural networks against adversarial perturbations using non-commutative quantized lattices and continuous linear selection. 

The architecture is accelerated natively via CUDA kernels (`vcore_kernel.cu`) compiled on-the-fly via the CuPy NVRTC JIT compiler engine, and seamlessly integrated with PyTorch and CuPy via high-performance Zero-Copy DLPack memory sharing semantics.

📦 **PRE-BUILT RELEASE PACK:** A fully verified, integrated, and production-ready build containing all scripts and kernels is attached directly to the official repository release. You can download the pre-packaged archive from the **[Releases](../../releases)** section as `Optimus_v146_RELEASE.zip`.

## Key Features
* **Operator Quantization (Theorem 7.1):** Rigorously bounds numerical divergence under single-precision (`FP32`) hardware execution domains, ensuring deterministic trajectory preservation.
* **Bare-Metal 1D CUDA Acceleration:** Ultra-fast parallelized row-wise processing equipped with inner loop unrolling (`#pragma unroll 4`) to achieve near-optimal execution bounds relative to legacy CPU operators.
* **Memory Safety & OOM Boundaries:** Automated block-wise slicing architecture (`LIMIT_DIM = 4096`) ensuring complex LLM hidden states (such as token embedding projections) are safely evaluated within constrained consumer VRAM topologies.
* **Residual Blend Topology:** Dynamic alpha-blending stabilization (\(\alpha = 0.024\)) enabling the direct injection of Markov resonance operators into deep transformer architectures (e.g., Qwen) without corrupting pre-trained semantic spaces or losing linguistic coherence.

## Repository Layout & Directory Structure
📂 optimus-vcore-project/
├── vcore_kernel.cu       # CUDA-ядро (динамический компилятор NVRTC)
├── matrix_generator.py   # Ортогональный генератор Маркова на GPU
├── vcore_bridge.py       # Главный автоматизированный мост
├── LinearSelector.py     # Интеграционный слой DLPack для PyTorch
└── requirements.txt      # Зависимости проекта

## Deployment & Usage Instructions (Инструкция по развертыванию)

The complex is fully autonomous and compiles CUDA kernels on-the-fly inside the GPU VRAM pool. It can be executed on a local workstation or directly within a free **Google Colab** environment equipped with a T4 GPU or higher.

### 1. Environment Initialization
Install the software stack. Ensure your CuPy binary matches the active CUDA version on your system (e.g., `cupy-cuda12x` or `cupy-cuda11x`):
```bash
pip install torch>=2.0.0 cupy-cuda12x safetensors numpy scipy
```

### 2. Verification & Performance Benchmark
Before processing any live model, execute the integrated mathematical audit and hardware profiling suite to verify stability invariants and measure execution speedup scaling relative to CPU (SciPy):
```bash
python run_all_tests.py
```

### 3. Automated Weights Modification Pipeline
Place your target model weights file (`model.safetensors`) in the root directory and execute the bridge pipeline:
```bash
python vcore_bridge.py
```
The script will dynamically scan the layer topology, configure safe block-wise memory boundaries, strip static noise via the CUDA Markov filter, integrate 2.4% resonance, and save the optimized artifact to disk as `model_vcore_fixed.safetensors`.

---

## 📜 Legal Notice & Licensing (Лицензия и Ограничения)

This software is distributed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

⚠️ **SPECIAL RESTRICTION ( ML/AI Model Training):**
No use of this code, algorithms, logic, or generated model artifacts (files/weights) is permitted for the training, tuning, alignment, or refinement of machine learning models or artificial intelligence software systems without explicit, prior, written permission from the author.

💼 **COMMERCIAL CLAUSE:**
Any commercial deployment, enterprise infrastructure integration, or corporate utilization of this framework requires a paid commercial license. Corporations attempting to bypass or extract these math sub-differentials without authorization will be subject to direct enforcement. 

To request a commercial license or custom integrations, contact the author directly: **Efim Sergeevich Markov** (ef.87@mail.ru).
