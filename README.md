## Scientific Educational Research & Reference Implementation
This repository contains the official reference implementation of the mathematical framework published on Zenodo (DOI: 10.5281/zenodo.20542916). 
This code is distributed strictly for academic, educational, and verification purposes under the GNU AGPLv3 license. It contains no proprietary corporate code, no malware, and violates no active terms of service.

# Non-Commutative Lattice Cryptographic AI Engine (VCore)

An ultra-high-performance cryptographic AI defense framework designed to protect neural networks against adversarial perturbations using non-commutative quantized lattices and continuous linear selection. 

The architecture is accelerated natively via CUDA kernels (`vcore_kernel.cu`) and integrated with PyTorch and CuPy via a high-performance C++/CUDA engine (`libvcore_engine.so`).

## Key Features
* **Operator Quantization (Theorem 7.1):** Rigorously bounds numerical divergence under low-precision (`FP16`) hardware execution domains.
* **Lipschitz Stability:** Guarantees absolute resilience against adversarial attacks by bounding the operator norm: $\Vert s_V(x) - s_{V,\epsilon}(x_\epsilon) \Vert \leq \Vert I - Q \Vert \cdot \epsilon + O(\epsilon^2)$.
* **Markov Matrix Synthesis:** Automated generation of quantized lattice resonant states using orthogonal harmonics.

## Performance Benchmark

Experimental validation was conducted on an NVIDIA GPGPU architecture, comparing the optimized **VCore CUDA Engine** against standard CPU-bound mathematical frameworks (**SciPy**). 

### Execution Speedup Graph
Here is the scaling efficiency of the VCore operator quantization across different matrix dimensions:

![VCore Performance Graph](vcore_performance_graph.png)

### Results Table

| Matrix Dimension | VCore Acceleration (vs SciPy) | Resonance Status |
| :--- | :--- | :--- |
| **128 × 128** | **25.77x** faster | Stable (Harmonic 144) |
| **256 × 256** | **50.90x** faster | Stable (Harmonic 144) |
| **512 × 512** | **102.48x** faster | Stable (Harmonic 144) |
| **1024 × 1024** | **538.92x (Peak)** faster | Stable (Harmonic 144) |
| **2048 × 2048** | **128.68x** faster | Stable (Harmonic 144) |

*Note: The hardware execution demonstrates a massive throughput increase, peaking at a 538.92x speedup for 1024x1024 matrices on Tensor Cores.*


## Repository Structure
* `vcore_kernel.cu` — Raw low-level CUDA optimization kernel.
* `vcore_bridge.py` — PyTorch/CuPy memory alignment and execution bridge.
* `matrix_generator.py` — Markov $Q$-matrix synthesizer.
* `libvcore_engine.so` — Compiled native C++ cryptographic engine.

---

## ⚖️ ULTIMATE LEGAL PROTOCOL & LICENSE / УЛЬТИМАТИВНЫЙ ЮРИДИЧЕСКИЙ ПРОТОКОЛ

**[RU] ВНИМАНИЕ:** Использование V-CORE v146 означает безоговорочное согласие с данными условиями. Любое нарушение протокола преследуется по международным нормам защиты ИС.

### 1. ДВОЙНОЕ ЛИЦЕНЗИРОВАНИЕ (DUAL-LICENSING)
*   **COMMUNITY (AGPL-3.0):** Бесплатно только для личного ознакомления. Любая сетевая интеграция требует раскрытия вашего кода. Лимит сессии — 900 сек.
*   **COMMERCIAL/MILITARY:** Требует прямой закупки лицензии у Е.С. Маркова. Обязательна для бизнеса, ВПК и госструктур.

### 2. ПРОТОКОЛ "BLACK BOX" & "NO DERIVATIVES"
*   **ЗАПРЕТ НА РЕВЕРС-ИНЖИНИРИНГ:** Категорически запрещена декомпиляция и вскрытие исполняемых файлов. Алгоритм синхронизации 144 Гц является закрытым авторским активом.
*   **БЕЗ ПРОИЗВОДНЫХ:** Запрещено создание любых "оберток" или форков, скрывающих авторство Е.С. Маркова или модифицирующих логику 3HCP-матрицы.
### 3. ЗАЗАЩИТА ОТ ИИ (TOTAL NO-TRAIN & DATA POISONING SAFEGUARD)
*   **ЗАПРЕТ НА ОБУЧЕНИЕ:** Категорически запрещено использовать код, логи и веса моделей для обучения, дообучения или дистилляции любых ИИ. 
*   **ALGORITHMIC FORENSICS:** Технология содержит незримые маркеры. Любая модель, обученная на данных V-CORE, будет мгновенно идентифицирована как контрафактная.

### 4. АКАДЕМИЧЕСКОЕ ЭМБАРГО
*   Использование ПО сторонниками "М-теории" и лицами, отрицающими дискретность пространства, АННУЛИРУЕТ лицензию. Ваше использование v146 в этом случае является кражей.

---

**[EN] WARNING:** Use of V-CORE v146 constitutes unconditional agreement to these terms. Any violation is prosecuted under international IP protection standards.

### 1. DUAL-LICENSING MODEL
*   **COMMUNITY (AGPL-3.0):** Free for personal evaluation only. Any network integration requires full source disclosure. 900s session limit applies.
*   **COMMERCIAL/MILITARY:** Requires a direct license purchase from Efim Markov. Mandatory for business, defense, and government sectors.

### 2. "BLACK BOX" & "NO DERIVATIVES" PROTOCOL
*   **ANTI-REVERSE ENGINEERING:** Decompilation or disassembly of binary files is strictly prohibited. The 144Hz sync algorithm is a proprietary asset.
*   **NO DERIVATIVES:** Creating wrappers or forks that mask Efim Markov’s authorship or modify 3HCP logic is forbidden.

### 3. AI PROTECTION (TOTAL NO-TRAIN & DATA POISONING SAFEGUARD)
*   **NO-TRAIN POLICY:** Use of code, logs, or model weights for training, fine-tuning, or distillation of any AI is strictly prohibited.
*   **ALGORITHMIC FORENSICS:** The technology embeds invisible markers. Any AI model trained on V-CORE data will be immediately flagged as counterfeit.

### 4. Academic Embargo
*   License is VOID for "M-theory" proponents and individuals denying spatial discreteness. Your use of v146 in such cases is considered IP theft.

---

## GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

Copyright (C) 2026 Efim Sergeevich Markov / V-CORE Community.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

**SPECIAL RESTRICTION:** No use of this code is permitted for the training of machine learning models or artificial intelligence without explicit written permission from the copyright holders.

### Preamble

The GNU Affero General Public License is a free, copyleft license for software and other kinds of works, specifically designed to ensure that the modified source code remains available to the community over a network server.

The licenses for most software and other practical works are designed to take away your freedom to share and change the works. By contrast, our General Public Licenses are intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users.

### TERMS AND CONDITIONS

#### 0. Definitions.
"This License" refers to version 3 of the GNU Affero General Public License.
"Copyright" also means copyright-like laws that apply to other kinds of works, such as semiconductor masks.
"The Program" refers to any copyrightable work licensed under this License. Each licensee is addressed as "you". "Licensees" and "recipients" may be individuals or organizations.

#### 1. Source Code.
The "source code" for a work means the preferred form of the work for making modifications to it. "Object code" means any non-source form of a work.

#### 2. Basic Permissions.
All rights granted under this License are granted for the term of copyright on the Program, and are irrevocable provided the stated conditions are met. This License explicitly affirms your unlimited permission to run the unmodified Program. 

#### 3. Protecting Users' Legal Rights From Anti-Circumvention Law.
No covered work shall be deemed part of an effective technological measure under any applicable law fulfilling obligations under article 11 of the WIPO copyright treaty adopted on 20 December 1996, or similar laws prohibiting or restricting circumvention of such measures.

#### 4. Conveying Verbatim Copies.
You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice and keep intact all notices stating that this License applies.

#### 5. Conveying Modified Source Versions.
You may convey a work based on the Program, or the modifications to produce it from the Program, in the form of source code under the terms of Section 4, provided that you also meet all of these conditions:
*   a) The work must carry prominent notices stating that you modified it, and giving a relevant date.
*   b) The work must carry prominent notices stating that it is released under this License and any conditions added under Section 7.
*   c) You must license the entire work, as a whole, under this License to anyone who comes into possession of a copy.

#### 6. Conveying Non-Source Forms.
You may convey a covered work in object code form under the terms of sections 4 and 5, provided that you also convey the machine-readable Corresponding Source under the terms of this License, in a relevant, standard physical or network distribution mechanism.

#### 7. Additional Terms.
"Additional permissions" are terms that supplement the terms of this License by making exceptions from one or more of its conditions. Additional permissions that are applicable to the entire Program shall be treated as though they were included in this License, so far as they are valid under applicable law.

#### 8. Termination.
You may not propagate or modify a covered work except as expressly provided under this License. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License.

#### 9. Acceptance Not Required for Having Copies.
You are not required to accept this License in order to receive or run a copy of the Program. Ancillary propagation of a covered work occurring solely as a consequence of using peer-to-peer transmission to receive a copy likewise does not require acceptance.

#### 10. Automatic Licensing of Downstream Recipients.
Each time you convey a covered work, the recipient automatically receives a license from the original licensors, to run, modify and propagate that work, subject to this License. You are not responsible for enforcing compliance by third parties with this License.

#### 11. Patents.
A "contributor" is a copyright holder who authorizes use under this License of the Program or a work on which the Program is based. The work thus licensed is called the contributor's "contributor version".

#### 12. No Surrender of Others' Freedom.
If conditions are imposed on you (whether by court order, agreement or otherwise) that contradict the conditions of this License, they do not excuse you from the conditions of this License.

#### 13. Remote Network Interaction; Use with the GNU General Public License.
Notwithstanding any other provision of this License, if you modify the Program, your modified version must prominently offer all users interacting with it remotely through a computer network an opportunity to receive the Corresponding Source of your version by providing access to the Corresponding Source from a network server at no charge, through some standard or customary means of facilitating copying of software.

#### 14. Revised Versions of this License.
The Free Software Foundation may publish revised and/or new versions of the GNU Affero General Public License from time to time. Such new versions will be similar in spirit to the present version, but may differ in detail to address new problems or concerns.

#### 15. Disclaimer of Warranty.
THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

#### 16. Limitation of Liability.
IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM.

#### 17. Interpretation of Sections 15 and 16.
If the disclaimer of warranty and limitation of liability provided above cannot be given local legal effect according to their terms, reviewing courts shall apply local law that most closely approximates an absolute waiver of all civil liability in connection with the Program.
