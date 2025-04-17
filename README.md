# ZernFDA
Open-source zernike polynomials acceleration


本项目是我们团队在FPGA-accelerated zernike polynomials computation领域的最新研究成果实现，相关论文已投稿至**TCAS-II**（Under Review）。提供从CPU/GPU实现到硬件FPGA实现的完整设计。
﻿
## 目录结构
```
.
├── FPGA/                  # FPGA硬件实现
│   ├── bitstream/         # 预生成的比特文件（*.bit）
│   ├── constraints/       # 时序约束与物理约束
│
├── Ours/                  # CPU/GPU参考实现
│   ├── Ours.py            # 基于pytorch的CPU/GPU设计，包含未优化和所提方法的设计
│
├── fonts/                 # 绘图使用的字体ttf文件
|
├── evaluation/            # 评估文件夹
│   ├── time/              # 保存测试生成的时间数据
|
├── ZERN/                  # 复现的开源工程,来自
|
├── zernike/               # 复现的开源工程,来自
|
├── zernpax/               # 复现的开源工程,来自
|
├── zernpy/                # 复现的开源工程,来自
|
└── LICENSE                # MIT开源协议
```
﻿
## 获取完整工程
完整FPGA工程需通过发送申请邮件至 helix@seu.edu.cn(注明机构/研究方向)学术审核获取：
﻿
## 快速开始
```python
# 基础依赖
pip install -r requirements.txt
﻿
# CPU版本运行样例
python main_cpu.py

# GPU版本运行样例
python main_gpu.py

# FPGA运行样例
﻿使用Vivado工具，进行bit文件烧写至KU060 FPGA，将HDMI连接至显示屏和FPGA，即可显示实时Zernike计算的测试图样
﻿
## 许可协议
本项目采用**学术共享许可**：
- FPGA实现部分受[NDA](docs/NDA.pdf)约束
- CPU/GPU实现遵循[MIT License](LICENSE)

**东南大学异构智能与量子计算实验室** © 2024
``` 
