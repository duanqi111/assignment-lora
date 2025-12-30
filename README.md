# 基于 LoRA 的网络安全问答系统开发 (CyberSecurity QA System)

## 📖 项目简介
随着大语言模型（LLMs）的广泛应用，其在垂直领域（如网络安全）的专业性需求日益增长。本项目旨在通过 **LoRA (Low-Rank Adaptation)** 参数高效微调技术，解决通用大模型在网络安全领域知识不足、推理不稳定及私有化部署成本高的问题。

本项目实现了从**数据准备、模型微调、API 推理服务化**到**前端可视化交互**的全链路开发，重点验证了 LoRA 方案在网络安全问答场景下的工程可行性与实际效果。

---

## ✨ 核心特性
- **参数高效微调**：采用 LoRA 技术，仅微调极少量参数，显著降低显存占用。
- **专业领域增强**：针对网络安全知识进行深度适配，提升模型在漏洞分析、威胁评估等场景的回答质量。
- **全栈式架构**：包含基于 Python 的高性能推理 API 及简洁易用的 Web 前端交互界面。

---

## 🛠️ 环境依赖与配置
### 1. 代码结构

```text
lora/
├── model/                  # 基础模型与 LoRA 权重
├── api/
│   └── api_server.py       # 推理 API 服务
├── dataset/
│   └── train.json          # 网络安全领域 SFT 数据集
├── scripts/
│   ├── train_lora.py       # LoRA 微调训练脚本
│   ├── prepare_data.py     # 数据预处理
│   └── inference.py        # 推理与测试
├── requirements.txt        # Python 依赖
├── front.html              # 前端交互页面
└── README.md               # 项目说明
```  

### 2. 软件依赖
项目核心环境要求如下（详见 `requirements.txt`）：
- **Python**: 3.10+
- **CUDA/PyTorch**: 2.1.0+
- **关键库**: `transformers (4.37.2)`, `peft (0.8.2)`, `bitsandbytes (0.43.1)`, `gradio (4.19.1)`

### 3. 环境安装
```bash
# 创建虚拟环境
conda create -n lora python=3.10 -y
conda activate lora

# 安装依赖
pip install -r requirements.txt



