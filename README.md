
\qquad 随着大语言模型（Large Language Models, LLMs）在问答系统与智能辅助决策领域的广泛应用，其在网络安全知识理解与问答场景中的潜力逐渐显现。然而，直接使用通用大模型往往存在专业领域知识不足、推理结果不稳定以及本地部署成本较高等问题。为提升模型在网络安全领域的专业性与实用性，参数高效微调方法（Parameter-Efficient Fine-Tuning, PEFT）成为一种可行方案，其中 LoRA（Low-Rank Adaptation）因其训练成本低、显存占用小、易于本地部署等优势，被广泛应用于领域适配任务。

本次实验以“基于 LoRA 的网络安全问答系统开发”为目标，围绕模型推理服务化、前端交互可视化以及系统整体联调与测试展开。作为一项个人技术汇报工作，本文中的系统设计、代码实现与实验验证均由本人独立完成，重点考察 LoRA 微调模型在网络安全问答场景下的工程可行性与实际交互效果

环境依赖配置：
\begin{verbatim}
transformers==4.37.2
accelerate==0.27.2
peft==0.8.2
datasets==2.16.1
bitsandbytes==0.43.1
torch>=2.1.0
sentencepiece==0.1.99
tokenizers==0.15.1
protobuf==4.25.1
einops==0.7.0
tqdm==4.66.1
gradio==4.19.1
numpy==1.26.4
\end{verbatim}

环境设置命令：
\begin{verbatim}
conda create -n lora python=3.10
conda activate lora
pip install -r requirements.txt
\end{verbatim}

\subsection{代码仓库结构}

项目采用模块化设计，结构清晰：

\begin{verbatim}
lora/
├── model/                          # 源代码目录
│   ├── model/                    # qwen2.5
├── api/                   
│   └──api_server.py             # api服务                     
├── dataset/                       # 数据目录
│   └── train.json          # 数据集
├── scripts/                    # 实验结果
│   ├── inference.py             #模型验证
│   ├── prepare_data.py          #数据准备
│   └── train_lora.py           #模型训练 
├── requirements.txt    # 环境依赖
├── front.html         #前端界面  
└── README.md                   # 项目说明文档
\end{verbatim}
