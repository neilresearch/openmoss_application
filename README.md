# OpenMOSS 助研申请项目：PyTorch 基础与具身智能实践

本项目包含申请复旦大学 OpenMOSS 组助研岗位所完成的系统性练习，涵盖了从底层张量运算、深度学习架构对比到具身智能（Embodied AI）模仿学习与大模型任务规划的深度探索。

## 🚀 项目亮点

* **底层架构重写**：脱离 `torch.nn` 高级封装，从线性代数层面手写 Softmax 回归及 Transformer 底层算子 。


* **架构性能消融**：针对同一文本分类任务，横向对比了 TextCNN、BiLSTM 与 Transformer Encoder 的归纳偏置差异 。


* **具身智能闭环**：实现了从基于 Diffusion Policy 的模仿学习到基于 LLM 的 Code as Policies（CaP）任务规划，并完成了 7B 级别模型的具身指令微调（SFT） 。



---

## 📂 仓库结构

```text
.
├── pytorch_basics/                    # PyTorch & NLP 基础练习
│   ├── task1/                         # 纯张量实现的 Softmax 分类器
│   ├── task2/                         # 深度学习文本分类对比 (CNN/RNN/Transformer)
│   └── task3/                         # Transformer & Decoder-only 从零实现
├── EI_beginner/                       # 具身智能实践
│   ├── CaP_benchmark_run/             # 在Pybullet仿真实验室验证Code as Policies(Finetuned)
│   ├── CaP_prompt_pybullet/           # 在Pybullet仿真实验室验证Code as Policies(by Prompt)，收集Finetune数据 
│   ├── plot_aloha_loss/               # diffusion policy aloha损失曲线绘制
│   ├── plot_pusht_ablation/           # diffusion policy pusht消融实验损失曲线绘制
│   └── run_diffusion_steps_ablation/  # Diffusion Policy 消融实验   
├── reports/                           # 深度实验报告 (PDF)
│   ├── PyTorch基本练习实验报告.pdf
│   └── 具身智能入门练习实验报告.pdf
└── assets/                            # 实验结果可视化 (曲线图、仿真视频)

```

---

## 🧪 核心任务综述

### 1. PyTorch & NLP 基础

* **Task-1: 机器学习基准**：基于基础线性代数构建 Softmax 回归，验证了 N-gram 相比 BoW 的特征表示优势（测试集准确率 48.27%） 。


* **Task-2: 深度架构对比**：使用 GloVe 300d 词嵌入，在控制参数量（$10^5$ 级）一致时，Transformer Encoder 在准确率（48.87%）与 F1-Macro（42.30%）上均显著优于 TextCNN 与 BiLSTM 。


* **Task-3: Transformer 与泛化探究**：
* **加法任务**：揭示了标准 Transformer 在未见数位组合上的内插与外推泛化局限 。


* **语言模型**：基于《老友记》剧本实践了 LM 预训练，观察到模型从“字符统计”到“高层句法”的涌现过程 。





### 2. 具身智能实践

* **模仿学习 (Diffusion Policy)**：通过 PushT 任务验证了生成式策略在多模态动作分布拟合上的优越性，并探讨了时序观测界限（Observation Horizon）对样本效率的影响 。


* **任务规划 (Code as Policies)**：
* 利用 LLM 生成 Python 代码驱动 PyBullet 仿真环境中的 Franka Panda 机械臂 。


* **知识蒸馏与 SFT**：将 DeepSeek-Chat 的规划能力蒸馏至 **Qwen2.5-7B**，使 7B 模型在大幅缩减推理耗时的同时，内化了空间坐标分析逻辑 。



* **场景级任务**：在 ALFRED 基准测试中，通过结构化 ICL 与双轨制 CoT 提升了多模态模型在复杂环境中的子目标拆解与错误恢复能力 。



---

## 📈 实验数据摘要

| 任务模块 | 核心模型/配置 | 主要指标 | 结论 |
| --- | --- | --- | --- |
| **文本分类** | Transformer Encoder | Acc: 48.87% / F1: 42.30% | 全局自注意力机制优于局部卷积 

 |
| **算术推理** | Encoder-Decoder | IID EM: ~100% | 长度泛化（外推）能力存在显著瓶颈 

 |
| **具身规划** | Qwen2.5-7B SFT | Loss: ~0.2141 | 蒸馏模型推理耗时缩减约 50% 


---

## 🛠️ 环境要求

* Python 3.9+
* PyTorch 2.x
* 仿真环境: PyBullet, Gym-PushT, AI2-THOR 


* 具身框架: LeRobot, EmbodiedBench 



