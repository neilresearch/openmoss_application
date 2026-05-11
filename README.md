# openmoss_application
# OpenMOSS Research Assistant Application: PyTorch & NLP Exercises

本项目记录了申请复旦大学 OpenMOSS 组助研岗位所完成的一系列 PyTorch 及 NLP 基础练习。内容涵盖从底层张量运算推导、深度学习架构对比到 Transformer 模型的从零构建。

## 🚀 项目亮点
* **底层重构**：Task-1 摒弃 `nn.Module` 和 `autograd`，基于线性代数算子手写 Softmax 回归的前向与反向传播。
* **架构消融**：Task-2 深入对比了 TextCNN、BiLSTM 与 Transformer 在短文本分类任务中的性能边界及归纳偏置差异。
* **模型构建**：Task-3 手写实现 Multi-head Attention 及标准 Encoder-Decoder / Decoder-only 架构。
* **实验严谨**：不仅包含模型实现，还针对学习率敏感度、长度泛化（多位数加法）及语言模型涌现过程进行了详细分析。

---

## 📂 模块说明

### [Task-1] 基于机器学习的文本分类
* **核心逻辑**：手动推导交叉熵（CE）与均方误差（MSE）的梯度公式，利用 PyTorch 张量运算实现 Mini-batch 训练。
* **特征工程**：对比了 BoW 与 N-gram (Uni/Bi/Tri-gram) 的表征优势。
* **结论**：验证了 Cross-Entropy 在处理极端偏差初始化时的收敛稳定性。

### [Task-2] 基于深度学习的文本分类
* **模型实现**：实现 TextCNN (Yoon Kim, 2014)、BiLSTM-Attention 及 Transformer Encoder。
* **性能对比**：在相同参数量级下，Transformer 以 **48.87%** 的准确率和 **42.30%** 的 F1-Macro 优于传统 CNN/RNN 结构。
* **词嵌入策略**：探讨了 GloVe 预训练权重在微调与冻结状态下的泛化表现。

### [Task-3] Transformer 基础结构实现
* **子任务 1 (加法逻辑)**：通过 3-5 位数加法测试验证了标准 Transformer 在逻辑推理中的长度泛化局限。
* **子任务 2 (语言模型)**：基于《老友记》剧本，使用 GPT-style 架构实践了 Next-token Prediction 训练链路，展示了模型从随机乱码到习得剧本格式及语法的演变过程。

---

## 📊 核心实验结果
> 详细实验图表及数学推导请参阅 [Full Report (PDF)](./reports/Fudan_OpenMOSS_Application_Report.pdf)

| 模型架构 | 测试集准确率 | F1-Macro | 备注 |
| :--- | :---: | :---: | :--- |
| Softmax (Tri-gram) | 48.27% | 42.00% | Task-1 Baseline |
| TextCNN | 45.00% | 29.34% | Task-2 |
| Transformer Encoder | 48.87% | 42.30% | Task-2 (SOTA) |

---

## 🛠️ 环境配置与运行指南
本项目基于 PyTorch 实现，建议使用 CUDA 环境以加速计算。

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
