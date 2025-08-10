# 任务三：基于注意力机制的文本匹配

## 任务目标

输入一个 "前提" (premise) 句子和一个 "假设" (hypothesis) 句子，判断它们之间的关系。关系分为三种：
1.  **Entailment (蕴含)**: 假设可以从前提中推断出来。
2.  **Contradiction (矛盾)**: 假设与前提相矛盾。
3.  **Neutral (中立)**: 两者之间没有明确的蕴含或矛盾关系。

我们将实现一个基于 **ESIM (Enhanced LSTM for Natural Language Inference)** 的模型来完成此任务。

## 知识点

-   **注意力机制 (Attention Mechanism)**: 这是本任务的核心。我们将实现一个 "token-to-token" 的注意力机制，用于计算两个句子中单词之间的对齐关系。
-   **模型组件**:
    -   使用 Bi-LSTM 对输入句子进行编码。
    -   计算句子间的注意力矩阵。
    -   对注意力结果进行增强和组合。
    -   使用 MLP (多层感知机) 进行最终分类。

## 数据集

-   **SNLI (Stanford Natural Language Inference) Corpus**: 这是一个包含超过57万个人工标注的句子对的大型数据集。
-   **下载链接**: [https://nlp.stanford.edu/projects/snli/snli_1.0.zip](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

由于数据集较大，我们将实现一个脚本来自动下载、解压和解析它。

## 参考资料

-   **ESIM 论文**: [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038v3.pdf)

## 文件结构

-   `data/`: 存放下载和解压后的 SNLI 数据集。
-   `utils.py`: 数据预处理，包括下载数据、解析jsonl文件、构建词汇表、创建DataLoader等。
-   `models/`: 存放模型定义文件。
    -   `esim.py`: ESIM 模型的核心实现。
-   `main.py`: 主运行文件，负责训练和评估流程。
-   `README.md`: 任务说明。
