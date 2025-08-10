# 任务四：基于LSTM+CRF的序列标注

## 任务目标

实现一个基于 BiLSTM-CRF 模型的序列标注器，并将其应用于命名实体识别 (NER) 任务。模型需要能够识别出句子中每个单词的标签（例如 `B-PER`, `I-PER`, `O` 等）。

## 知识点

-   **序列标注 (Sequence Labeling)**: 理解其作为一种为序列中的每个元素分配标签的任务的本质。
-   **BiLSTM-CRF 模型**:
    -   **BiLSTM 层**: 从两个方向（前向和后向）上学习每个单词的上下文表示，生成"发射分数" (emission scores)。
    -   **条件随机场 (CRF) 层**: 在 BiLSTM 的输出之上添加一个CRF层。CRF层可以学习标签之间的约束关系（例如，`I-PER` 标签不能跟在 `B-LOC` 之后），从而生成更合理的标签序列。
-   **评价指标**: 学习序列标注任务的常用评价指标，如精确率 (Precision)、召回率 (Recall) 和 F1-score。

## 数据集

-   **CoNLL 2003**: 这是 NER 任务的一个基准数据集。由于其许可证限制，直接下载比较麻烦。
-   **替代方案**: 我们将使用 Hugging Face 的 `datasets` 库来方便地加载和使用 CoNLL 2003 数据集，这避免了手动下载和解析的麻烦。

## 参考资料

-   **BiLSTM-CRF 论文**: [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
-   **PyTorch-CRF 库**: 我们将使用一个流行的第三方库 `pytorch-crf` 来简化 CRF 层的实现。

## 文件结构

-   `utils.py`: 使用 `datasets` 库加载和预处理 CoNLL 2003 数据，构建词汇表和标签映射。
-   `models/`: 存放模型定义文件。
    -   `bilstm_crf.py`: BiLSTM-CRF 模型的核心实现。
-   `main.py`: 主运行文件，负责训练和评估流程。
-   `README.md`: 任务说明。
