# 任务二：基于深度学习的文本分类

## 任务目标

熟悉 PyTorch，使用 PyTorch 重写任务一的文本分类问题。具体要求实现两种经典的深度学习模型：

1.  **TextCNN**: 基于卷积神经网络的文本分类模型。
2.  **TextRNN**: 基于循环神经网络 (LSTM或GRU) 的文本分类模型。

## 知识点

-   **PyTorch**: 学习并使用其核心组件，如 `nn.Module`, `nn.Embedding`, `nn.Conv2d`, `nn.LSTM` 等。
-   **词嵌入 (Word Embedding)**:
    -   使用随机初始化的 Embedding 层。
    -   加载预训练的词向量 (如 GloVe) 来初始化 Embedding 层。
-   **CNN/RNN 特征抽取**: 理解这两种网络结构如何从文本序列中提取特征。
-   **Dropout**: 了解并使用 Dropout 来防止模型过拟合。

## 参考资料

-   **PyTorch官网**: [https://pytorch.org/](https://pytorch.org/)
-   **TextCNN论文**: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
-   **GloVe词向量**: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

## 文件结构

-   `data/`: (可选) 存放 GloVe 词向量文件。
-   `main.py`: 主运行文件，负责数据加载、模型训练和评估。
-   `utils.py`: 数据预处理，包括构建词汇表、加载 GloVe 向量等。
-   `models/`: 存放模型定义文件。
    -   `cnn.py`: TextCNN 模型。
    -   `rnn.py`: TextRNN 模型。
-   `README.md`: 任务说明。
