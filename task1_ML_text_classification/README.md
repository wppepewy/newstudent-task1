# 任务一：基于机器学习的文本分类

## 任务目标

实现基于 logistic/softmax regression 的文本分类。

## 知识点

-   **文本特征表示**: Bag-of-Word, N-gram
-   **分类器**: logistic/softmax regression, 损失函数, (随机)梯度下降, 特征选择
-   **数据集**: 训练集/验证集/测试集的划分

## 数据集

-   [Rotten Tomatoes dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

## 实现要求

-   使用 NumPy 实现

## 实验要求

1.  分析不同的特征、损失函数、学习率对最终分类性能的影响。
2.  实现 `shuffle`、`batch`、`mini-batch` 的功能。

## 文件结构

-   `data/`: 存放数据集
-   `main.py`: 主运行文件
-   `utils.py`: 数据处理和辅助函数
-   `model.py`: logistic/softmax 回归模型
-   `README.md`: 任务说明
