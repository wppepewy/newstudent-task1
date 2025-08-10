# 语言模型使用说明

## 快速开始

### 1. 环境要求

确保已安装以下Python包：
```bash
pip install torch torchvision torchaudio
pip install tqdm
```

### 2. 运行示例

#### 运行测试脚本
```bash
cd newcode/task5_LM_LSTM_GRU
python test.py
```

#### 运行示例训练
```bash
python run_example.py
```

#### 运行完整训练
```bash
# 训练LSTM模型
python main.py --model lstm --epochs 50 --batch_size 32 --seq_length 50

# 训练GRU模型
python main.py --model gru --epochs 50 --batch_size 32 --seq_length 50

# 训练并生成文本
python main.py --model lstm --epochs 20 --generate --start_text "春" --max_length 100
```

## 参数说明

### 主要参数

- `--model`: 模型类型，可选 `lstm` 或 `gru`，默认为 `lstm`
- `--epochs`: 训练轮数，默认为 50
- `--batch_size`: 批次大小，默认为 32
- `--seq_length`: 序列长度，默认为 50
- `--embedding_dim`: 嵌入维度，默认为 128
- `--hidden_dim`: 隐藏层维度，默认为 256
- `--num_layers`: LSTM/GRU层数，默认为 2
- `--learning_rate`: 学习率，默认为 0.001
- `--dropout`: Dropout率，默认为 0.5

### 生成参数

- `--generate`: 是否生成文本，默认不生成
- `--start_text`: 生成文本的起始字符，默认为 "春"
- `--max_length`: 生成文本的最大长度，默认为 50
- `--temperature`: 生成文本的温度参数，默认为 1.0

## 文件结构

```
task5_LM_LSTM_GRU/
├── README.md           # 任务说明
├── USAGE.md           # 使用说明
├── main.py            # 主运行文件
├── test.py            # 测试脚本
├── run_example.py     # 运行示例
├── utils.py           # 数据处理工具
├── models/            # 模型定义
│   ├── lstm_lm.py     # LSTM语言模型
│   └── gru_lm.py      # GRU语言模型
└── data/              # 数据目录
    └── poetryFromTang.txt  # 唐诗数据
```

## 模型说明

### LSTM语言模型

- 使用LSTM作为循环神经网络
- 支持多层LSTM
- 包含词嵌入层和全连接输出层
- 支持dropout正则化

### GRU语言模型

- 使用GRU作为循环神经网络
- 支持多层GRU
- 包含词嵌入层和全连接输出层
- 支持dropout正则化

## 评估指标

### 困惑度 (Perplexity)

困惑度是语言模型的重要评估指标，计算公式为：
```
perplexity = exp(cross_entropy_loss)
```

困惑度越低，模型性能越好。

## 文本生成

模型支持基于温度采样的文本生成：

- `temperature = 1.0`: 标准采样
- `temperature < 1.0`: 更确定性的生成
- `temperature > 1.0`: 更随机的生成

## 注意事项

1. 首次运行时会自动创建示例数据文件
2. 训练时间取决于数据大小和模型复杂度
3. 建议使用GPU进行训练以提高速度
4. 生成的文本质量取决于训练数据的质量和数量
