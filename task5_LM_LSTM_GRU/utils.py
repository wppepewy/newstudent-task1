import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

class CharDataset(Dataset):
    """
    字符级语言模型数据集
    """
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.text) - self.seq_length
        
    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class CharVocabulary:
    """
    字符级词汇表
    """
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def encode(self, text):
        """将文本编码为索引序列"""
        return [self.char2idx[ch] for ch in text]
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        return ''.join([self.idx2char[idx] for idx in indices])

def load_poetry_data(data_path="data/poetryFromTang.txt"):
    """
    加载唐诗数据
    """
    if not os.path.exists(data_path):
        # 如果文件不存在，创建一个示例数据
        print(f"数据文件 {data_path} 不存在，创建示例数据...")
        sample_text = """
        春眠不觉晓，处处闻啼鸟。
        夜来风雨声，花落知多少。
        
        床前明月光，疑是地上霜。
        举头望明月，低头思故乡。
        
        白日依山尽，黄河入海流。
        欲穷千里目，更上一层楼。
        
        千山鸟飞绝，万径人踪灭。
        孤舟蓑笠翁，独钓寒江雪。
        """
        
        # 确保data目录存在
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

def prepare_data(text, seq_length=50, batch_size=32):
    """
    准备训练数据
    """
    # 构建词汇表
    vocab = CharVocabulary(text)
    
    # 将文本编码为索引序列
    encoded_text = vocab.encode(text)
    
    # 创建数据集
    dataset = CharDataset(encoded_text, seq_length)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, vocab

def calculate_perplexity(model, dataloader, device):
    """
    计算困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs, _ = model(inputs)  # 修复：LSTM和GRU模型返回(output, hidden)
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
            
            total_loss += loss.item() * inputs.size(0) * inputs.size(1)
            total_tokens += inputs.size(0) * inputs.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()

if __name__ == '__main__':
    # 测试数据处理
    text = load_poetry_data()
    print(f"文本长度: {len(text)}")
    print(f"前100个字符: {text[:100]}")
    
    dataloader, vocab = prepare_data(text, seq_length=20, batch_size=4)
    print(f"词汇表大小: {vocab.vocab_size}")
    
    # 获取一个batch的数据
    for inputs, targets in dataloader:
        print(f"输入形状: {inputs.shape}")
        print(f"目标形状: {targets.shape}")
        print(f"输入示例: {inputs[0]}")
        print(f"目标示例: {targets[0]}")
        break
