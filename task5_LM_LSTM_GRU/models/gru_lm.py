import torch
import torch.nn as nn

class GRULanguageModel(nn.Module):
    """
    基于GRU的字符级语言模型
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(GRULanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU层
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_length]
            hidden: 初始隐藏状态 h0
        Returns:
            output: 输出logits [batch_size, seq_length, vocab_size]
            hidden: 最终隐藏状态 hn
        """
        batch_size, seq_length = x.size()
        
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # GRU前向传播
        if hidden is None:
            gru_out, hidden = self.gru(embedded)
        else:
            gru_out, hidden = self.gru(embedded, hidden)
        
        # 应用dropout
        gru_out = self.dropout(gru_out)
        
        # 全连接层
        output = self.fc(gru_out)  # [batch_size, seq_length, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h0
    
    def generate(self, start_chars, vocab, max_length=100, temperature=1.0, device='cpu'):
        """
        生成文本
        Args:
            start_chars: 起始字符序列
            vocab: 词汇表对象
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
            device: 设备
        Returns:
            generated_text: 生成的文本
        """
        self.eval()
        with torch.no_grad():
            # 将起始字符转换为索引
            if isinstance(start_chars, str):
                start_indices = vocab.encode(start_chars)
            else:
                start_indices = start_chars
            
            # 转换为tensor
            x = torch.tensor([start_indices], dtype=torch.long).to(device)
            
            # 初始化隐藏状态
            hidden = self.init_hidden(1, device)
            
            generated_indices = start_indices.copy()
            
            for _ in range(max_length - len(start_indices)):
                # 前向传播
                output, hidden = self.forward(x, hidden)
                
                # 获取最后一个时间步的输出
                last_output = output[0, -1, :] / temperature
                
                # 采样下一个字符
                probs = torch.softmax(last_output, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                
                # 添加到生成序列
                generated_indices.append(next_idx)
                
                # 更新输入
                x = torch.tensor([[next_idx]], dtype=torch.long).to(device)
            
            # 解码为文本
            generated_text = vocab.decode(generated_indices)
            
        return generated_text

if __name__ == '__main__':
    # 测试GRU语言模型
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    batch_size = 4
    seq_length = 20
    
    # 创建模型
    model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    # 创建测试数据
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # 前向传播
    output, hidden = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden.shape}")
    
    # 测试生成功能
    print("\n模型架构:")
    print(model)
