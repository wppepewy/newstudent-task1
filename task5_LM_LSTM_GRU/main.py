import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm

from utils import load_poetry_data, prepare_data, calculate_perplexity
from models.lstm_lm import LSTMLanguageModel
from models.gru_lm import GRULanguageModel

def train_model(model, dataloader, vocab, device, epochs=50, learning_rate=0.001, clip=1.0):
    """
    训练语言模型
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"开始训练 {model.__class__.__name__}...")
    print(f"设备: {device}")
    print(f"词汇表大小: {vocab.vocab_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs, _ = model(inputs)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            model_path = f"models/{model.__class__.__name__}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
    
    return model

def evaluate_model(model, dataloader, vocab, device):
    """
    评估模型并计算困惑度
    """
    print(f"评估 {model.__class__.__name__}...")
    
    # 计算困惑度
    perplexity = calculate_perplexity(model, dataloader, device)
    print(f"困惑度: {perplexity:.2f}")
    
    return perplexity

def generate_text(model, vocab, device, start_text="春", max_length=50, temperature=1.0):
    """
    生成文本
    """
    print(f"使用 {model.__class__.__name__} 生成文本...")
    print(f"起始文本: '{start_text}'")
    print(f"最大长度: {max_length}")
    print(f"温度: {temperature}")
    
    generated_text = model.generate(start_text, vocab, max_length, temperature, device)
    
    print(f"生成的文本: {generated_text}")
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='字符级语言模型训练和评估')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru'], 
                        help='选择模型类型 (lstm 或 gru)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--seq_length', type=int, default=50, help='序列长度')
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM/GRU层数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--data_path', type=str, default='data/poetryFromTang.txt', 
                        help='数据文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu, cuda, auto)')
    parser.add_argument('--generate', action='store_true', help='是否生成文本')
    parser.add_argument('--start_text', type=str, default='春', help='生成文本的起始字符')
    parser.add_argument('--max_length', type=int, default=50, help='生成文本的最大长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成文本的温度参数')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    text = load_poetry_data(args.data_path)
    print(f"文本长度: {len(text)} 字符")
    
    # 准备数据
    dataloader, vocab = prepare_data(text, args.seq_length, args.batch_size)
    print(f"词汇表大小: {vocab.vocab_size}")
    
    # 创建模型
    if args.model == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # gru
        model = GRULanguageModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    model = train_model(
        model=model,
        dataloader=dataloader,
        vocab=vocab,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # 评估模型
    perplexity = evaluate_model(model, dataloader, vocab, device)
    
    # 生成文本
    if args.generate:
        generated_text = generate_text(
            model=model,
            vocab=vocab,
            device=device,
            start_text=args.start_text,
            max_length=args.max_length,
            temperature=args.temperature
        )
    
    # 保存最终模型
    model_path = f"models/{model.__class__.__name__}_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到 {model_path}")

if __name__ == '__main__':
    # 创建models目录
    os.makedirs('models', exist_ok=True)
    
    main()
