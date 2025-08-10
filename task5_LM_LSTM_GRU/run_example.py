#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语言模型运行示例
"""

import torch
import argparse
from utils import load_poetry_data, prepare_data
from models.lstm_lm import LSTMLanguageModel
from models.gru_lm import GRULanguageModel

def run_example():
    """
    运行语言模型示例
    """
    print("=== 字符级语言模型示例 ===\n")
    
    # 设置参数
    seq_length = 30
    batch_size = 16
    embedding_dim = 64
    hidden_dim = 128
    num_layers = 1
    epochs = 10
    learning_rate = 0.001
    
    # 加载数据
    print("1. 加载数据...")
    text = load_poetry_data()
    print(f"   文本长度: {len(text)} 字符")
    
    # 准备数据
    print("\n2. 准备数据...")
    dataloader, vocab = prepare_data(text, seq_length, batch_size)
    print(f"   词汇表大小: {vocab.vocab_size}")
    
    # 创建LSTM模型
    print("\n3. 创建LSTM模型...")
    lstm_model = LSTMLanguageModel(
        vocab_size=vocab.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 创建GRU模型
    print("4. 创建GRU模型...")
    gru_model = GRULanguageModel(
        vocab_size=vocab.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n5. 使用设备: {device}")
    
    # 训练LSTM模型
    print("\n6. 训练LSTM模型...")
    lstm_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        lstm_model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = lstm_model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 训练GRU模型
    print("\n7. 训练GRU模型...")
    gru_model.to(device)
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        gru_model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = gru_model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 生成文本
    print("\n8. 生成文本...")
    start_texts = ["春", "月", "山", "水"]
    
    for start_text in start_texts:
        print(f"\n   起始文本: '{start_text}'")
        
        # LSTM生成
        lstm_generated = lstm_model.generate(start_text, vocab, max_length=30, temperature=0.8)
        print(f"   LSTM生成: {lstm_generated}")
        
        # GRU生成
        gru_generated = gru_model.generate(start_text, vocab, max_length=30, temperature=0.8)
        print(f"   GRU生成: {gru_generated}")
    
    print("\n=== 示例完成 ===")

if __name__ == '__main__':
    run_example()
