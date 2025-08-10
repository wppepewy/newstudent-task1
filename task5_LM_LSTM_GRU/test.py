#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import load_poetry_data, prepare_data, calculate_perplexity
from models.lstm_lm import LSTMLanguageModel
from models.gru_lm import GRULanguageModel

def test_models():
    """
    测试LSTM和GRU语言模型
    """
    print("=== 测试字符级语言模型 ===\n")
    
    # 加载数据
    print("1. 加载数据...")
    text = load_poetry_data()
    print(f"   文本长度: {len(text)} 字符")
    print(f"   前50个字符: {text[:50]}")
    
    # 准备数据
    print("\n2. 准备数据...")
    dataloader, vocab = prepare_data(text, seq_length=20, batch_size=4)
    print(f"   词汇表大小: {vocab.vocab_size}")
    print(f"   词汇表前10个字符: {vocab.chars[:10]}")
    
    # 测试LSTM模型
    print("\n3. 测试LSTM模型...")
    lstm_model = LSTMLanguageModel(
        vocab_size=vocab.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1
    )
    
    # 获取一个batch的数据进行测试
    for inputs, targets in dataloader:
        print(f"   输入形状: {inputs.shape}")
        print(f"   目标形状: {targets.shape}")
        
        # 前向传播
        outputs, hidden = lstm_model(inputs)
        print(f"   输出形状: {outputs.shape}")
        print(f"   隐藏状态形状: {hidden[0].shape}, {hidden[1].shape}")
        break
    
    # 测试GRU模型
    print("\n4. 测试GRU模型...")
    gru_model = GRULanguageModel(
        vocab_size=vocab.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1
    )
    
    # 获取一个batch的数据进行测试
    for inputs, targets in dataloader:
        # 前向传播
        outputs, hidden = gru_model(inputs)
        print(f"   输出形状: {outputs.shape}")
        print(f"   隐藏状态形状: {hidden.shape}")
        break
    
    # 测试文本生成
    print("\n5. 测试文本生成...")
    start_text = "春"
    print(f"   起始文本: '{start_text}'")
    
    # LSTM生成
    lstm_generated = lstm_model.generate(start_text, vocab, max_length=20, temperature=1.0)
    print(f"   LSTM生成: {lstm_generated}")
    
    # GRU生成
    gru_generated = gru_model.generate(start_text, vocab, max_length=20, temperature=1.0)
    print(f"   GRU生成: {gru_generated}")
    
    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_models()
