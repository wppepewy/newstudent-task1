import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, bidirectional, dropout_prob):
        """
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词向量维度
        :param hidden_dim: LSTM 隐藏层维度
        :param num_classes: 类别数
        :param num_layers: LSTM 层数
        :param bidirectional: 是否使用双向LSTM
        :param dropout_prob: dropout的概率
        """
        super(TextRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True, # 输入和输出张量将以[batch, seq, feature]的形式提供
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 如果是双向的，全连接层的输入维度要乘以2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # 1. Embedding
        embedded = self.embedding(x)  # -> [batch_size, seq_len, embedding_dim]
        
        # 2. LSTM
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # cell: [num_layers * num_directions, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 3. We use the final hidden state from the last layer
        # Concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            # hidden is shaped [num_layers * 2, batch_size, hidden_dim]
            # We want the last layer's forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        # hidden is now [batch_size, hidden_dim * num_directions]
        
        # 4. Dropout and fully-connected layer
        dropped = self.dropout(hidden)
        logits = self.fc(dropped) # -> [batch_size, num_classes]
        
        return logits

if __name__ == '__main__':
    # --- Hyperparameters for testing ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 2
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT_PROB = 0.5
    
    # Create a model instance
    model = TextRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout_prob=DROPOUT_PROB
    )
    
    # Create some dummy input data
    BATCH_SIZE = 16
    SEQ_LEN = 50
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Get model output
    output = model(dummy_input)
    
    print("Model Architecture:")
    print(model)
    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("Test passed!")
