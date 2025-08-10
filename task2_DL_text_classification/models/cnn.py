import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, filter_sizes, dropout_prob):
        """
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词向量维度
        :param num_classes: 类别数
        :param num_filters: 每种尺寸卷积核的数量
        :param filter_sizes: 卷积核尺寸的列表 (e.g., [3, 4, 5])
        :param dropout_prob: dropout的概率
        """
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, 
                         out_channels=num_filters, 
                         kernel_size=(fs, embedding_dim)) 
             for fs in filter_sizes]
        )
        
        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # 1. Embedding
        embedded = self.embedding(x)  # -> [batch_size, seq_len, embedding_dim]
        
        # 2. Add a channel dimension for Conv2d
        embedded = embedded.unsqueeze(1)  # -> [batch_size, 1, seq_len, embedding_dim]
        
        # 3. Convolution + ReLU + Max-pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i]: [batch_size, num_filters, seq_len - filter_sizes[i] + 1, 1]
        
        pooled = [F.max_pool1d(conv.squeeze(3), conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[i]: [batch_size, num_filters]
        
        # 4. Concatenate pooled features
        cat = torch.cat(pooled, dim=1)  # -> [batch_size, num_filters * len(filter_sizes)]
        
        # 5. Dropout and fully-connected layer
        cat = self.dropout(cat)
        logits = self.fc(cat)  # -> [batch_size, num_classes]
        
        return logits

if __name__ == '__main__':
    # --- Hyperparameters for testing ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 100
    NUM_CLASSES = 2
    NUM_FILTERS = 128
    FILTER_SIZES = [3, 4, 5]
    DROPOUT_PROB = 0.5
    
    # Create a model instance
    model = TextCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
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

