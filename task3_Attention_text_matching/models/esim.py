import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEncoder(nn.Module):
    """
    使用Bi-LSTM对输入句子进行编码
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(InputEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, lengths):
        x = self.dropout(self.embedding(x))
        # Pack sequence to handle variable lengths
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output

class SoftmaxAttention(nn.Module):
    """
    计算前提和假设之间的注意力权重
    """
    def forward(self, premise_encoding, hypothesis_encoding, premise_mask, hypothesis_mask):
        # premise_encoding: [batch_size, premise_len, 2*hidden_dim]
        # hypothesis_encoding: [batch_size, hypothesis_len, 2*hidden_dim]
        
        # 计算相似度矩阵
        similarity_matrix = torch.bmm(premise_encoding, hypothesis_encoding.transpose(1, 2))
        
        # 应用 mask
        premise_mask = premise_mask.unsqueeze(2).expand_as(similarity_matrix)
        hypothesis_mask = hypothesis_mask.unsqueeze(1).expand_as(similarity_matrix)
        
        masked_similarity = similarity_matrix.masked_fill(~premise_mask, -1e10)
        masked_similarity = masked_similarity.masked_fill(~hypothesis_mask, -1e10)

        # Softmax over columns (attention for premise)
        attn_premise = F.softmax(masked_similarity, dim=2)
        
        # Softmax over rows (attention for hypothesis)
        attn_hypothesis = F.softmax(masked_similarity, dim=1)
        
        # 计算对齐后的向量
        premise_aligned = torch.bmm(attn_premise, hypothesis_encoding)
        hypothesis_aligned = torch.bmm(attn_hypothesis.transpose(1, 2), premise_encoding)
        
        return premise_aligned, hypothesis_aligned


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_prob):
        super(ESIM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = InputEncoder(vocab_size, embedding_dim, hidden_dim, dropout_prob)
        self.attention = SoftmaxAttention()
        
        self.composition_lstm = nn.LSTM(hidden_dim * 8, hidden_dim, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, premise, hypothesis):
        premise_lengths = (premise != 0).sum(dim=1)
        hypothesis_lengths = (hypothesis != 0).sum(dim=1)
        
        premise_mask = (premise != 0)
        hypothesis_mask = (hypothesis != 0)

        # 1. Input Encoding
        premise_encoded = self.encoder(premise, premise_lengths)
        hypothesis_encoded = self.encoder(hypothesis, hypothesis_lengths)
        
        # 2. Local Inference (Attention)
        premise_aligned, hypothesis_aligned = self.attention(
            premise_encoded, hypothesis_encoded, premise_mask, hypothesis_mask
        )
        
        # 3. Inference Composition
        m_premise = torch.cat([premise_encoded, premise_aligned, 
                               premise_encoded - premise_aligned, 
                               premise_encoded * premise_aligned], dim=2)
        m_hypothesis = torch.cat([hypothesis_encoded, hypothesis_aligned,
                                  hypothesis_encoded - hypothesis_aligned,
                                  hypothesis_encoded * hypothesis_aligned], dim=2)
        
        # Pack sequences for composition LSTM
        packed_premise = nn.utils.rnn.pack_padded_sequence(m_premise, premise_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_hypothesis = nn.utils.rnn.pack_padded_sequence(m_hypothesis, hypothesis_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        _, (premise_ht, _) = self.composition_lstm(packed_premise)
        _, (hypothesis_ht, _) = self.composition_lstm(packed_hypothesis)
        
        # Concatenate final hidden states of both directions
        v_premise = torch.cat([premise_ht[0], premise_ht[1]], dim=1)
        v_hypothesis = torch.cat([hypothesis_ht[0], hypothesis_ht[1]], dim=1)
        
        # 4. Prediction
        v_final = torch.cat([v_premise, v_hypothesis], dim=1)
        
        logits = self.classifier(v_final)
        return logits

if __name__ == '__main__':
    # --- Hyperparameters for testing ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 3
    DROPOUT_PROB = 0.5
    BATCH_SIZE = 16
    SEQ_LEN = 20
    
    # Create a model instance
    model = ESIM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT_PROB)
    
    # Create dummy input data
    premise_dummy = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    hypothesis_dummy = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    # Make some of them padded
    premise_dummy[0, -5:] = 0
    hypothesis_dummy[1, -10:] = 0

    # Get model output
    output = model(premise_dummy, hypothesis_dummy)
    
    print("Model Architecture:")
    print(model)
    print("\nInput premise shape:", premise_dummy.shape)
    print("Input hypothesis shape:", hypothesis_dummy.shape)
    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("Test passed!")
