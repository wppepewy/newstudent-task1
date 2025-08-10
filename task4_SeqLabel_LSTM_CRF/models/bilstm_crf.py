import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # 全连接层，将LSTM的输出映射到标签空间
        # 输出的是CRF层的 "emission scores"
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_lstm_features(self, sentence):
        """
        从BiLSTM获取发射分数
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, is_test=False):
        """
        前向传播
        - 在训练时，返回CRF层的损失
        - 在测试时，返回解码后的最优路径 (标签序列)
        """
        # 从BiLSTM获取发射分数
        emissions = self._get_lstm_features(sentence)

        if not is_test:
            # 训练模式：计算损失
            # CRF.forward() 返回的是对数似然损失的相反数，所以我们需要取负
            loss = -self.crf(emissions, tags, mask=mask.byte(), reduction='mean')
            return loss
        else:
            # 测试模式：维特比解码
            # CRF.decode() 返回一个包含最优标签序列的列表
            decoded_tags = self.crf.decode(emissions, mask=mask.byte())
            return decoded_tags

if __name__ == '__main__':
    # --- 测试参数 ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    # 假设的标签映射
    tag_to_idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "<PAD>": 5}
    
    # 创建模型实例
    model = BiLSTM_CRF(
        vocab_size=VOCAB_SIZE,
        tag_to_idx=tag_to_idx,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # 创建虚拟输入数据
    BATCH_SIZE = 16
    SEQ_LEN = 20
    dummy_sentence = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_tags = torch.randint(0, len(tag_to_idx)-1, (BATCH_SIZE, SEQ_LEN))
    dummy_mask = torch.ones_like(dummy_sentence, dtype=torch.bool)
    
    # -- 训练模式测试 --
    loss = model(dummy_sentence, dummy_tags, dummy_mask, is_test=False)
    print("Training Mode Test:")
    print("Calculated Loss:", loss.item())
    assert loss.item() > 0
    print("Train test passed.")

    # -- 解码模式测试 --
    decoded_paths = model(dummy_sentence, dummy_tags, dummy_mask, is_test=True)
    print("\nDecoding Mode Test:")
    print("Number of decoded paths:", len(decoded_paths))
    print("Length of first path:", len(decoded_paths[0]))
    assert len(decoded_paths) == BATCH_SIZE
    assert len(decoded_paths[0]) <= SEQ_LEN
    print("Decode test passed.")
    
    print("\nModel Architecture:")
    print(model)
