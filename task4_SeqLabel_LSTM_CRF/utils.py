import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

class NerDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "tokens": self.sentences[idx],
            "tags": self.tags[idx]
        }

def build_vocab_and_tags(data, min_freq=1):
    """
    从数据集中构建词汇表和标签映射
    """
    word_counts = Counter()
    tag_set = set()

    for item in data:
        word_counts.update([word.lower() for word in item['tokens']])
        tag_set.update(item['ner_tags'])

    # Word vocabulary
    word2idx = {"<pad>": 0, "<unk>": 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    
    # Tag mapping
    # CoNLL2003 tags are already integers, but let's create a mapping for clarity
    # and to ensure a consistent order.
    # We can get the tag names from the dataset features.
    # 0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC, 7: B-MISC, 8: I-MISC
    tag_names = data.features['ner_tags'].feature.names
    tag2idx = {tag: i for i, tag in enumerate(tag_names)}
    idx2tag = {i: tag for i, tag in enumerate(tag_names)}

    return word2idx, tag2idx, idx2tag


def prepare_data(dataset, word2idx, tag2idx):
    """
    将文本数据转换为索引序列
    """
    sentences = []
    tags = []
    for item in dataset:
        sentence_indices = [word2idx.get(word.lower(), 1) for word in item['tokens']]
        tag_indices = item['ner_tags']
        sentences.append(torch.tensor(sentence_indices, dtype=torch.long))
        tags.append(torch.tensor(tag_indices, dtype=torch.long))
    return sentences, tags

def collate_fn(batch):
    """
    自定义的collate_fn来处理padding
    """
    tokens = [item['tokens'] for item in batch]
    tags = [item['tags'] for item in batch]
    
    # Pad sequences
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0) # Pad tags with 0 ('O' tag)
    
    # Create a mask for the actual (non-padded) tokens
    mask = (padded_tokens != 0)
    
    return {"tokens": padded_tokens, "tags": padded_tags, "mask": mask}


def get_conll_loaders(batch_size, min_freq=1):
    """
    加载CoNLL 2003数据集并返回DataLoaders
    """
    print("Loading CoNLL 2003 dataset from Hugging Face...")
    dataset = load_dataset("conll2003")

    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']
    
    print("Building vocabulary and tag mappings...")
    word2idx, tag2idx, idx2tag = build_vocab_and_tags(train_data, min_freq=min_freq)
    
    print("Preparing data...")
    train_sentences, train_tags = prepare_data(train_data, word2idx, tag2idx)
    val_sentences, val_tags = prepare_data(val_data, word2idx, tag2idx)
    test_sentences, test_tags = prepare_data(test_data, word2idx, tag2idx)
    
    train_dataset = NerDataset(train_sentences, train_tags)
    val_dataset = NerDataset(val_sentences, val_tags)
    test_dataset = NerDataset(test_sentences, test_tags)
    
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, word2idx, tag2idx, idx2tag


if __name__ == '__main__':
    # --- 测试 ---
    # 这将下载CoNLL 2003数据集 (约6MB)
    train_loader, val_loader, test_loader, word2idx, tag2idx, idx2tag = get_conll_loaders(batch_size=4, min_freq=2)
    
    print(f"\nVocabulary size: {len(word2idx)}")
    print(f"Number of tags: {len(tag2idx)}")
    
    # 获取一个batch的数据
    sample_batch = next(iter(train_loader))
    tokens = sample_batch['tokens']
    tags = sample_batch['tags']
    mask = sample_batch['mask']
    
    print("\nSample batch shapes:")
    print("Tokens shape:", tokens.shape)
    print("Tags shape:", tags.shape)
    print("Mask shape:", mask.shape)
    
    print("\nSample tokens vector:", tokens[0])
    print("Sample tags vector:  ", tags[0])
    print("\nTag mapping (first 5):")
    for i in range(5):
        print(f"  {i}: {idx2tag[i]}")

