import os
import re
import zipfile
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json

# --- Constants ---
DATA_DIR = "data"
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
GLOVE_URL_TEMPLATE = "http://nlp.stanford.edu/data/glove.6B.{dim}d.zip" # e.g., glove.6B.100d.zip


# --- Data Handling Functions ---

def download_and_unzip(url, dir_path):
    """
    通用下载和解压函数
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    
    # 检查解压后的文件夹/文件是否已存在
    unzipped_name = filename.replace('.zip', '')
    if os.path.exists(os.path.join(dir_path, unzipped_name)) or \
       os.path.exists(os.path.join(dir_path, "snli_1.0")): # SNLI 特有的文件夹名
        print(f"'{unzipped_name}' already exists. Skipping download and extraction.")
        return os.path.join(dir_path, "snli_1.0") if "snli" in url else os.path.join(dir_path, unzipped_name)

    # 下载
    print(f"Downloading {filename}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in r.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

    # 解压
    print(f"Extracting {filename}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return None
    finally:
        # 删除zip文件以节省空间
        os.remove(filepath)
    
    return os.path.join(dir_path, "snli_1.0") if "snli" in url else os.path.join(dir_path, unzipped_name)


class SNLIDataset(Dataset):
    """
    用于SNLI数据集的自定义Dataset类
    """
    def __init__(self, file_path, vocab, max_len=50):
        self.vocab = vocab
        self.max_len = max_len
        self.label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
        
        self.premises = []
        self.hypotheses = []
        self.labels = []
        
        print(f"Loading and processing data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                # 跳过没有共识的标签
                if data['gold_label'] not in self.label_map:
                    continue
                
                premise = self.vocab.text_to_sequence(self.clean_text(data['sentence1']), self.max_len)
                hypothesis = self.vocab.text_to_sequence(self.clean_text(data['sentence2']), self.max_len)
                label = self.label_map[data['gold_label']]
                
                self.premises.append(premise)
                self.hypotheses.append(hypothesis)
                self.labels.append(label)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9(),!?\'\`]", " ", text)
        # ... (可以添加更多清洗规则)
        return text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'premise': torch.tensor(self.premises[idx], dtype=torch.long),
            'hypothesis': torch.tensor(self.hypotheses[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class Vocabulary:
    """
    简化的词汇表类
    """
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.word_counts = Counter()

    def build_vocab(self, file_path, min_freq=5):
        print("Building vocabulary...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                if data['gold_label'] not in ["entailment", "contradiction", "neutral"]:
                    continue
                self.word_counts.update(self.clean_text(data['sentence1']).split())
                self.word_counts.update(self.clean_text(data['sentence2']).split())
        
        # Add words that meet the min_freq threshold
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"Built vocabulary with {len(self.word2idx)} words.")
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9(),!?\'\`]", " ", text)
        return text

    def text_to_sequence(self, text, max_len=None):
        words = text.split()
        seq = [self.word2idx.get(word, 1) for word in words] # 1 is <unk>
        if max_len:
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq += [0] * (max_len - len(seq)) # 0 is <pad>
        return seq

    def __len__(self):
        return len(self.word2idx)


def get_snli_loaders(batch_size, max_len=50, min_freq=5):
    """
    主函数：下载、处理数据并返回DataLoaders
    """
    snli_folder = download_and_unzip(SNLI_URL, DATA_DIR)
    if not snli_folder:
        raise RuntimeError("Failed to download or extract SNLI dataset.")
        
    snli_path = os.path.join(snli_folder, "snli_1.0")
    train_file = os.path.join(snli_path, "snli_1.0_train.jsonl")
    dev_file = os.path.join(snli_path, "snli_1.0_dev.jsonl")
    test_file = os.path.join(snli_path, "snli_1.0_test.jsonl")

    # Build vocabulary from training data
    vocab = Vocabulary()
    vocab.build_vocab(train_file, min_freq=min_freq)
    
    # Create datasets
    train_dataset = SNLIDataset(train_file, vocab, max_len)
    dev_dataset = SNLIDataset(dev_file, vocab, max_len)
    test_dataset = SNLIDataset(test_file, vocab, max_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, dev_loader, test_loader, vocab


if __name__ == '__main__':
    # --- 测试 ---
    # 这个过程会比较慢，因为它会下载和处理整个数据集
    print("Testing SNLI data pipeline...")
    train_loader, dev_loader, test_loader, vocab = get_snli_loaders(batch_size=4, max_len=20)
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # 获取一个batch的数据
    sample_batch = next(iter(train_loader))
    premise = sample_batch['premise']
    hypothesis = sample_batch['hypothesis']
    label = sample_batch['label']
    
    print("\nSample batch shapes:")
    print("Premise shape:", premise.shape)
    print("Hypothesis shape:", hypothesis.shape)
    print("Label shape:", label.shape)
    
    print("\nSample premise vector:", premise[0])
    print("Corresponding label:", label[0].item())

