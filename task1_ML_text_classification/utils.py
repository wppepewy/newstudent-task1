import os
import re
import numpy as np
from collections import Counter

# --- Hardcoded Data Samples ---
# 这只是一个小子集，用于演示和测试。
# 完整的 rt-polarity.pos 和 rt-polarity.neg 文件可以在网上找到，例如：
# https://github.com/yoonkim/CNN_sentence/blob/master/rt-polarity.pos
# https://github.com/yoonkim/CNN_sentence/blob/master/rt-polarity.neg

POSITIVE_EXAMPLES = [
    "the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .",
    "the gorgeously elaborate continuation of \" the lord of the rings \" trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .",
    "effective but too-tepid biopic",
    "if you sometimes like to go to the movies to have fun , wasabi is a good place to start .",
    "emerges as something rare , an issue movie that's so honest and keenly observed that it doesn't feel like one .",
]

NEGATIVE_EXAMPLES = [
    "simplistic , silly and tedious .",
    "it's so laddish and juvenile , only teenage boys could possibly find it funny .",
    "exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .",
    "a visually flashy but narratively opaque and emotionally vapid adaptation of shanghainese author wei hui's controversial novel .",
    "the story is also as unoriginal as they come , already having been recycled more times than i can count .",
]
# --- End of Hardcoded Data ---


def clean_str(string):
    """
    Tokenization/string cleaning.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    """
    Loads MR polarity data from hardcoded variables, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from hardcoded lists
    positive_examples = [s.strip() for s in POSITIVE_EXAMPLES]
    negative_examples = [s.strip() for s in NEGATIVE_EXAMPLES]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def build_vocab(sentences, max_words=10000):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    """
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence.split())
    
    vocabulary_inv = [x[0] for x in word_counts.most_common(max_words)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary

def vectorize_sentences(sentences, vocabulary):
    """
    Converts sentences to bag-of-words vectors.
    """
    vectorized_sentences = []
    vocab_size = len(vocabulary)
    for sentence in sentences:
        vector = np.zeros(vocab_size)
        for word in sentence.split():
            if word in vocabulary:
                vector[vocabulary[word]] += 1
        vectorized_sentences.append(vector)
    return np.array(vectorized_sentences)


def load_rotten_tomatoes_data(max_words=10000):
    """
    Loads and preprocessed the Rotten Tomatoes dataset from hardcoded data.
    Returns train, validation, and test sets.
    """
    x_text, y = load_data_and_labels()

    # Build vocabulary
    vocab = build_vocab(x_text, max_words=max_words)

    # Vectorize sentences
    x = vectorize_sentences(x_text, vocab)

    # Shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # Using a 90/10 split for this small dataset
    dev_sample_index = -1 * int(0.9 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    # Use the dev set as the test set as well
    x_test, y_test = x_dev, y_dev

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev, x_test, y_test, vocab

if __name__ == '__main__':
    # Example usage:
    x_train, y_train, x_dev, y_dev, x_test, y_test, vocab = load_rotten_tomatoes_data()
    if x_train.size > 0:
        print("\nSample training data vector (first 50 features):")
        print(x_train[0][:50])
        print("\nLabel for the sample:")
        print(y_train[0])
        print("\nA word from vocab:", list(vocab.keys())[5])
    else:
        print("Training data is empty!")

