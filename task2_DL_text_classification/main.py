import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Import from our own modules
import utils
from models.cnn import TextCNN
from models.rnn import TextRNN

def train(model, train_loader, dev_loader, optimizer, criterion, epochs, device):
    """
    Training loop for the model.
    """
    model.to(device)
    
    best_dev_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train() # Set model to training mode
        epoch_start_time = time.time()
        total_loss = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(texts)
            
            loss = criterion(predictions, labels)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate on the validation set
        dev_acc, dev_loss = evaluate(model, dev_loader, criterion, device)
        
        print(f'Epoch {epoch:03d} | Time: {time.time() - epoch_start_time:.2f}s | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val. Loss: {dev_loss:.4f} | Val. Acc: {dev_acc*100:.2f}%')
        
        # Save the best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            # You can save the model weights here if needed
            # torch.save(model.state_dict(), 'best_model.pt')

def evaluate(model, data_loader, criterion, device):
    """
    Evaluation function.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            predictions = model(texts)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Get the predicted class
            _, predicted = torch.max(predictions, 1)
            correct_preds += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / len(data_loader.dataset)
    return accuracy, avg_loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    X, Y, vocab = utils.load_data(max_words=args.vocab_size, max_len=args.max_len)
    
    # Split data into training, validation, and test sets
    # 80% train, 10% validation, 10% test
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train size: {len(x_train)}")
    print(f"Validation size: {len(x_val)}")
    print(f"Test size: {len(x_test)}")

    # Create TensorDatasets and DataLoaders
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Initialize model
    if args.model_type.lower() == 'cnn':
        model = TextCNN(
            vocab_size=len(vocab),
            embedding_dim=args.embed_dim,
            num_classes=2,
            num_filters=args.num_filters,
            filter_sizes=[int(fs) for fs in args.filter_sizes.split(',')],
            dropout_prob=args.dropout
        )
        print("Initialized TextCNN model.")
    elif args.model_type.lower() == 'rnn':
        model = TextRNN(
            vocab_size=len(vocab),
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            bidirectional=True,
            dropout_prob=args.dropout
        )
        print("Initialized TextRNN model.")
    else:
        raise ValueError("Model type not supported. Choose 'cnn' or 'rnn'.")

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion, args.epochs, device)
    
    # Evaluate on the test set
    print("\nEvaluating on the test set...")
    test_acc, test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Classification with PyTorch')
    
    # General arguments
    parser.add_argument('--model_type', type=str, default='cnn', help='Model to use: "cnn" or "rnn"')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of word embeddings')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Maximum size of the vocabulary')
    parser.add_argument('--max_len', type=int, default=56, help='Maximum sequence length (sentences are padded/truncated to this)')

    # CNN specific arguments
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters for each CNN filter size')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='Comma-separated filter sizes for CNN')
    
    # RNN specific arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')

    args = parser.parse_args()
    main(args)
