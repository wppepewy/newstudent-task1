import argparse
import time
import torch
import torch.nn as nn

# Import from our own modules
import utils
from models.esim import ESIM

def train(model, train_loader, dev_loader, optimizer, criterion, epochs, device, clip=5.0):
    """
    Training loop for the ESIM model.
    """
    model.to(device)
    
    best_dev_acc = 0.0
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        for i, batch in enumerate(train_loader):
            premise = batch['premise'].to(device)
            hypothesis = batch['hypothesis'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(premise, hypothesis)
            
            loss = criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted_labels = torch.max(predictions, 1)
            correct_preds += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            if i % 100 == 0 and i > 0:
                print(f"  Batch {i}/{len(train_loader)} | "
                      f"Avg Train Loss: {total_loss / (i+1):.4f} | "
                      f"Train Acc: {correct_preds / total_samples * 100:.2f}%")

        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate on the validation set
        dev_acc, dev_loss = evaluate(model, dev_loader, criterion, device)
        
        print("-" * 50)
        print(f'End of Epoch {epoch:03d} | Time: {time.time() - epoch_start_time:.2f}s | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val. Loss: {dev_loss:.4f} | Val. Acc: {dev_acc*100:.2f}%')
        print("-" * 50)
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("Found new best model. Saving...")
            torch.save(model.state_dict(), 'esim-best-model.pt')

def evaluate(model, data_loader, criterion, device):
    """
    Evaluation function.
    """
    model.eval()
    total_loss = 0
    correct_preds = 0
    
    with torch.no_grad():
        for batch in data_loader:
            premise = batch['premise'].to(device)
            hypothesis = batch['hypothesis'].to(device)
            labels = batch['label'].to(device)

            predictions = model(premise, hypothesis)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            correct_preds += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / len(data_loader.dataset)
    return accuracy, avg_loss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, dev_loader, test_loader, vocab = utils.get_snli_loaders(
        batch_size=args.batch_size, 
        max_len=args.max_len,
        min_freq=args.min_freq
    )

    # Initialize model
    model = ESIM(
        vocab_size=len(vocab),
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=3, # entailment, contradiction, neutral
        dropout_prob=args.dropout
    )
    
    # Initialize weights (optional but good practice)
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, dev_loader, optimizer, criterion, args.epochs, device, args.clip)

    # Load the best model and evaluate on the test set
    print("\nLoading best model and evaluating on the test set...")
    model.load_state_dict(torch.load('esim-best-model.pt'))
    test_acc, test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESIM for Natural Language Inference')
    
    # --- Arguments ---
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--min_freq', type=int, default=5, help="Min word frequency to be in vocab")
    parser.add_argument('--clip', type=float, default=5.0, help="Gradient clipping")

    args = parser.parse_args()
    main(args)
