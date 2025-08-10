import argparse
import time
import torch
import torch.optim as optim
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# Import from our own modules
import utils
from models.bilstm_crf import BiLSTM_CRF

def train(model, train_loader, dev_loader, optimizer, device, epochs, idx2tag):
    """
    Training loop for the BiLSTM-CRF model.
    """
    model.to(device)
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0
        
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            tags = batch['tags'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            
            # Forward pass to get loss
            loss = model(tokens, tags, mask, is_test=False)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate on the validation set
        dev_f1, dev_precision, dev_recall = evaluate(model, dev_loader, device, idx2tag)
        
        print("-" * 50)
        print(f'End of Epoch {epoch:03d} | Time: {time.time() - epoch_start_time:.2f}s | '
              f'Train Loss: {avg_train_loss:.4f}')
        print(f'Dev F1: {dev_f1:.4f} | Dev Precision: {dev_precision:.4f} | Dev Recall: {dev_recall:.4f}')
        print("-" * 50)

def evaluate(model, data_loader, device, idx2tag):
    """
    Evaluation function.
    """
    model.eval()
    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            tags = batch['tags'].to(device)
            mask = batch['mask'].to(device)
            
            # Decode to get predicted tags
            predicted_paths = model(tokens, tags, mask, is_test=True)
            
            # Convert true tags and predicted tags to string labels
            for i in range(len(tags)):
                true_len = int(mask[i].sum())
                true_tags = [idx2tag[tag.item()] for tag in tags[i][:true_len]]
                
                # predicted_paths is a list of lists, we need to access the i-th list
                pred_tags = [idx2tag[tag_idx] for tag_idx in predicted_paths[i]]
                
                all_true_tags.append(true_tags)
                all_pred_tags.append(pred_tags)

    f1 = f1_score(all_true_tags, all_pred_tags)
    precision = precision_score(all_true_tags, all_pred_tags)
    recall = recall_score(all_true_tags, all_pred_tags)
    
    return f1, precision, recall

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, word2idx, tag2idx, idx2tag = utils.get_conll_loaders(
        batch_size=args.batch_size,
        min_freq=args.min_freq
    )
    
    # Initialize model
    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tag_to_idx=tag2idx,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, val_loader, optimizer, device, args.epochs, idx2tag)
    
    # Final evaluation on the test set
    print("\nFinal evaluation on the test set...")
    test_f1, test_precision, test_recall = evaluate(model, test_loader, device, idx2tag)
    print(f'Test F1: {test_f1:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Named Entity Recognition')
    
    # --- Arguments ---
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--min_freq', type=int, default=2, help="Min word frequency for vocab")

    args = parser.parse_args()
    main(args)
