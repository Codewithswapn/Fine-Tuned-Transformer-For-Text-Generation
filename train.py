import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data_loader
import tokenizer
from model import SimpleTransformer

# Configuration
VOCAB_PATH = 'vocab.json'
MODEL_SAVE_PATH = 'small_transformer.pth'
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 256
NUM_LAYERS = 3
NUM_HEADS = 8
MAX_SEQUENCE_LENGTH = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 20
DROPOUT_RATE = 0.1  

class SimpleTextDataset(Dataset):
    def __init__(self, token_ids, sequence_length):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.token_ids) - self.sequence_length - 1

    def __getitem__(self, idx):
        input_seq = self.token_ids[idx : idx + self.sequence_length]
        target_seq = self.token_ids[idx + 1 : idx + self.sequence_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_data = data_loader.load_and_preprocess()
    
    if os.path.exists(VOCAB_PATH):
        vocab = tokenizer.load_vocab(VOCAB_PATH)
    else:
        vocab = tokenizer.build_vocab(text_data, vocab_size=MAX_VOCAB_SIZE)
        tokenizer.save_vocab(vocab, VOCAB_PATH)
    
    actual_vocab_size = len(vocab)
    token_ids = tokenizer.tokenize(text_data, vocab)

    dataset = SimpleTextDataset(token_ids, MAX_SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleTransformer(
        vocab_size=actual_vocab_size,
        embedding_size=EMBEDDING_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("Starting training...")

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, actual_vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved")


train()