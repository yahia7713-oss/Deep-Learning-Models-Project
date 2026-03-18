# -*- coding: utf-8 -*-
"""
RNN/LSTM on IMDb
"""

!pip install --upgrade datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import re
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset = load_dataset("imdb")
train_texts = dataset['train']['text'][:20000]
train_labels = dataset['train']['label'][:20000]
test_texts = dataset['test']['text'][:5000]
test_labels = dataset['test']['label'][:5000]

print(f"Train: {len(train_texts)}")
print(f"Test: {len(test_texts)}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

train_clean = [clean_text(t) for t in train_texts]
test_clean = [clean_text(t) for t in test_texts]

def build_vocab(texts, max_words=10000):
    word_counter = Counter()
    for text in texts:
        word_counter.update(text.split())
    
    most_common = word_counter.most_common(max_words - 2)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    
    return word2idx

vocab = build_vocab(train_clean)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

def text_to_sequence(text, vocab, max_len=100):
    words = text.split()
    seq = [vocab.get(w, vocab['<UNK>']) for w in words[:max_len]]
    
    if len(seq) < max_len:
        seq += [vocab['<PAD>']] * (max_len - len(seq))
    
    return seq

X_train = [text_to_sequence(t, vocab) for t in train_clean]
X_test = [text_to_sequence(t, vocab) for t in test_clean]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, train_labels, test_size=0.2, random_state=42
)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.LongTensor(texts)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

batch_size = 64
train_dataset = IMDBDataset(X_train, y_train)
val_dataset = IMDBDataset(X_val, y_val)
test_dataset = IMDBDataset(X_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze()

model = LSTMModel(vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in tqdm(loader, desc="Training"):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    
    with torch.no_grad():
        for texts, labels in tqdm(loader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds

epochs = 10
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(epochs):
    print(f"
Epoch {epoch+1}/{epochs}")
    print("-" * 50)
    
    start = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss, val_acc, val_preds = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    pos_pct = np.mean(val_preds) * 100
    epoch_time = time.time() - start
    print(f"Time: {epoch_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"Predictions: {pos_pct:.1f}% Positive")

test_loss, test_acc, test_preds = evaluate(model, test_loader, criterion)
print(f"
Final Test Accuracy: {test_acc:.2f}%")
