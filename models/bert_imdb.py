# -*- coding: utf-8 -*-
"""
BERT on IMDb
"""

!pip install --upgrade transformers datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset = load_dataset("imdb")
train_texts = dataset['train']['text'][:1500]
train_labels = dataset['train']['label'][:1500]
test_texts = dataset['test']['text'][:500]
test_labels = dataset['test']['label'][:500]

print(f"Train: {len(train_texts)}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, labels, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text in tqdm(texts, desc="Encoding"):
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)

train_ids, train_masks, train_labels = encode_texts(train_texts, train_labels, tokenizer)
test_ids, test_masks, test_labels = encode_texts(test_texts, test_labels, tokenizer)

class IMDbDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.ids[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

batch_size = 8
train_dataset = IMDbDataset(train_ids, train_masks, train_labels)
test_dataset = IMDbDataset(test_ids, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(ids, attention_mask=mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    return accuracy, precision, recall, f1

test_acc, precision, recall, f1 = evaluate(model, test_loader, device)
print(f"
Test Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
