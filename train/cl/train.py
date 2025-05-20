import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging

from models.embedding import ContrastiveTransformerModel
from utils import save_utils

logging.set_verbosity_error()  # Silence excessive warnings


# 1. Dataset
class IPAContrastiveDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples = save_utils.load(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        bai_ipa = rec['ipa']
        en = rec['en']
        return en, bai_ipa


# 2. Collate function: batch Tokenization + Padding
def collate_batch(batch, tokenizer, max_length=32):
    en, bai_ipa = zip(*batch)
    en_inputs = tokenizer(
        list(en),
        padding='longest',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    bai_inputs = tokenizer(
        list(bai_ipa),
        padding='longest',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return en_inputs, bai_inputs


# 4. Compute InfoNCE contrastive loss
def contrastive_loss(zh_emb, en_emb, temperature=0.05):
    """
    zh_emb, en_emb: [B, D], already L2 normalized
    Use Multiple Negatives Ranking Loss: treat all other samples in the batch as negatives.
    """
    # L2 normalization
    zh_norm = nn.functional.normalize(zh_emb, dim=1)  # [B, D]
    en_norm = nn.functional.normalize(en_emb, dim=1)  # [B, D]

    # For positive pair i, zh_emb[i] should match en_emb[i]; others are negatives
    logits = torch.matmul(zh_norm, en_norm.T)  # [B, B]
    logits = logits / temperature

    labels = torch.arange(logits.size(0)).to(logits.device)

    # Compute loss for both zh->en and en->zh directions
    loss_zh2en = nn.CrossEntropyLoss()(logits, labels)
    loss_en2zh = nn.CrossEntropyLoss()(logits.T, labels)

    return (loss_zh2en + loss_en2zh) / 2


# 5. Training and Evaluation
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for en_inputs, bai_inputs in tqdm(dataloader):
        # Move inputs to device
        en_inputs = {k: v.to(device) for k, v in en_inputs.items()}
        bai_inputs = {k: v.to(device) for k, v in bai_inputs.items()}

        optimizer.zero_grad()
        en_emb = model(en_inputs)  # [B, D]
        bai_emb = model(bai_inputs)  # [B, D]
        loss = contrastive_loss(en_emb, bai_emb, temperature=0.05)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for en_inputs, bai_inputs in tqdm(dataloader):
            en_inputs = {k: v.to(device) for k, v in en_inputs.items()}
            bai_inputs = {k: v.to(device) for k, v in bai_inputs.items()}

            en_emb = model(en_inputs)  # [B, D]
            bai_emb = model(bai_inputs)  # [B, D]

            # L2 normalize
            en_emb = nn.functional.normalize(en_emb, dim=1)
            bai_emb = nn.functional.normalize(bai_emb, dim=1)

            # Similarity matrix [B, B]: en rows vs bai columns
            sims = torch.matmul(en_emb, bai_emb.T)

            # en → bai: prediction is the index of the max value in each row
            preds = torch.argmax(sims, dim=1)
            labels = torch.arange(sims.size(0), device=device)

            correct += (preds == labels).sum().item()
            total += sims.size(0)

    acc = correct / total
    return acc


# 6. Main Training Flow
if __name__ == '__main__':
    TRAIN_DATA_PATH = '../datasets/SOLAN/bfs_language_zh_base_train.json'
    VALID_DATA_PATH = '../datasets/SOLAN/bfs_language_zh_base_valid.json'
    MODEL_NAME = '../../models/bert-base-uncased'  # Or any preferred Transformer model
    BATCH_SIZE = 32
    LR = 2e-5
    EPOCHS = 64
    MAX_LEN = 32
    best_acc = 0.0

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_dataset = IPAContrastiveDataset(TRAIN_DATA_PATH)
    valid_dataset = IPAContrastiveDataset(VALID_DATA_PATH)
    collate = lambda b: collate_batch(b, tokenizer, max_length=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(valid_dataset, batch_size=189, shuffle=False, collate_fn=collate)

    # Model & Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveTransformerModel(MODEL_NAME, pooling='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Training & Validation
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f}, val_acc: {val_acc:.4f}')
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.backbone.save_pretrained('./saved_model')
            print(f'>>> New best model saved with acc: {best_acc:.4f}')
