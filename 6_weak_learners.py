# 6_weak_learners.py

import pandas as pd
import torch
import numpy as np
import os
import time
import gc

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#1. CONFIGURARE
base_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_path, "1_output_all_articles_paragraph_dataset_with_sections.csv")

device = torch.device("cpu") 
print(f"Device: {device} | Etapa de WEAK LEARNERS (TURBO Mode - Layer Freezing)")

# Deoarece am inghetat 95% din model, consuma mai putin RAM, deci pot mari Batch-ul la 4 pt viteza!
BATCH_SIZE = 4 
MAX_LEN = 128  
EPOCHS = 1 

# --- 2. PREGATIRE DATE ---
if not os.path.exists(filename):
    raise FileNotFoundError(f"Nu gasesc {filename}")

df = pd.read_csv(filename).dropna(subset=["section"]).copy()

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["section"])
NUM_LABELS = len(label_encoder.classes_)

np.save(os.path.join(base_path, "classes_encoder.npy"), label_encoder.classes_)

class ParagraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

texts = df["paragraph_text"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

def train_weak_learner(model, train_loader, optimizer, model_name):
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        print(f"\n[{model_name}] Epoca {epoch+1}/{EPOCHS}...")
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            if step % 10 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"   Pas {step}/{len(train_loader)} | Loss: {loss.item():.4f} | Timp: {elapsed:.0f}s")

def generate_predictions(model, dataloader, model_name):
    print(f"\n--> {model_name} genereaza predictiile finale...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1).numpy()
            all_preds.extend(preds)
    return all_preds

# --- 3. BERT WEAK LEARNER ---
print("\n" + "="*40 + "\n1. Descarc / Incarc BERT TURBO...\n" + "="*40)
bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_ds_bert = ParagraphDataset(train_texts, train_labels, bert_tok)
train_loader_bert = DataLoader(train_ds_bert, batch_size=BATCH_SIZE, shuffle=True)
full_ds_bert = ParagraphDataset(texts, labels, bert_tok)
full_loader_bert = DataLoader(full_ds_bert, batch_size=BATCH_SIZE, shuffle=False)

bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS).to(device)

# Înghețăm baza modelului BERT
for param in bert_model.bert.parameters():
    param.requires_grad = False

# Antrenăm doar ultimul strat (clasificatorul)
optim_bert = AdamW(filter(lambda p: p.requires_grad, bert_model.parameters()), lr=1e-3)

train_weak_learner(bert_model, train_loader_bert, optim_bert, "BERT")
df["bert_pred"] = generate_predictions(bert_model, full_loader_bert, "BERT")

print("\n--> Curatam memoria inainte de RoBERTa...")
del bert_model, optim_bert, train_loader_bert, full_loader_bert
gc.collect()

# 4. RoBERTa WEAK LEARNER
print("\n" + "="*40 + "\n2. Descarc / Incarc RoBERTa TURBO...\n" + "="*40)
rob_tok = RobertaTokenizer.from_pretrained("roberta-base")

train_ds_rob = ParagraphDataset(train_texts, train_labels, rob_tok)
train_loader_rob = DataLoader(train_ds_rob, batch_size=BATCH_SIZE, shuffle=True)
full_ds_rob = ParagraphDataset(texts, labels, rob_tok)
full_loader_rob = DataLoader(full_ds_rob, batch_size=BATCH_SIZE, shuffle=False)

rob_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_LABELS).to(device)

# Înghețăm baza modelului RoBERTa 
for param in rob_model.roberta.parameters():
    param.requires_grad = False

optim_rob = AdamW(filter(lambda p: p.requires_grad, rob_model.parameters()), lr=1e-3)

train_weak_learner(rob_model, train_loader_rob, optim_rob, "RoBERTa")
df["roberta_pred"] = generate_predictions(rob_model, full_loader_rob, "RoBERTa")

rob_out = os.path.join(base_path, "6_output_roberta_results.csv")
df.to_csv(rob_out, index=False)

print("\n" + "="*50)
print(f" Salvat in: {rob_out}")
print("="*50)