# 8_arhitectura_finala.py

import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 1. CONFIGURARE
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "7_output_dataset_clean.csv")
classes_file = os.path.join(base_path, "classes_encoder.npy")

# Fortam CPU 
device = torch.device("cpu")
print("\n" + "="*60)
print(" Începem Pasul 8: ANTRENAMENTUL ARHITECTURII FINALE (10 EPOCI)")
print("="*60)
print(f"Device: {device}")

if not os.path.exists(input_file):
    print(f"EROARE: Nu găsesc '{input_file}'. Rulează 7_cleaner_consens.py mai întâi.")
    exit()

# 2. PREGĂTIRE DATE CURATE
df = pd.read_csv(input_file)
if os.path.exists(classes_file):
    class_names = np.load(classes_file, allow_pickle=True)
else:
    print("AVERTISMENT: Nu am găsit classes_encoder.npy, folosesc clase numerice.")
    class_names = [str(i) for i in sorted(df['label'].unique())]

NUM_LABELS = len(class_names)
print(f"--> Antrenăm pe {len(df)} paragrafe curățate (Clase: {NUM_LABELS})")

MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 10 # Setat pe 10 pentru curba de loss

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

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
train_ds = ParagraphDataset(train_texts, train_labels, tokenizer)
val_ds = ParagraphDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS).to(device)

# INGHETARE BAZA BERT PENTRU VITEZA MAXIMA PE CPU
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# --- 3. FUNCȚIE ANTRENARE ---
def train_final_model(model, train_loader, optimizer, scheduler):
    start_time = time.time()
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        print(f"\n[Model Suprem] Epoca {epoch+1}/{EPOCHS}...")
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if step % 50 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"   Pas {step}/{len(train_loader)} | Loss: {loss.item():.4f} | Timp: {elapsed:.0f}s")

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f">>> FINAL Epoca {epoch+1} - Loss Mediu: {avg_loss:.4f}")
    
    return loss_history

train_loss_history = train_final_model(model, train_loader, optimizer, scheduler)

# --- 4. EVALUARE ȘI EXTRAGERE REZULTATE ---
print("\n--> Generăm rezultatele și graficele finale pe setul de validare...")
model.eval()
preds, true_labels, all_logits = [], [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch["labels"].numpy())
        all_logits.extend(logits.cpu().numpy())

y_test_arr = np.array(true_labels)
y_score_arr = np.array(all_logits)

# --- 5. GENERARE GRAFICE ---

# Grafic 1: Matricea de Confuzie (CLARĂ)
cm = confusion_matrix(true_labels, preds, labels=range(NUM_LABELS))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Model Final (Date Curățate) - Matrice de Confuzie", fontsize=12)
plt.xlabel("Prezis")
plt.ylabel("Adevărat")
plt.xticks(rotation=45, ha='right')
save_cm = os.path.join(base_path, "8_output_Final_ConfusionMatrix.png")
plt.savefig(save_cm, bbox_inches='tight', dpi=300)
plt.close()

# Grafic 2: Curba de Învățare (Loss Curve)
plt.figure(figsize=(7,5))
epochs_axis = np.arange(1, EPOCHS + 1)
plt.plot(epochs_axis, train_loss_history, marker="o", color="blue", linewidth=1.5, label="Training Loss")

if len(train_loss_history) > 1:
    plt.axhline(y=train_loss_history[-1], color='r', linestyle=':', label='Plajă (Valoare Finală)')

plt.title(f"BERT Base — Evoluția pierderii (Training Loss) pe {EPOCHS} epoci", fontsize=12)
plt.xlabel("Epocă")
plt.ylabel("Pierdere (Loss)")
plt.xticks(epochs_axis)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
save_loss = os.path.join(base_path, "8_output_Final_LossCurve.png")
plt.savefig(save_loss, bbox_inches='tight', dpi=300)
plt.close()

# Grafic 3: Curba ROC - CLARĂ ȘI SIMPLĂ (Linii Solide)
y_test_bin = label_binarize(y_test_arr, classes=range(NUM_LABELS))
probs = torch.nn.functional.softmax(torch.tensor(y_score_arr), dim=1).numpy()

plt.figure(figsize=(8, 6))
# Culori 
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan'])

for i, color in zip(range(NUM_LABELS), colors):
    if i < len(class_names) and np.sum(y_test_bin[:, i]) > 0:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        # Linia solidă, groasă (lw=2)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:0.2f})')

# Linia neagră întreruptă de la mijloc (Random Guess)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Setări axe 
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Rata Fals Pozitivă (FPR)')
plt.ylabel('Rata Adevărat Pozitivă (TPR)')
plt.title('Curba ROC (Model Final Pe Date Curățate)')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

save_roc = os.path.join(base_path, "8_output_Final_ROC.png")
plt.savefig(save_roc, bbox_inches='tight', dpi=300)
plt.close()

# Salvare Raport Text ca imagine
report_str = classification_report(
    true_labels, 
    preds, 
    labels=range(NUM_LABELS), 
    target_names=class_names, 
    zero_division=0
)
plt.figure(figsize=(10, 6))
plt.axis('off')
plt.text(0.01, 0.95, f"Raport Clasificare - Model Final (Consens)\n\n{report_str}", 
         fontfamily='monospace', fontsize=12, verticalalignment='top')
save_text = os.path.join(base_path, "8_output_Final_Report.png")
plt.savefig(save_text, bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "="*60)
print(" PROIECT FINALIZAT ! ")
print(" Graficele au fost generate")
print("="*60)