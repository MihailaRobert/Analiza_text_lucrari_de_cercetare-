# 5_roberta_binary.py

import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import time

# CONFIGURARE CALE
base_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_path, "1_output_all_articles_paragraph_dataset_with_sections.csv")

if not os.path.exists(filename):
    print(f"EROARE: Nu găsesc fișierul {filename}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rulez pe device: {device}")
print("Ptr 'cpu', antrenarea va dura mai mult. Am optimizat parametrii pentru laptop.)")


# 4.3 — RoBERTa Paragraph Classification (Introduction vs Conclusion)
# Load dataset
print("--> Incarc datele...")
df = pd.read_csv(filename)
df = df[df["section"].isin(["Introduction", "Conclusion"])]

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["section"])
num_classes = len(label_encoder.classes_)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Custom Dataset
class ParagraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256): # Redus la 256 pentru viteza/memorie
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["paragraph_text"].astype(str).tolist(),
    df["label"].tolist(),
    stratify=df["label"],
    test_size=0.2,
    random_state=42
)

# Batch size redus la 4 pentru a nu bloca CPU-ul
BATCH_SIZE = 4 
train_ds = ParagraphDataset(train_texts, train_labels, tokenizer)
val_ds = ParagraphDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# Model
print("--> Incarc modelul RoBERTa")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_classes
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)


# Training Loop
epochs = 3 # Pot reduce la 1 sau 2 daca dureaza prea mult
print(f"--> Incep antrenarea pentru {epochs} epoci...")
print(f"    (Total pasi per epoca: {len(train_loader)})")

start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    print(f"\n[Epoca {epoch+1}/{epochs}]")
    
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Afisam progresul la fiecare 20 de pasi 
        if step % 20 == 0 and step > 0:
            elapsed = time.time() - start_time
            print(f"   Pas {step}/{len(train_loader)} | Loss curent: {loss.item():.4f} | Timp scurs: {elapsed:.0f}s")

    avg_loss = total_loss / len(train_loader)
    print(f"   >>> FINAL EPOCA {epoch+1} - Average Loss: {avg_loss:.4f}")


# Evaluation
print("\n--> Evaluare pe datele de test...")
model.eval()
preds = []
true = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true.extend(batch["labels"].numpy())

# Salvare raport text ca imagine
print("--> Generare Raport Final...")
report_str = classification_report(true, preds, target_names=label_encoder.classes_)
print(report_str) # Afisare in consola

plt.figure(figsize=(10, 6))
plt.axis('off')
plt.text(0.01, 0.95, f"RoBERTa Binary Classification Report\n\n{report_str}", 
         fontfamily='monospace', fontsize=12, verticalalignment='top')
output_text_png = "5_output_RoBERTa_Metrics.png"
plt.savefig(os.path.join(base_path, output_text_png), bbox_inches='tight')
plt.close()


# Save model
output_dir = os.path.join(base_path, "RoBERTa_paragraph_classifier")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"--> Model salvat in {output_dir}")


# Vizualizări (Salvare PNG direct)
# Confusion Matrix
cm = confusion_matrix(true, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("RoBERTa — Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")

save_cm = os.path.join(base_path, "5_output_RoBERTa_Binary_CM.png")
plt.savefig(save_cm, bbox_inches='tight')
plt.close()
print(f"--> Grafic CM salvat la: {save_cm}")

# Histogram of predictions
plt.figure(figsize=(6,4))
sns.countplot(x=preds, palette="viridis")
plt.xticks([0,1], label_encoder.classes_)
plt.title("RoBERTa — Prediction Distribution")
plt.xlabel("Predicted class")
plt.ylabel("Count")

save_hist = os.path.join(base_path, "5_output_RoBERTa_Binary_Hist.png")
plt.savefig(save_hist, bbox_inches='tight')
plt.close()
print(f"--> Grafic Hist salvat la: {save_hist}")

print("\nSCRIPT 5 FINALIZAT CU SUCCES!")