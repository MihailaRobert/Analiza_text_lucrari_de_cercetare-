# 0_eda_si_grafice_nivel_propozitii.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import chardet
import re

# --- 1. CONFIGURARE CALE ---
base_path = os.path.dirname(os.path.abspath(__file__))
input_filename = os.path.join(base_path, "raw_dataset.csv")

if not os.path.exists(input_filename):
    print(f"EROARE: Nu găsesc fișierul '{input_filename}'!")
    exit()

# --- 2. ÎNCĂRCARE ȘI DETECTARE ENCODING
print(f"--> [1/3] Încarc datele la nivel de propoziție din: raw_dataset.csv")
with open(input_filename, "rb") as f:
    rawdata = f.read(50000)
    enc = chardet.detect(rawdata)['encoding']

df = pd.read_csv(input_filename, encoding=enc)
df.columns = [c.strip() for c in df.columns]

# Curățare sumară: eliminăm rândurile goale sau fără secțiune
df = df.dropna(subset=["section", "sentence"]).copy()

# Funcții pentru calcularea numărului de caractere și cuvinte (tokens) pentru fiecare propoziție
def count_tokens(text):
    return len(str(text).split())

df["n_chars"] = df["sentence"].astype(str).apply(len)
df["n_tokens"] = df["sentence"].astype(str).apply(count_tokens)
# Numărul de propoziții e mereu 1, pentru că suntem la nivel de propoziție
df["n_sentences"] = 1 



# GRAFIC 1: Distribuția propozițiilor pe secțiuni
print("--> [2/3] Generăm Graficul 1 (Distribuția pe secțiuni)...")
plt.figure(figsize=(12, 8))

ax = sns.countplot(
    y="section",
    data=df,
    order=df["section"].value_counts().index,
    palette="viridis" # Schimbat la viridis pentru a-l diferenția de cel cu paragrafe
)
plt.title("Distribuția PROPOZIȚIILOR pe secțiuni (Date Brute)")
plt.xlabel("Număr de propoziții")
plt.ylabel("Secțiune")

# Afișare valori pe bare
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge", padding=3)

plt.tight_layout()

file_name_1 = "0_output_Propozitii_pe_sectiuni.png"
plt.savefig(os.path.join(base_path, file_name_1), dpi=300, bbox_inches='tight')
plt.close()



# TABEL: Statistici (Salvat ca POZĂ .png)
print("--> [3/3] Generăm Tabelul de Statistici și Graficul 2 (Tokens)")

stats = df[["n_sentences", "n_tokens", "n_chars"]].describe().round(2)

plt.figure(figsize=(6, 4))
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

table = plt.table(
    cellText=stats.values,
    rowLabels=stats.index,
    colLabels=["Propoziții", "Cuvinte (Tokens)", "Caractere"],
    cellLoc='center',
    loc='center'
)
table.scale(1.2, 1.5)
plt.title("Statistici Descriptive (Nivel Propoziție)", y=1.05)

file_name_2 = "0_output_eda_summary_propozitii.png"
plt.savefig(os.path.join(base_path, file_name_2), dpi=300, bbox_inches='tight')
plt.close()



# GRAFIC 2: Distribuția numărului de tokens (cuvinte per propoziție)
plt.figure(figsize=(12, 7))

# La propoziții, evitam distorsiuni asupra graficului, limităm la 99%
limit_99 = np.percentile(df["n_tokens"], 99)

n, bins, patches = plt.hist(
    df["n_tokens"], 
    bins=50, 
    range=(0, limit_99), 
    edgecolor="black",
    color="#2ca02c" # Verde pentru propoziții
)

plt.title("Distribuția lungimii propozițiilor (Zoom 99%)")
plt.xlabel("Număr de cuvinte (tokens) per propoziție")
plt.ylabel("Frecvență")

for i in range(len(n)):
    if n[i] > (max(n) * 0.05): # Afișăm doar valorile relevante, nu și cele foarte mici
        plt.text(
            (bins[i] + bins[i+1]) / 2,
            n[i] + (max(n)*0.01),
            str(int(n[i])),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

plt.tight_layout()

file_name_3 = "0_output_Distributia_tokens_propozitii.png"
plt.savefig(os.path.join(base_path, file_name_3), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 50)
print(" S-au generat 3 imagini pentru datele la nivel de propoziție:")
print(f" 1. {file_name_1}")
print(f" 2. {file_name_2}")
print(f" 3. {file_name_3}")
print("=" * 50)