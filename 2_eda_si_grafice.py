# 2_eda_si_grafice.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 1. CONFIGURARE & ÎNCĂRCARE pt VS Code
base_path = os.path.dirname(os.path.abspath(__file__))

# Citim fișierul rezultat din pasul 1
input_filename = "1_output_all_articles_paragraph_dataset_with_sections.csv"
full_path = os.path.join(base_path, input_filename)

if not os.path.exists(full_path):
    print(f"EROARE: Nu găsesc {input_filename}. Rulează 1_creare_dataset.py mai întâi.")
    exit()

print(f"--> Încărcăm datele din: {input_filename}")
final_df = pd.read_csv(full_path)



# GRAFIC 1: Distribuția paragrafelor pe secțiuni
print("Generăm Graficul 1...")
plt.figure(figsize=(12, 8)) # Dimensiune optimizată

ax = sns.countplot(
    y="section",
    data=final_df,
    order=final_df["section"].value_counts().index,
    palette="plasma"
)
plt.title("Distribuția paragrafelor pe secțiuni")
plt.xlabel("Număr de paragrafe")
plt.ylabel("Secțiune")

# Afișare valori
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge", padding=3)

plt.tight_layout()

# Salvare cu numele cerut
file_name_1 = "2_output_Distributia_paragrafelor_pe_sectiuni.png"
plt.savefig(os.path.join(base_path, file_name_1), dpi=300, bbox_inches='tight')
plt.close()
print(f" [Salvat] {file_name_1}")



# TABEL: Statistici descriptive (Salvat ca POZĂ .png)
print("Generăm Tabelul de Statistici...")

# Pregătim datele
stats = final_df[["n_sentences","n_tokens","n_chars"]].describe().round(2)

# Creăm o figură goală pentru tabel
plt.figure(figsize=(6, 4))
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

# Desenăm tabelul
table = plt.table(
    cellText=stats.values,
    rowLabels=stats.index,
    colLabels=stats.columns,
    cellLoc='center',
    loc='center'
)
table.scale(1.2, 1.5)
plt.title("Sumar Statistici Descriptive", y=1.05)

# Salvare
file_name_2 = "2_output_eda_summary.png"
plt.savefig(os.path.join(base_path, file_name_2), dpi=300, bbox_inches='tight')
plt.close()
print(f" [Salvat] {file_name_2}")



# GRAFIC 2: Distribuția numărului de tokens
print("Generăm Graficul 2...")
plt.figure(figsize=(12, 7))

# Limităm la 99% pentru a evita eventuale distorsiuni asupra graficului
limit_99 = np.percentile(final_df["n_tokens"], 99)

n, bins, patches = plt.hist(
    final_df["n_tokens"], 
    bins=60, 
    range=(0, limit_99),
    edgecolor="black",
    color="#4a90e2"
)

plt.title("Distribuția numărului de tokens (Zoom 99%)")
plt.xlabel("Număr de tokens")
plt.ylabel("Frecvență")

# Afișare valori (rotite, ca să nu se suprapună)
for i in range(len(n)):
    if n[i] > 10: 
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

# Salvare 
file_name_3 = "2_output_Distributia_numarului_de_tokens.png"
plt.savefig(os.path.join(base_path, file_name_3), dpi=300, bbox_inches='tight')
plt.close()
print(f" [Salvat] {file_name_3}")

print("-" * 50)
print(" Cele 3 imagini au fost generate.")