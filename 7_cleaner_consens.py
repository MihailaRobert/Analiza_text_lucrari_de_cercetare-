# 7_cleaner_consens.py

import pandas as pd
import os

# --- CONFIGURARE ---
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "6_output_roberta_results.csv")
output_file = os.path.join(base_path, "7_output_dataset_clean.csv")

print("\n" + "="*50)
print(" Pasul 7: CURĂȚAREA PRIN CONSENS (Weak Learner Consensus)")
print("="*50)

if not os.path.exists(input_file):
    print(f"EROARE: Nu găsesc fișierul '{input_file}'. Ai rulat pasul 6?")
    exit()

# 1. Încărcare date
df = pd.read_csv(input_file)
initial_rows = len(df)
print(f"--> S-au încărcat {initial_rows} paragrafe inițiale.")

# 2. Aplicare filtru (Consensul)
# Păstrăm DOAR rândurile unde BERT și RoBERTa au prezis aceeași etichetă
df_clean = df[(df['bert_pred'] == df['roberta_pred']) & (df['bert_pred'] == df['label'])].copy()

final_rows = len(df_clean)
removed_rows = initial_rows - final_rows
elimination_rate = (removed_rows / initial_rows) * 100

print(f"--> S-au identificat {removed_rows} paragrafe ambigue.")
print(f"--> Rata de eliminare: {elimination_rate:.2f}%")

# 3. Salvare date curate
df_clean.to_csv(output_file, index=False)

print("\n" + "="*50)
print(f" Noul set de date CURAT are {final_rows} paragrafe perfecte.")
print(f" A fost salvat în: {output_file}")
print(" Urmează pasul final: 8_arhitectura_finala.py")
print("="*50)