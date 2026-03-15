# 9_rezumat_paragrafe.py

import pandas as pd
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. CONFIGURARE 
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "1_output_all_articles_paragraph_dataset_with_sections.csv")
output_file = os.path.join(base_path, "9_output_rezumate_paragrafe.txt")

print("\n" + "="*60)
print(" Pasul 9: REZUMAREA AUTOMATĂ A PARAGRAFELOR (T5-Small)")
print("="*60)

if not os.path.exists(input_file):
    print(f"EROARE: Nu găsesc '{input_file}'. Rulează pașii anteriori!")
    exit()

# 2. ÎNCĂRCARE DATE
df = pd.read_csv(input_file)
print(f"--> Am încărcat datele ({len(df)} paragrafe în total).")

# Selectare aleatorie 20 paragrafe pentru demonstrație
NR_EXEMPLE = 20
df_sample = df.sample(n=min(NR_EXEMPLE, len(df)), random_state=42).reset_index(drop=True)

# 3. INIȚIALIZARE MODEL REZUMARE 
# Verificăm dacă există GPU (NVIDIA) disponibil, altfel folosim CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Folosim hardware: {device.type.upper()}")

print("--> Încărcăm modelul Google T5-Small... ")
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 4. GENERARE REZUMATE 
print(f"\n--> Generare rezumate pentru cele {NR_EXEMPLE} exemple...")

rezultate_text = []
rezultate_text.append("=== REZUMARE AUTOMATĂ A PARAGRAFELOR ===\n")
rezultate_text.append(f"Model folosit: {model_name}\n")
rezultate_text.append(f"Hardware utilizat: {device.type.upper()}\n")
rezultate_text.append("="*60 + "\n\n")

start_time = time.time()

# Dezactivăm calculul gradienților pentru a economisi memorie și viteză
with torch.no_grad():
    for index, row in df_sample.iterrows():
        text_original = str(row['paragraph_text'])
        sectiune = row['section']
        nume_document = row['paperName'] 
        
        cuvinte = text_original.split()
        
        # Filtru pentru text prea scurt
        if len(cuvinte) < 30:
            rezumat = "[Text prea scurt pentru a fi rezumat corect.]"
        else:
            try:
                # Formatare specifică T5
                text_input = "summarize: " + text_original
                
                # Tokenizare
                inputs = tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True).to(device)
                
                # Generare optimizată
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    max_length=90,             # Lungime suficientă pentru context mic
                    min_length=25,             # Evităm rezumate de tip "telegramă"
                    length_penalty=2.5,        # Favorizăm fraze complete, nu doar fragmente
                    num_beams=5,               # Analizăm mai multe variante (calitate superioară)
                    no_repeat_ngram_size=3,    # Prevenim repetițiile de cuvinte
                    early_stopping=True
                )
                
                # Decodare și post-procesare
                rezumat = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
                
                # Capitalizare automată (pentru aspect estetic)
                if rezumat:
                    rezumat = rezumat[0].upper() + rezumat[1:]
                
            except Exception as e:
                rezumat = f"[Eroare la generare: {e}]"

        # Construim blocul de text pentru raport
        bloc_text = f"EXEMPLUL {index + 1} (Secțiunea: {sectiune} | Sursă: {nume_document})\n"
        bloc_text += f"TEXT ORIGINAL ({len(cuvinte)} cuvinte):\n{text_original}\n\n"
        bloc_text += f"REZUMAT AI:\n>> {rezumat}\n"
        bloc_text += "-"*60 + "\n\n"
        
        rezultate_text.append(bloc_text)
        print(f"   [OK] {index+1}/{NR_EXEMPLE} procesat.")

# 5. SALVARE REZULTATE 
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(rezultate_text)

timp_total = time.time() - start_time
print("\n" + "="*60)
print(f" Proces finalizat în {timp_total:.1f} secunde.")
print(f" Rezultatele se află în: {output_file}")
print("="*60)