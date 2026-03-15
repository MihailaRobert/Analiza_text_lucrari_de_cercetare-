# 10_rezumat_si_traducere_ro.py

import pandas as pd
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. CONFIGURARE
base_path = os.path.dirname(os.path.abspath(__file__))
# Folosim fișierul de paragrafe generat la pasul anterior
input_file = os.path.join(base_path, "1_output_all_articles_paragraph_dataset_with_sections.csv")
output_file = os.path.join(base_path, "10_output_rezumat_si_traducere_ro.txt")

print("\n" + "="*60)
print(" PASUL 10: GENERARE REZUMATE ȘI TRADUCERE ÎN ROMÂNĂ")
print("="*60)

if not os.path.exists(input_file):
    print(f"EROARE: Nu găsesc '{input_file}'! Verifică s-a rulat scriptul .")
    exit()

# 2. ÎNCĂRCARE MODELE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Hardware detectat: {device.type.upper()}")

# Model pentru Rezumare (English)
print("--> Încărcăm modelul de rezumare (T5-Small)...")
sum_model_name = "t5-small"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name).to(device)

# Model pentru Traducere (English -> Romanian)
print("--> Încărcăm modelul de traducere (MarianMT EN-RO)...")
trans_model_name = "Helsinki-NLP/opus-mt-en-ro"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(device)

# --- 3. PROCESARE DATE ---
df = pd.read_csv(input_file)
NR_EXEMPLE = 20
# Selectăm aceleași 20 de paragrafe (folosim random_state pentru consistență)
df_sample = df.sample(n=min(NR_EXEMPLE, len(df)), random_state=42).reset_index(drop=True)

rezultate_final = []
rezultate_final.append("=== RAPORT FINAL: REZUMATE ȘI TRADUCERI ÎN LIMBA ROMÂNĂ ===\n")
rezultate_final.append(f"Data generării: {time.strftime('%d-%m-%Y %H:%M:%S')}\n")
rezultate_final.append("="*60 + "\n\n")

start_time = time.time()

# viteză și economisire memorie
with torch.no_grad():
    for index, row in df_sample.iterrows():
        text_original = str(row['paragraph_text'])
        sectiune = row['section']
        nume_doc = row['paperName']
        
        # Filtru pentru text prea scurt
        cuvinte = text_original.split()
        if len(cuvinte) < 30:
            continue
            
        # A. REZUMARE (în Engleză)
        input_sum = "summarize: " + text_original
        tokens_sum = sum_tokenizer(input_sum, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        output_sum = sum_model.generate(
            tokens_sum["input_ids"], 
            max_length=90, 
            num_beams=5, 
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        rezumat_en = sum_tokenizer.decode(output_sum[0], skip_special_tokens=True).strip()

        # B. TRADUCERE (EN -> RO)
        # Modelul de traducere preia rezumatul în engleză
        inputs_trans = trans_tokenizer(rezumat_en, return_tensors="pt", padding=True).to(device)
        output_trans = trans_model.generate(**inputs_trans)
        rezumat_ro = trans_tokenizer.decode(output_trans[0], skip_special_tokens=True).strip()

        # C. FORMATARE ȘI CURĂȚARE
        if rezumat_ro:
            rezumat_ro = rezumat_ro[0].upper() + rezumat_ro[1:] # Capitalizare

        # Construim blocul de text
        bloc = f"EXEMPLUL {index + 1}\n"
        bloc += f"SURSA: {nume_doc} | SECȚIUNE: {sectiune}\n"
        bloc += f"TEXT ORIGINAL (EN):\n{text_original}\n\n"
        bloc += f"REZUMAT ENGLEZĂ:\n>> {rezumat_en}\n\n"
        bloc += f"REZUMAT ROMÂNĂ:\n>> {rezumat_ro}\n"
        bloc += "-"*60 + "\n\n"
        
        rezultate_final.append(bloc)
        print(f"   [OK] Procesat și tradus exemplul {index+1}/{NR_EXEMPLE}")

# 4. SALVARE REZULTAT FINAL 
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(rezultate_final)

print("\n" + "="*60)
print(f" Fișierul '{os.path.basename(output_file)}' a fost generat.")
print(f" Timp total de procesare: {time.time() - start_time:.1f} secunde.")
print("="*60)