# 3_embeddings.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# 1. CONFIGURARE CALE & FOLDERE
base_path = os.path.dirname(os.path.abspath(__file__))

# Input: Fișierul de la pasul 1
input_filename = "1_output_all_articles_paragraph_dataset_with_sections.csv"
full_path = os.path.join(base_path, input_filename)

# Output: Foldere și fișiere
# Folderul 
HEATMAP_DIR = os.path.join(base_path, "3_output_Heatmaps")
if not os.path.exists(HEATMAP_DIR):
    os.makedirs(HEATMAP_DIR)

EMBED_SAVE = os.path.join(base_path, "3_output_paragraph_embeddings.npy")
META_SAVE = os.path.join(base_path, "3_output_meta.csv")
SCOR_GLOBAL_FILE = os.path.join(base_path, "3_output_Scor_global.txt")

# Configurare Model
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("EROARE: Trebuie instalat sentence-transformers (pip install sentence-transformers)")
    exit()

# Verificare existență fișier input
if not os.path.exists(full_path):
    print(f"EROARE: Nu găsesc '{input_filename}'.")
    print("Rulează  '1_creare_dataset.py'.")
    exit()

# Încărcare date
print(f"--> [1/4] Încărcăm datele din: {input_filename}")
df_par = pd.read_csv(full_path)
# Resetăm indexul pentru siguranță
df_par = df_par.reset_index(drop=True)



# 2. GENERARE EMBEDDINGS (SAU ÎNCĂRCARE)
embeddings = None

# Verificăm dacă există deja calculate
if os.path.exists(EMBED_SAVE) and os.path.exists(META_SAVE):
    try:
        print("--> [INFO] Am găsit embeddings existente. Le încărcăm...")
        embeddings = np.load(EMBED_SAVE, allow_pickle=True)
        # Reîncărcăm și metadatele ca să fim siguri că se potrivesc
        meta = pd.read_csv(META_SAVE)
    except:
        embeddings = None

if embeddings is None:
    print("--> [2/4] Generăm embeddings noi (Model: all-MiniLM-L6-v2)...")
    texts = df_par["paragraph_text"].astype(str).tolist()
    
    # Pregătre metadate
    meta = df_par[["paperName", "section"]].copy()
    meta["orig_index"] = np.arange(len(meta))

    # Inițializare model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generare (batching automat)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    # Salvare
    np.save(EMBED_SAVE, embeddings)
    meta.to_csv(META_SAVE, index=False)
    print(f"   [Salvat] Embeddings în {EMBED_SAVE}")



# 3. CALCUL SIMILARITATE & GENERARE HEATMAPS
print(f"--> [3/4] Generăm Heatmaps în folderul '{HEATMAP_DIR}'...")

papers = meta["paperName"].unique()
scores_per_paper = []
SIM_THRESHOLD = 0.70

for paper in papers:
    # Selectare date pentru articolul curent
    mask = (meta["paperName"] == paper)
    if not mask.any(): continue

    # Extragere embeddings
    paper_embeddings = embeddings[mask]
    paper_sections = meta.loc[mask, "section"].tolist()

    # Identificare indicii pentru Intro și Concluzie
    intro_idxs = [i for i, x in enumerate(paper_sections) if x == "Introduction"]
    concl_idxs = [i for i, x in enumerate(paper_sections) if x == "Conclusion"]

    # Dacă lipsește una dintre secțiuni, sărim peste
    if not intro_idxs or not concl_idxs:
        continue

    # Extragem vectorii
    intro_emb = paper_embeddings[intro_idxs]
    concl_emb = paper_embeddings[concl_idxs]

    # Calculăm similaritatea 
    sim_matrix = cosine_similarity(intro_emb, concl_emb)

    # Calculăm scorul mediu maxim (cât de bine se potrivește fiecare paragraf din intro cu cel mai bun din concluzie)
    max_sim = sim_matrix.max(axis=1)
    mean_score = float(np.mean(max_sim))
    scores_per_paper.append(mean_score)

    # --- GENERARE HEATMAP (SALVARE FĂRĂ AFIȘARE) ---
    plt.figure(figsize=(8, 5))
    sns.heatmap(sim_matrix, annot=False, cmap="viridis", cbar=True)
    
    plt.title(f"Similaritate: {paper}")
    plt.xlabel("Paragrafe Concluzie")
    plt.ylabel("Paragrafe Introducere")
    
    # Nume fișier curat
    clean_name = paper.replace(".txt", "").replace("/", "_")
    save_path = os.path.join(HEATMAP_DIR, f"Heatmap_{clean_name}.png")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close() # <--- Închide graficul pentru a nu bloca memoria



# 4. CALCUL SCOR GLOBAL & FINALIZARE
print("--> [4/4] Calculăm Scor Global...")

if len(scores_per_paper) > 0:
    global_score = np.mean(scores_per_paper)
    
    # 1. Salvare în fișier text (simplu de citit)
    with open(SCOR_GLOBAL_FILE, "w", encoding="utf-8") as f:
        f.write(f"SCOR GLOBAL DE SIMILARITATE (Intro vs Concluzie)\n")
        f.write(f"===============================================\n")
        f.write(f"Valoare: {global_score:.4f}\n")
        f.write(f"Nr. Articole analizate: {len(scores_per_paper)}\n")
    
    print("\n" + "="*50)
    print(f" Rezultate generate.")
    print(f" 1. Heatmaps:  Vezi folderul '3_output_Heatmaps'")
    print(f" 2. Scor Global: {global_score:.4f} (Salvat în '3_output_Scor_global.txt')")
    print("="*50)

else:
    print("\n[AVERTISMENT] Nu s-au putut calcula scoruri (lipsesc secțiunile Intro/Concluzie).")