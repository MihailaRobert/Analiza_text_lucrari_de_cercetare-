import pandas as pd
import chardet
import os
import re
import requests
from bs4 import BeautifulSoup
import unicodedata

# CONFIGURARE
base_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_path, "raw_dataset.csv")

# 1. Încărcare CSV
with open(filename, "rb") as f:
    enc = chardet.detect(f.read(50000))['encoding']
df = pd.read_csv(filename, encoding=enc)
df.columns = [c.strip() for c in df.columns]

articles = df["paperName"].dropna().unique()

def is_complete(text):
    return bool(re.search(r'[.!?]$', text.strip()))

def count_tokens(text):
    return len(text.split())

# 3. Procesare
all_rows = []

for paper in articles:
    print(f"Procesăm: {paper}")
    df_paper = df[df["paperName"] == paper]
    if df_paper.empty: continue

    # Propozițiile "curate" din CSV
    raw_sentences = df_paper["sentence"].dropna().astype(str).unique()
    sentence_to_section = dict(zip(df_paper["sentence"], df_paper["section"]))

    arxiv_id = paper.replace(".txt", "")
    html_url = f"https://arxiv.org/html/{arxiv_id}"

    try:
        response = requests.get(html_url, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8' # Prevenim caracterele ciudate la citire
    except:
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    
    # IDENTIFICARE <p> din HTML
    html_paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(" ", strip=True)]

    merged = []
    buffer = ""
    for para in html_paragraphs:
        if not buffer: buffer = para
        else: buffer += " " + para
        if is_complete(buffer):
            merged.append(buffer.strip())
            buffer = ""
    if buffer: merged.append(buffer.strip())

    
    # FILTRARE: Păstrăm doar ce este în CSV
    for para in merged:
        matched_sentences_in_order = []
        matched_sections = []

        # 1. Normalizăm paragraful (rezolvă etÂ al. și spații speciale)
        para_norm = unicodedata.normalize("NFKC", para)
        
        # Creăm o versiune ultra-curată doar pentru CĂUTARE (fără semne de punctuație/spații)
        para_searchable = re.sub(r'[^a-zA-Z0-9]', '', para_norm)

        for s in raw_sentences:
            s_str = str(s)
            # 2. Normalizăm și propoziția din CSV
            s_norm = unicodedata.normalize("NFKC", s_str)
            # Creăm versiunea curată pentru propoziția din CSV
            s_searchable = re.sub(r'[^a-zA-Z0-9]', '', s_norm)
            
            # 3. Verificăm dacă conținutul există în paragraf
            # Folosim s_searchable în para_searchable pentru a ignora diferențele de spații/puncte
            if s_searchable and s_searchable in para_searchable:
                matched_sentences_in_order.append(s_str)
                sec = sentence_to_section.get(s)
                if pd.notna(sec): 
                    matched_sections.append(sec)

        if not matched_sentences_in_order:
            continue

        # Alegem secțiunea cea mai frecventă
        section = max(set(matched_sections), key=matched_sections.count)
        
        # Reconstruim paragraful cu textul curat din CSV
        clean_paragraph = " ".join(matched_sentences_in_order)

        all_rows.append({
            "paperName": paper,
            "section": section,
            "paragraph_text": clean_paragraph, 
            "n_sentences": len(matched_sentences_in_order),
            "n_chars": len(clean_paragraph),
            "n_tokens": count_tokens(clean_paragraph)
        })

# 4. Salvare (folosim utf-8-sig pentru Excel)
final_df = pd.DataFrame(all_rows)
output_file = os.path.join(base_path, "1_output_all_articles_paragraph_dataset_with_sections.csv")
final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Gata! Fișier salvat: {output_file}")