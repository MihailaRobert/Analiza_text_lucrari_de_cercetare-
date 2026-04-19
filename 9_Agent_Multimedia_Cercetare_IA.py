import streamlit as st
import pandas as pd
import os, torch, urllib.parse, requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from bs4 import BeautifulSoup
import streamlit.components.v1 as components


# 1. PREGĂTIREA (MODELE IA ȘI CONFIGURARE)
# Setăm aspectul paginii să fie lat pentru a vedea totul clar
st.set_page_config(page_title="Agent IA Cercetare", layout="wide", page_icon="")

@st.cache_resource
def incarca_modelele_ia():
    """
    Această funcție încarcă 'creierul' aplicației (Modelele IA).
    Folosim @st.cache_resource pentru a nu le reîncărca la fiecare click (economisim timp).
    """
    dispozitiv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modelul pentru Rezumat (T5-Small: mic și rapid)
    tok_rezumat = AutoTokenizer.from_pretrained("t5-small", legacy=False)
    mod_rezumat = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(dispozitiv)
    
    # Modelul pentru Traducere (English -> Romanian)
    tok_traducere = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ro")
    mod_traducere = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ro").to(dispozitiv)
    
    return tok_rezumat, mod_rezumat, tok_traducere, mod_traducere, dispozitiv


# 2. FUNCȚII DE LOGICĂ (CUM PROCESĂM DATELE)
@st.cache_data(show_spinner=False)
def extrage_date_de_pe_internet(id_arxiv):
    """
    Merge pe site-ul ArXiv și adună: Titlu, Autori și Imagini.
    """
    url_articol = f"https://arxiv.org/html/{id_arxiv}"
    date_gasite = {"titlu": id_arxiv, "autori": "N/A", "imagini": [], "html": ""}
    
    try:
        raspuns = requests.get(url_articol, timeout=5)
        if raspuns.status_code == 200:
            browser_virtual = BeautifulSoup(raspuns.text, 'html.parser')
            
            # Căutăm titlul și autorii în codul paginii
            titlu = browser_virtual.find('h1', class_='ltx_title') or browser_virtual.find('title')
            autori = browser_virtual.find('div', class_='ltx_authors')
            
            if titlu: date_gasite["titlu"] = titlu.get_text().strip()
            if autori: date_gasite["autori"] = autori.get_text().strip().replace('\n', ' ')
            
            # Căutăm imaginile și eliminăm iconițele mici care nu ne interesează
            for imagine in browser_virtual.find_all('img'):
                sursa = imagine.get('src', '')
                alt_text = imagine.get('alt', '').lower()
                cuvinte_de_ignorat = ["logo", "button", "accessibility", "icon", "wheelchair"]
                
                if any(cuvant in sursa.lower() or cuvant in alt_text for cuvant in cuvinte_de_ignorat):
                    continue
                
                if sursa:
                    # Transformăm calea relativă într-o adresă web completă
                    if sursa.startswith('http'):
                        url_complet = sursa
                    elif sursa.startswith('/'):
                        url_complet = f"https://arxiv.org{sursa}"
                    else:
                        url_complet = f"{url_articol}/{sursa}"
                    date_gasite["imagini"].append(url_complet)
            
            date_gasite["html"] = browser_virtual.prettify()
    except:
        pass
    return date_gasite

def executa_analiza_ia(text_sursa, t_rez, m_rez, t_tra, m_tra, dev):
    """
    Primește un text și returnează un rezumat tradus în Română.
    """
    # Pasul A: Facem rezumatul în Engleză
    intrari = t_rez("summarize: " + text_sursa, return_tensors="pt", truncation=True).to(dev)
    iesiri = m_rez.generate(intrari.input_ids, max_length=150, min_length=40, num_beams=5)
    rezumat_en = t_rez.decode(iesiri[0], skip_special_tokens=True)
    
    # Pasul B: Traducem rezumatul în Română
    in_tra = t_tra(rezumat_en, return_tensors="pt", padding=True).to(dev)
    out_tra = m_tra.generate(**in_tra)
    rezumat_ro = t_tra.decode(out_tra[0], skip_special_tokens=True)
    
    return rezumat_en, rezumat_ro


# 3. CONSTRUIREA INTERFEȚEI (CE VEDE UTILIZATORUL)
CALE_CSV = "1_output_all_articles_paragraph_dataset_with_sections.csv"

if os.path.exists(CALE_CSV):
    # Încărcăm tabelul cu paragrafe
    date_tabel = pd.read_csv(CALE_CSV)
    date_tabel.columns = [coloana.strip() for coloana in date_tabel.columns]

    # --- SIDEBAR (MENIUL LATERAL) ---
    st.sidebar.title("📚 Control Bibliotecă")
    
    # [FUNCȚIA 10]: Status Bibliotecă
    nr_total = date_tabel['paperName'].nunique()
    st.sidebar.success(f"**10. Sincronizat:** {nr_total} documente")

    # [FUNCȚIA 1]: Căutare
    cautare_text = st.sidebar.text_input("1. Căutare articol:", "")
    lista_filtrata = [t for t in sorted(date_tabel['paperName'].unique()) if cautare_text.lower() in t.lower()]
    articol_selectat = st.sidebar.selectbox("Selectează:", lista_filtrata)

    if articol_selectat:
        # GESTIUNEA MEMORIEI (Session State)
        # Verificăm dacă am schimbat articolul pentru a reseta imaginile și zoom-ul
        if st.session_state.get('articol_activ') != articol_selectat:
            st.session_state.articol_activ = articol_selectat
            st.session_state.img_idx = 0
            st.session_state.zoom = 1.0
            st.session_state.ultima_analiza = "" 

        date_specifice = date_tabel[date_tabel['paperName'] == articol_selectat]
        
        # [FUNCȚIA 6]: Status Document
        st.sidebar.info(f"Paragrafe disponibile: {len(date_specifice)}")
        if len(date_specifice) > 50: 
            st.sidebar.warning("6. Atenție: Document foarte lung")

        # [FUNCȚIA 3]: Selector Paragraf
        paragraf_ales = st.sidebar.selectbox("3. Paragraf:", date_specifice['paragraph_text'].unique(), format_func=lambda x: str(x)[:50] + "...")
        
        # Extragem datele de pe ArXiv (Scraping)
        id_arxiv = articol_selectat.replace(".txt", "").strip()
        # CORECȚIE: Folosim numele corect al funcției redefinite mai sus
        info_web = extrage_date_de_pe_internet(id_arxiv)
        sectiune_nume = str(date_specifice[date_specifice['paragraph_text'] == paragraf_ales]['section'].iloc[0])

        # --- AFIȘARE ANTET (ZOTERO 4 & 2) ---
        st.title(f"{info_web['titlu']}")
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            # [FUNCȚIA 4]: Metadate
            anul = "20" + id_arxiv[:2] if id_arxiv[0].isdigit() else "N/A"
            st.markdown(f"**4. Metadate:** ID: `{id_arxiv}` | An: `{anul}`")
        with col_meta2:
            # [FUNCȚIA 2]: Etichete
            culoare_tag = "#1f77b4" if "Intro" in sectiune_nume else "#2ca02c"
            st.markdown(f'**2. Etichetă:** <span style="background-color:{culoare_tag}; color:white; padding:2px 8px; border-radius:5px;">{sectiune_nume}</span>', unsafe_allow_html=True)
        
        # [FUNCȚIA 8]: Citație IEEE
        st.write("**8. Referință Bibliografică (IEEE):**")
        st.code(f"{info_web['autori']}, \"{info_web['titlu']}\", ArXiv: {id_arxiv}, {anul}.", language="text")
        
        st.divider()

        # --- CELE 3 COLOANE PRINCIPALE ---
        c1, c2, c3 = st.columns([1.5, 1.2, 0.7])

        # COLOANA 1: CITITORUL (FUNCȚIA 5)
        with c1:
            st.subheader("5. Vizualizare Sursă")
            text_codificat = urllib.parse.quote(str(paragraf_ales)[:100])
            url_iframe = f"https://arxiv.org/html/{id_arxiv}#:~:text={text_codificat}"
            st.markdown(f'<iframe src="{url_iframe}" width="100%" height="800px" style="border:1px solid orange; border-radius:10px;"></iframe>', unsafe_allow_html=True)

        # COLOANA 2: ANALIZA IA (FUNCȚIILE 7 & 9)
        with c2:
            st.subheader("Analiză IA")
            
            # Verificăm dacă trebuie să rulăm IA (doar dacă s-a schimbat paragraful)
            if st.session_state.get('ultima_analiza') != paragraf_ales:
                with st.spinner("IA lucrează la rezumat..."):
                    t_r, m_r, t_t, m_t, d = incarca_modelele_ia()
                    en, ro = executa_analiza_ia(str(paragraf_ales), t_r, m_r, t_t, m_t, d)
                    st.session_state.txt_en, st.session_state.txt_ro = en, ro
                    st.session_state.ultima_analiza = paragraf_ales

            st.info(f"**Text original:** {paragraf_ales}")
            st.success(f"**🇺🇸 EN:** {st.session_state.txt_en}")
            st.success(f"**🇷🇴 RO:** {st.session_state.txt_ro}")

            # [FUNCȚIA 9]: Audio
            if st.button("9. Redare Audio"):
                st.balloons()
                gTTS(text=st.session_state.txt_ro, lang='ro').save("voce.mp3")
                st.session_state.audio_activ = True
                st.rerun()
            if st.session_state.get('audio_activ'):
                st.audio("voce.mp3", autoplay=True)
                st.session_state.audio_activ = False

            # [FUNCȚIA 7]: Export
            st.download_button("7. Exportă Raport", str(paragraf_ales), f"Raport_{id_arxiv}.txt")

        # COLOANA 3: GALERIE MULTIMEDIA
        with c3:
            st.subheader("Figuri")
            poze = info_web['imagini']
            if poze:
                index_curent = st.session_state.img_idx
                if index_curent >= len(poze):
                    st.warning("🏁 Nu mai sunt Figuri")
                else:
                    zoom_procent = int(100 * st.session_state.zoom)
                    # HTML + JS pentru mișcarea imaginii (Pan)
                    html_vizualizator = f"""
                    <div id="ecran" style="width:100%;height:400px;overflow:hidden;border:1px solid #ddd;border-radius:10px;background:#f9f9f9;position:relative;display:flex;align-items:center;justify-content:center;cursor:grab;">
                        <img id="poza" src="{poze[index_curent]}" 
                             onload="this.style.opacity='1';"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                             style="position:absolute; width:{zoom_procent}%; opacity:0; user-select:none; pointer-events:none; left:0; top:0; transition:opacity 0.2s;">
                        <div style="display:none; color:#333; font-weight:bold; text-align:center;">🏁 Nu mai sunt Figuri</div>
                    </div>
                    <script>
                        let posX=0, posY=0;
                        const ecran=document.getElementById('ecran'), poza=document.getElementById('poza');
                        ecran.onmousedown = (e) => {{
                            ecran.style.cursor = 'grabbing';
                            let startX = e.clientX - posX;
                            let startY = e.clientY - posY;
                            window.onmousemove = (ev) => {{
                                posX = ev.clientX - startX;
                                posY = ev.clientY - startY;
                                poza.style.left = posX + 'px';
                                poza.style.top = posY + 'px';
                            }};
                        }};
                        window.onmouseup = () => {{
                            ecran.style.cursor = 'grab';
                            window.onmousemove = null;
                        }};
                    </script>
                    """
                    components.html(html_vizualizator, height=420)
                    st.caption(f"Figura {index_curent + 1} din {len(poze)}")

                    # Butoane navigare
                    b_stanga, b_dreapta = st.columns(2)
                    if b_stanga.button("⬅️"): st.session_state.img_idx = max(0, index_curent - 1); st.rerun()
                    if b_dreapta.button("➡️"): st.session_state.img_idx = index_curent + 1; st.rerun()
                    
                    # Butoane Zoom
                    z_plus, z_minus = st.columns(2)
                    if z_plus.button("➕"): st.session_state.zoom += 0.5; st.rerun()
                    if z_minus.button("➖"): st.session_state.zoom = max(0.5, st.session_state.zoom - 0.5); st.rerun()
            else:
                st.warning("Nu am găsit imagini.")
else:
    st.error("Eroare: Fișierul CSV nu a fost găsit în folder!")
    #python -m streamlit run 9_Agent_Multimedia_Cercetare_IA.py