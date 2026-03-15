import streamlit as st
import pandas as pd
import os
import re
from gtts import gTTS

# Configurare Vizuală
st.set_page_config(page_title="Agent AI Multimedia", page_icon="", layout="wide")

st.title("Agent Multimedia")

cale_rezumat = "10_output_rezumat_si_traducere_ro.txt"

# Inițializare stare pentru indexul imaginii și Zoom
if 'img_index' not in st.session_state:
    st.session_state.img_index = 1
if 'zoom_active' not in st.session_state:
    st.session_state.zoom_active = False

def extrage_date_complete(cale_fisier):
    if not os.path.exists(cale_fisier):
        return None
    with open(cale_fisier, "r", encoding="utf-8") as f:
        continut = f.read()
    
    blocuri = re.split(r'(EXEMPLUL \d+)', continut)
    date_finale = []
    
    for i in range(1, len(blocuri), 2):
        titlu = blocuri[i].strip() 
        corp = blocuri[i+1]
        
        # Extracție cod ArXiv (ex: 2503.16271v1)
        cod_match = re.search(r'(\d{4,5}\.\d{4,5}(?:v\d+)?)', corp)
        cod_arxiv = cod_match.group(1) if cod_match else ""
        link_complet = f"https://arxiv.org/html/{cod_arxiv}" if cod_arxiv else "https://arxiv.org/html/"
            
        rez_match = re.search(r'REZUMAT ROMÂNĂ:\s*(.*?)(?=\n-{10,}|$)', corp, re.DOTALL)
        rez_ro = rez_match.group(1).strip() if rez_match else ""
        text_audio = rez_ro.replace(">>", "").replace(">", "").strip()
        
        date_finale.append({
            "id": titlu,
            "link_sursa": link_complet,
            "cod": cod_arxiv,
            "continut_complet": corp.strip(),
            "text_citibil": text_audio
        })
    return date_finale

date_agent = extrage_date_complete(cale_rezumat)

if date_agent:
    st.sidebar.header("Navigare")
    
    # Resetăm indexul și zoom-ul la schimbarea exemplului
    def reset_index():
        st.session_state.img_index = 1
        st.session_state.zoom_active = False

    selectie = st.sidebar.selectbox(
        "Alege exemplul:", 
        [d['id'] for d in date_agent], 
        on_change=reset_index
    )
    
    date_sel = next(x for x in date_agent if x['id'] == selectie)
    
    # URL-ul imaginii curent
    url_img = f"{date_sel['link_sursa']}/x{st.session_state.img_index}.png"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🔍 Vizualizare {date_sel['id']}")
        st.write(f"**SURSA:** {date_sel['link_sursa']}")
        
        st.subheader("DATELE DOCUMENTULUI:")
        st.text_area("Continut:", date_sel['continut_complet'], height=800, label_visibility="collapsed")
        
        if st.button("Redare Audio (Rezumat RO)", key="btn_audio"):
            st.balloons()
            with st.spinner("Se generează vocea..."):
                tts = gTTS(text=date_sel['text_citibil'], lang='ro')
                tts.save("audio_agent.mp3")
                st.audio("audio_agent.mp3", autoplay=True)

    with col2:
        st.header("Galerie Desene ArXiv")
        
        if date_sel['cod']:
            # Rând butoane Navigare + Zoom
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            
            with c1:
                if st.button("⬅️", key="btn_prev"):
                    if st.session_state.img_index > 1:
                        st.session_state.img_index -= 1
            
            with c2:
                # Buton de ZOOM (comută starea)
                zoom_label = "🔍 Zoom Out" if st.session_state.zoom_active else "🔍 Zoom In"
                if st.button(zoom_label, key="btn_zoom"):
                    st.session_state.zoom_active = not st.session_state.zoom_active
            
            with c3:
                st.write(f"Fig. **{st.session_state.img_index}**")
            
            with c4:
                if st.button("➡️", key="btn_next"):
                    st.session_state.img_index += 1
            
            # Afișare Imagine normală sau Zoom
            if st.session_state.zoom_active:
                st.warning("Mod Zoom activat. Imaginea este afișată la lățime maximă mai jos.")
            
            st.image(url_img, 
                     caption=f"Figura {st.session_state.img_index} - {date_sel['cod']}", 
                     use_container_width=True)
            
            st.markdown(f"[Link Direct ArXiv]({date_sel['link_sursa']})")
        else:
            st.warning("Cod ArXiv nedetectat pentru acest exemplu.")

        st.divider()
        st.info("Utilizați 🔍 pentru a mări imaginea sau săgețile pentru navigare.")

    # Secțiune specială de Zoom 
    if st.session_state.zoom_active:
        st.divider()
        st.subheader("Vedere Extinsă (Zoom)")
        st.image(url_img, use_container_width=False) # False păstrează rezoluția originală sau pun True pentru full screen