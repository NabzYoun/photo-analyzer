import streamlit as st
import json
import pandas as pd
import sys
import os

# --- FONCTION DE CHARGEMENT ---
def load_data(source):
    try:
        if isinstance(source, str): # Si c'est un chemin de fichier (string)
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        else: # Si c'est un objet uploadé via Streamlit
            return json.load(source)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du JSON : {e}")
        return None

# Configuration de la page
st.set_page_config(page_title="Analyseur Photo Dynamique", layout="wide")

# --- GESTION DU PARAMÈTRE DE LANCEMENT ---
# Streamlit permet de passer des arguments après '--' 
# Exemple: streamlit run app.py -- analysis_5.json
json_data = None
file_path_arg = sys.argv[1] if len(sys.argv) > 1 else None

if file_path_arg and os.path.exists(file_path_arg):
    json_data = load_data(file_path_arg)
    st.sidebar.success(f"Fichier chargé : {file_path_arg}")
else:
    # Si pas d'argument, on propose l'upload manuel
    st.sidebar.header("Source des données")
    uploaded_file = st.sidebar.file_uploader("Glissez votre JSON ici", type=['json'])
    if uploaded_file:
        json_data = load_data(uploaded_file)

# --- AFFICHAGE SI DONNÉES DISPONIBLES ---
if json_data:
    data = json_data # On utilise 'data' pour la suite du script

    # --- HEADER ---
    st.title("📸 Rapport d'Analyse Photo")
    st.subheader(f"Analyse de : {data.get('path', 'Fichier inconnu')}")
    
    if 'caption' in data:
        st.info(f"**Description :** {data['caption']}")

    # --- METRIQUES PRINCIPALES ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score Qualité", f"{data.get('quality_score', 'N/A')}/100")
    
    comp_score = data.get('composition', {}).get('composition_score', 'N/A')
    col2.metric("Score Composition", f"{comp_score}/100")
    
    # Verdict de composition
    elig = data.get('composition_eligibility', {})
    v_color = elig.get('color', 'grey')
    st.markdown(f"**Verdict :** <span style='color:{v_color}; font-size:24px;'>{elig.get('verdict', 'Inconnu')}</span>", unsafe_allow_html=True)
    
    col4.metric("Netteté", f"{round(data.get('sharpness', 0), 2)}")

    # --- TABS POUR LES DÉTAILS ---
    tab1, tab2, tab3 = st.tabs(["📊 Composition & Conseils", "🎨 Style & Couleur", "⚙️ EXIF & Détection"])

    with tab1:
        st.header("Analyse de la Composition")
        if 'composition_rules' in data:
            rules = data['composition_rules'].get('rules', {})
            cols = st.columns(3)
            for i, (name, details) in enumerate(rules.items()):
                with cols[i % 3]:
                    status = "✅" if details.get('respected') else "❌"
                    st.write(f"**{status} {name}**")
                    st.caption(details.get('message', ''))
        
        if elig.get('priority_fixes'):
            st.warning("### 💡 Conseils d'amélioration")
            for fix in elig['priority_fixes']:
                st.write(f"- {fix}")

    with tab2:
        # Couleurs
        st.header("Palette de Couleurs")
        if 'dominant_colors' in data:
            cols_c = st.columns(len(data['dominant_colors']))
            for i, c_info in enumerate(data['dominant_colors']):
                rgb = c_info['color']
                hex_c = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
                cols_c[i].markdown(f"<div style='background:{hex_c}; height:60px; border-radius:5px;'></div>", unsafe_allow_html=True)
                cols_c[i].code(hex_c)

        # Style (V2)
        if 'v2' in data:
            st.divider()
            style = data['v2']['style_affinities']['all_styles'][0]
            st.header(f"Style suggéré : {style['label']}")
            st.write(style['description'])
            
            c_pro, c_avoid = st.columns(2)
            with c_pro:
                st.success("**À faire (Pro Tips)**")
                for tip in style['pro_tips']: st.write(f"✔️ {tip}")
            with c_avoid:
                st.error("**À éviter**")
                for m in style['common_mistakes']: st.write(f"⚠️ {m}")
            
            st.info(f"**Réglages Lightroom :** {style['lightroom_preset']}")

    with tab3:
        c_exif, c_obj = st.columns(2)
        with c_exif:
            st.header("Données EXIF")
            st.json(data.get('exif', {}))
        with c_obj:
            st.header("Sujets détectés")
            for sub in data.get('subjects', []):
                st.write(f"🎯 {sub['class']} ({int(sub['confidence']*100)}%)")

else:
    st.warning("Veuillez fournir un fichier JSON via la ligne de commande ou l'importateur ci-contre.")
    st.code("Exemple: streamlit run votre_script.py -- mon_analyse.json")