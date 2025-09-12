import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Diagnostic CSV Final", layout="wide")
st.title("🐛 Diagnostic Final des Fichiers CSV")

@st.cache_data
def load_debug_data():
    """Charge les fichiers catalogue et événements pour le diagnostic."""
    try:
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        import requests
        
        response_catalog = requests.get(catalog_url)
        response_catalog.raise_for_status()
        catalog_text = response_catalog.text

        response_events = requests.get(events_url)
        response_events.raise_for_status()
        events_text = response_events.text

        return catalog_text, events_text
    except Exception as e:
        st.error(f"Erreur critique lors du téléchargement d'un fichier depuis GitHub : {e}")
        return None, None

# --- Chargement et Diagnostic ---
catalog_text_data, events_text_data = load_debug_data()

if catalog_text_data:
    st.header("Analyse du fichier `catalogue_produits.csv`")
    st.subheader("Tentative de lecture avec un point-virgule (`;`)")
    try {
        df_catalog = pd.read_csv(io.StringIO(catalog_text_data), sep=';')
        st.write("Aperçu :")
        st.dataframe(df_catalog.head())
        st.info("Colonnes trouvées :")
        st.code(df_catalog.columns.tolist())
    } catch (Exception e) {
        st.error(f"La lecture a échoué : {e}")
    }

st.markdown("---")

if events_text_data:
    st.header("Analyse du fichier `evenements.csv`")
    st.subheader("Tentative de lecture avec un point-virgule (`;`)")
    try {
        df_events = pd.read_csv(io.StringIO(events_text_data), sep=';')
        st.write("Aperçu :")
        st.dataframe(df_events.head())
        st.info("Colonnes trouvées :")
        st.code(df_events.columns.tolist())
    } catch (Exception e) {
        st.error(f"La lecture a échoué : {e}")
    }

