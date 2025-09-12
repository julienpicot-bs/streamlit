import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="Diagnostic CSV",
    page_icon="ախ",
    layout="wide"
)

st.title("ախ Diagnostic Avancé du Fichier CSV")

@st.cache_data
def load_catalog_data():
    """Charge uniquement le fichier catalogue pour le diagnostic."""
    try:
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        # On utilise requests pour mieux gérer le contenu texte
        import requests
        response = requests.get(catalog_url)
        response.raise_for_status() # Lève une erreur si le téléchargement échoue
        return response.text
    except Exception as e:
        st.error(f"Erreur critique lors du téléchargement du fichier depuis GitHub : {e}")
        return None

# --- Chargement et Diagnostic ---
csv_text_data = load_catalog_data()

if csv_text_data:
    st.header("Analyse du contenu du fichier `catalogue_produits.csv`")
    
    st.subheader("Tentative de lecture avec une virgule (`,`) comme séparateur")
    try:
        df_comma = pd.read_csv(io.StringIO(csv_text_data), sep=',')
        st.write("Aperçu des données (séparateur virgule) :")
        st.dataframe(df_comma.head())
        st.info("Colonnes trouvées (séparateur virgule) :")
        st.write(df_comma.columns.tolist())
    except Exception as e:
        st.error(f"La lecture avec une virgule a échoué : {e}")

    st.markdown("---")

    st.subheader("Tentative de lecture avec un point-virgule (`;`) comme séparateur")
    try:
        df_semicolon = pd.read_csv(io.StringIO(csv_text_data), sep=';')
        st.write("Aperçu des données (séparateur point-virgule) :")
        st.dataframe(df_semicolon.head())
        st.info("Colonnes trouvées (séparateur point-virgule) :")
        st.write(df_semicolon.columns.tolist())
    except Exception as e:
        st.error(f"La lecture avec un point-virgule a échoué : {e}")
    
    st.markdown("---")
    st.warning("Regardez quelle tentative a correctement séparé les colonnes. C'est celle-là qui nous donne le bon séparateur.")

else:
    st.error("Le fichier n'a pas pu être chargé. Impossible de continuer le diagnostic.")
