import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diagnostic Trafic", layout="wide")
st.title("🐛 Diagnostic du Fichier `magento_traffic_24months.csv`")

try:
    # On charge uniquement le fichier qui pose problème
    traffic_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_traffic_24months.csv'
    df_traffic = pd.read_csv(traffic_url)

    st.header("Analyse du fichier")
    
    st.info("Voici la liste exacte des colonnes trouvées dans votre fichier :")
    # AFFICHE LA LISTE BRUTE DES COLONNES
    st.code(df_traffic.columns.tolist())
    
    st.write("Aperçu des 5 premières lignes du fichier :")
    st.dataframe(df_traffic.head())

except Exception as e:
    st.error(f"Une erreur est survenue lors de la lecture du fichier : {e}")
