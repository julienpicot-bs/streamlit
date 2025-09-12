import streamlit as st
import pandas as pd
# On n'importe pas Prophet pour l'instant pour simplifier le débogage
# from prophet import Prophet
# from prophet.plot import plot_plotly, plot_components_plotly
# import plotly.graph_objs as go

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Débogage des Données",
    page_icon="🐛",
    layout="wide"
)

st.title("🐛 Outil de Débogage des Fichiers CSV")

@st.cache_data
def load_data():
    """Charge les 3 fichiers CSV depuis GitHub."""
    try:
        sales_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_fake_24months.csv'
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        df_sales = pd.read_csv(sales_url)
        df_catalog = pd.read_csv(catalog_url)
        df_events = pd.read_csv(events_url)
        
        return df_sales, df_catalog, df_events
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis GitHub : {e}")
        return None, None, None

# --- Chargement et VÉRIFICATION des données ---
df_sales, df_catalog, df_events = load_data()

if df_catalog is not None:
    st.header("Analyse du fichier `catalogue_produits.csv`")
    
    # AFFICHE LES 5 PREMIÈRES LIGNES DU FICHIER TEL QU'IL EST LU
    st.write("Voici un aperçu des données lues :")
    st.dataframe(df_catalog.head())
    
    # AFFICHE LA LISTE EXACTE DES COLONNES
    st.info("Voici la liste exacte des colonnes trouvées dans `catalogue_produits.csv` :")
    st.write(df_catalog.columns.tolist())
    
    st.warning("L'application est arrêtée ici pour le débogage. Comparez la liste ci-dessus avec le nom de colonne attendu ('date_lancement').")
    
    # Arrête le script pour voir le résultat sans provoquer l'erreur
    st.stop()

# Le reste du code ne sera pas exécuté pour l'instant
# ...
