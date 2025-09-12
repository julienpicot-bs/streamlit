import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Prédiction de Chiffre d'Affaires",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Outil de Prédiction de Chiffre d'Affaires")
st.write("Cette application analyse les données de ventes et les événements marketing pour prédire le chiffre d'affaires futur.")

# --- Fonctions de chargement et de traitement des données ---

# IMPORTANT : Le cache est désactivé pour forcer le rechargement des fichiers
def load_data():
    """Charge les 3 fichiers CSV depuis GitHub."""
    try:
        # Liens vers les fichiers RAW sur GitHub
        sales_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_fake_24months.csv'
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        df_sales = pd.read_csv(sales_url)
        df_catalog = pd.read_csv(catalog_url)
        df_events = pd.read_csv(events_url)
        
        # Sécurité : On nettoie les noms de colonnes pour enlever les espaces invisibles
        df_catalog.columns = df_catalog.columns.str.strip()
        
        return df_sales, df_catalog, df_events
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis GitHub : {e}")
        return None, None, None

# IMPORTANT : Le cache est désactivé
def create_features(df_sales, df_catalog, df_events):
    """Fusion
