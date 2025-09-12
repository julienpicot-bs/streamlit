import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# ---------------------
# Chargement des fichiers (simplifié)
# ---------------------
st.sidebar.header("Charger vos fichiers CSV")
sales_file = st.sidebar.file_uploader("1. Fichier ventes", type=["csv"])
traffic_file = st.sidebar.file_uploader("2. Fichier trafic", type=["csv"])
catalog_file = st.sidebar.file_uploader("3. Fichier catalogue", type=["csv"])
events_file = st.sidebar.file_uploader("4. Fichier événements", type=["csv"])

if not all([sales_file, traffic_file, catalog_file, events_file]):
    st.warning("Veuillez charger les 4 fichiers CSV.")
    st.stop()

# Lecture simple (plus besoin de sep=';')
df_sales = pd.read_csv(sales_file)
df_traffic = pd.read_csv(traffic_file)
df_catalog = pd.read_csv(catalog_file)
df_events = pd.read_csv(events_file)

# Conversion des dates (plus besoin de dayfirst=True)
df_sales["order_date"] = pd.to_datetime(df_sales["order_date"])
df_traffic["date"] = pd.to_datetime(df_traffic["date"])
df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'])
df_events['date_debut'] = pd.to_datetime(df_events['date_debut'])
df_events['date_fin'] = pd.to_datetime(df_events['date_fin'])


# ---------------------
# Fusion des fichiers (simplifié)
# ---------------------
traffic_agg = df_traffic.groupby(['sku', 'date']).agg({
    'visits': 'sum',
    'unique_visitors': 'sum',
    'avg_time_on_page': 'mean',
    'bounce_rate': 'mean'
}).reset_index()

# La fusion se fait maintenant sur 'sku' et 'order_date'/'date'
df_combined = pd.merge(df_sales, traffic_agg, on=['sku', 'order_date'], how='left')

# Ajout des features marketing
df_combined['est_en_promo'] = False
df_combined['promo_avec_media'] = False

for _, event in df_events.iterrows():
    event_mask = (
        (df_combined['sku'] == event['sku']) &
        (df_combined['order_date'] >= event['date_debut']) &
        (df_combined['order_date'] <= event['date_fin'])
    )
    if event['type_evenement'] in ['PROMOTION', 'SOLDES']:
        df_combined.loc[event_mask, 'est_en_promo'] = True
    elif event['type_evenement'] == 'PLAN_MEDIA':
        df_combined.loc[event_mask, 'promo_avec_media'] = True

# Ajouter features temporelles
df_combined['month'] = df_combined['order_date'].dt.month
df_combined['year'] = df_combined['order_date'].dt.year
df_combined['weekday'] = df_combined['order_date'].dt.weekday


# ---------------------
# Sidebar : paramètres
# ---------------------
st.sidebar.header("Paramètres de prédiction")
sku = st.sidebar.selectbox("Choisir un produit (SKU)", df_combined["sku"].unique())
periods = st.sidebar.slider("Nombre de mois à prédire", 3, 12, 6)
model_choice = st.sidebar.radio("Choisir un modèle", ["Prophet", "ARIMA", "RandomForest"])

# Le reste du code est identique à votre version originale et devrait fonctionner sans changement.
# ... (votre code pour les modèles Prophet, ARIMA, RandomForest) ...
