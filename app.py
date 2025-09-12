import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

# ---------------------
# Chargement des fichiers via l'utilisateur
# ---------------------
st.sidebar.header("Charger vos fichiers CSV")
sales_file = st.sidebar.file_uploader("1. Fichier ventes Magento", type=["csv"])
traffic_file = st.sidebar.file_uploader("2. Fichier trafic", type=["csv"])
catalog_file = st.sidebar.file_uploader("3. Fichier catalogue produits", type=["csv"])
events_file = st.sidebar.file_uploader("4. Fichier événements (promos)", type=["csv"])

if not all([sales_file, traffic_file, catalog_file, events_file]):
    st.warning("Veuillez charger les 4 fichiers CSV pour continuer.")
    st.stop()

# Lecture des fichiers en respectant les séparateurs
df_sales = pd.read_csv(sales_file)
df_traffic = pd.read_csv(traffic_file)
df_catalog = pd.read_csv(catalog_file, sep=';')
df_events = pd.read_csv(events_file, sep=';')

# Conversion des dates
df_sales["order_date"] = pd.to_datetime(df_sales["order_date"], dayfirst=True)
df_traffic["date"] = pd.to_datetime(df_traffic["date"], dayfirst=True)
df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'], dayfirst=True)
df_events['date_debut'] = pd.to_datetime(df_events['date_debut'], dayfirst=True)
df_events['date_fin'] = pd.to_datetime(df_events['date_fin'], dayfirst=True)


# ---------------------
# Fusion des 4 fichiers
# ---------------------
# 1. Agréger le trafic
traffic_agg = df_traffic.groupby(['sku', 'date']).agg({
    'visits': 'sum',
    'unique_visitors': 'sum',
    'avg_time_on_page': 'mean',
    'bounce_rate': 'mean'
}).reset_index()

# 2. Fusionner ventes et trafic
df_merged = pd.merge(
    df_sales,
    traffic_agg,
    left_on=['product_sku', 'order_date'],
    right_on=['sku', 'date'],
    how='left'
).drop(columns=['sku','date'])

# 3. Ajouter les features marketing (promo, media)
df_merged['est_en_promo'] = False
df_merged['promo_avec_media'] = False

for _, event in df_events.iterrows():
    event_mask = (
        (df_merged['product_sku'] == event['sku']) &
        (df_merged['order_date'] >= event['date_debut']) &
        (df_merged['order_date'] <= event['date_fin'])
    )
    if event['type_evenement'] in ['PROMOTION', 'SOLDES']:
        df_merged.loc[event_mask, 'est_en_promo'] = True
    elif event['type_evenement'] == 'PLAN_MEDIA':
        df_merged.loc[event_mask, 'promo_avec_media'] = True

# 4. Ajouter les features temporelles
df_merged['month'] = df_merged['order_date'].dt.month
df_merged['year'] = df_merged['order_date'].dt.year
df_merged['weekday'] = df_merged['order_date'].dt.weekday


# ---------------------
# Sidebar : paramètres
# ---------------------
st.sidebar.header("Paramètres de prédiction")
sku = st.sidebar.selectbox("Choisir un produit (SKU)", df_merged["product_sku"].unique())
periods = st.sidebar.slider("Nombre de mois à prédire", 3, 12, 6)
model_choice = st.sidebar.radio("Choisir un modèle", ["Prophet", "ARIMA", "RandomForest"])

# Filtrer le produit sélectionné
df_sku = df_merged[df_merged["product_sku"] == sku]
df_monthly = df_sku.groupby(pd.Grouper(key="order_date", freq="M"))["qty_ordered"].sum().reset_index()

st.title("📊 Prédiction des ventes Magento enrichie")
st.subheader(f"Produit sélectionné : {sku}")
st.info(f"Modèle choisi : **{model_choice}**. Les prévisions utilisent les ventes historiques, les données de trafic et les événements marketing.")

# ---------------------
# Fonction d'analyse des pics
# ---------------------
def analyze_forecast(forecast_values, historical_max):
    messages = []
    if forecast_values.max() > historical_max * 1.2:
        messages.append("⚠️ Le modèle prévoit un pic supérieur aux ventes historiques.")
    if forecast_values.min() < 1:
        messages.append("ℹ️ Certaines prévisions sont très faibles, période creuse possible.")
    return messages

# ---------------------
# MODELE 1 : PROPHET
# ---------------------
if model_choice == "Prophet":
    df_prophet = df_monthly.rename(columns={"order_date": "ds", "qty_ordered": "y"})
    
    # Créer les "holidays" pour Prophet à partir des événements
    promo_dates = df_sku[df_sku['est_en_promo']]['order_date'].unique()
    media_dates =
