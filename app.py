import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Configuration de la page
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

# Lecture des fichiers (tous avec une virgule comme séparateur)
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
# Fusion des 4 fichiers
# ---------------------
traffic_agg = df_traffic.groupby(['sku', 'date']).agg({
    'visits': 'sum',
    'unique_visitors': 'sum',
    'avg_time_on_page': 'mean',
    'bounce_rate': 'mean'
}).reset_index()

df_combined = pd.merge(
    df_sales,
    traffic_agg,
    left_on=['product_sku', 'order_date'],
    right_on=['sku', 'date'],
    how='left'
).drop(columns=['sku','date'])

# Ajout des features marketing
df_combined['est_en_promo'] = False
df_combined['promo_avec_media'] = False

for _, event in df_events.iterrows():
    event_mask = (
        (df_combined['product_sku'] == event['sku']) &
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
sku = st.sidebar.selectbox("Choisir un produit (SKU)", df_combined["product_sku"].unique())
periods = st.sidebar.slider("Nombre de mois à prédire", 3, 12, 6)
model_choice = st.sidebar.radio("Choisir un modèle", ["Prophet", "ARIMA", "RandomForest"])

# Filtrer le produit sélectionné
df_sku = df_combined[df_combined["product_sku"] == sku]
df_monthly = df_sku.groupby(pd.Grouper(key="order_date", freq="M"))["qty_ordered"].sum().reset_index()

st.title("📊 Prédiction des ventes Magento enrichie")
st.subheader(f"Produit sélectionné : {sku}")
st.info(f"Modèle choisi : **{model_choice}**. Les prévisions utilisent les ventes historiques, le trafic et les événements marketing.")

# ---------------------
# Fonction d'analyse des pics
# ---------------------
def analyze_forecast(forecast_values, historical_max):
    messages = []
    if historical_max > 0 and forecast_values.max() > historical_max * 1.5:
        messages.append("⚠️ Le modèle prévoit un pic très supérieur aux ventes historiques.")
    if forecast_values.min() < 1 and forecast_values.min() >= 0:
        messages.append("ℹ️ Certaines prévisions sont très faibles (proches de zéro).")
    return messages

# ---------------------
# MODELE 1 : PROPHET
# ---------------------
if model_choice == "Prophet":
    df_prophet = df_monthly.rename(columns={"order_date": "ds", "qty_ordered": "y"})
    
    promo_dates = df_sku[df_sku['est_en_promo']]['order_date'].unique()
    media_dates = df_sku[df_sku['promo_avec_media']]['order_date'].unique()
    
    promotions = pd.DataFrame({'holiday': 'promotion', 'ds': pd.to_datetime(promo_dates)})
    media_plans = pd.DataFrame({'holiday': 'plan_media', 'ds': pd.to_datetime(media_dates)})
    holidays_df = pd.concat((promotions, media_plans))

    model = Prophet(yearly_seasonality=True, holidays=holidays_df)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_title(f"Prévisions des ventes pour {sku} (Prophet)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantité vendue")
    st.pyplot(fig)

    st.markdown("### Lecture du graphique")
    st.write("- **Points noirs** : ventes historiques\n- **Ligne bleue** : prévisions\n- **Zone bleue claire** : intervalle de confiance")

    messages = analyze_forecast(forecast["yhat"], df_prophet["y"].max())
    for msg in messages:
        st.warning(msg)

    st.subheader("Prévisions détaillées")
    st.dataframe(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods))

# ---------------------
# MODELE 2 : ARIMA
# ---------------------
elif model_choice == "ARIMA":
    ts = df_monthly.set_index("order_date")["qty_ordered"]
    try:
        model = ARIMA(ts, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)

        plt.figure(figsize=(10,5))
        ts.plot(label="Historique")
        forecast.plot(label="Prévision", color="red")
        plt.title(f"Prévisions des ventes pour {sku} (ARIMA)")
        plt.xlabel("Date")
        plt.ylabel("Quantité vendue")
        plt.legend()
        st.pyplot(plt.gcf())

        st.markdown("### Lecture du graphique")
        st.write("- **Bleu** : ventes historiques\n- **Rouge** : prévisions ARIMA")

        messages = analyze_forecast(forecast, ts.max())
        for msg in messages:
            st.warning(msg)

        st.subheader("Prévisions détaillées")
        st.dataframe(pd.DataFrame({
            "date": pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(), periods=periods, freq="M"),
            "forecast": forecast.values
        }))
    except Exception as e:
        st.error(f"ARIMA n'a pas pu être entraîné. Cela arrive avec peu de données. Erreur : {e}")

# ---------------------
# MODELE 3 : RANDOM FOREST
# ---------------------
elif model_choice == "RandomForest":
    features = ['month','year','weekday','visits','unique_visitors','avg_time_on_page','bounce_rate', 'est_en_promo', 'promo_avec_media']
    df_rf = df_sku.copy()
    df_rf = df_rf.fillna(0)
    
    X = df_rf[features]
    y = df_rf['qty_ordered']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_date = df_rf["order_date"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="M")[1:]
    
    future_df = pd.DataFrame({"month": future_dates.month, "year": future_dates.year, "weekday": future_dates.weekday}, index=future_dates)

    for col in ['visits','unique_visitors','avg_time_on_page','bounce_rate']:
        future_df[col] = df_rf[col].mean()
        
    future_df['est_en_promo'] = False
    future_df['promo_avec_media'] = False

    forecast = model.predict(future_df[features])

    plt.figure(figsize=(10,5))
    plt.plot(df_rf["order_date"], y, label="Historique", marker='.')
    plt.plot(future_dates, forecast, label="Prévision", color="green", marker='o')
    plt.title(f"Prévisions des ventes pour {sku} (Random Forest)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.markdown("### Lecture du graphique")
    st.write("- **Bleu** : ventes historiques\n- **Vert** : prévisions Random Forest")

    messages = analyze_forecast(forecast, df_rf["qty_ordered"].max())
    for msg in messages:
        st.warning(msg)

    st.subheader("Prévisions détaillées")
    st.dataframe(pd.DataFrame({"date": future_dates, "forecast": forecast}))
