import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------
# Chargement des fichiers via l'utilisateur
# ---------------------
st.sidebar.header("Charger vos fichiers CSV")
sales_file = st.sidebar.file_uploader("Fichier ventes Magento", type=["csv"])
traffic_file = st.sidebar.file_uploader("Fichier trafic", type=["csv"])

if sales_file is None or traffic_file is None:
    st.warning("Veuillez charger à la fois le fichier de ventes et le fichier de trafic.")
    st.stop()

# Lecture des fichiers
df_sales = pd.read_csv(sales_file)
df_sales["order_date"] = pd.to_datetime(df_sales["order_date"])

df_traffic = pd.read_csv(traffic_file)
df_traffic["date"] = pd.to_datetime(df_traffic["date"])

# ---------------------
# Fusion ventes + trafic
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
st.info(f"Modèle choisi : **{model_choice}**. Les prévisions utilisent les ventes historiques et les données de trafic.")

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
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    # Graphique
    plt.figure(figsize=(10,5))
    plt.plot(df_prophet["ds"], df_prophet["y"], label="Historique")
    plt.plot(forecast["ds"], forecast["yhat"], label="Prévision", color="orange")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='orange', alpha=0.2)
    plt.title(f"Prévisions des ventes pour {sku} (Prophet)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.legend()
    st.pyplot(plt.gcf())

    # Explications
    st.markdown("### Lecture du graphique")
    st.write("""
    - **Bleu** : ventes historiques
    - **Orange** : prévisions
    - **Zone orange clair** : intervalle de confiance
    """)

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
    st.write("""
    - **Bleu** : ventes historiques
    - **Rouge** : prévisions ARIMA
    """)

    messages = analyze_forecast(forecast, ts.max())
    for msg in messages:
        st.warning(msg)

    st.subheader("Prévisions détaillées")
    st.dataframe(pd.DataFrame({
        "date": pd.date_range(ts.index[-1]+pd.offsets.MonthEnd(), periods=periods, freq="M"),
        "forecast": forecast.values
    }))

# ---------------------
# MODELE 3 : RANDOM FOREST
# ---------------------
elif model_choice == "RandomForest":
    # Features pour RandomForest
    features = ['month','year','weekday','visits','unique_visitors','avg_time_on_page','bounce_rate']
    df_rf = df_sku.copy()
    df_rf = df_rf.fillna(0)  # remplacer les NaN par 0 pour les jours sans trafic
    X = df_rf[features]
    y = df_rf['qty_ordered']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prévision future
    last_date = df_rf["order_date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(), periods=periods, freq="M")
    future_df = pd.DataFrame({
        "month": future_dates.month,
        "year": future_dates.year,
        "weekday": future_dates.weekday
    }, index=future_dates)

    # Pour le trafic futur, utiliser la moyenne historique
    for col in ['visits','unique_visitors','avg_time_on_page','bounce_rate']:
        future_df[col] = df_rf[col].mean()

    forecast = model.predict(future_df[features])

    plt.figure(figsize=(10,5))
    plt.plot(df_rf["order_date"], y, label="Historique")
    plt.plot(future_dates, forecast, label="Prévision", color="red")
    plt.title(f"Prévisions des ventes pour {sku} (Random Forest avec trafic)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown("### Lecture du graphique")
    st.write("""
    - **Bleu** : ventes historiques
    - **Rouge** : prévisions Random Forest
    - Les features trafic sont utilisées pour améliorer la prédiction
    """)

    messages = analyze_forecast(forecast, df_rf["qty_ordered"].max())
    for msg in messages:
        st.warning(msg)

    st.subheader("Prévisions détaillées")
    st.dataframe(pd.DataFrame({"date": future_dates, "forecast": forecast}))
