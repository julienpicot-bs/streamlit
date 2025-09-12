import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------
# Chargement des données
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("magento_fake_24months.csv")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df

df = load_data()

# ---------------------
# Sidebar : paramètres
# ---------------------
st.sidebar.header("Paramètres de prédiction")
sku = st.sidebar.selectbox("Choisir un produit (SKU)", df["product_sku"].unique())
periods = st.sidebar.slider("Nombre de mois à prédire", 3, 12, 6)
model_choice = st.sidebar.radio("Choisir un modèle", ["Prophet", "ARIMA", "RandomForest"])

# ---------------------
# Filtrer le produit
# ---------------------
df_sku = df[df["product_sku"] == sku]
df_monthly = df_sku.groupby(pd.Grouper(key="order_date", freq="M"))["qty_ordered"].sum().reset_index()

st.title("📊 Prédiction des ventes Magento")
st.subheader(f"Produit sélectionné : {sku}")
st.info(f"Modèle choisi : **{model_choice}**. Les prévisions sont basées sur l'historique des 24 derniers mois.")

# ---------------------
# Fonction pour analyser pics et anomalies
# ---------------------
def analyze_forecast(forecast_values, historical_max):
    messages = []
    if forecast_values.max() > historical_max * 1.2:
        messages.append("⚠️ Attention : le modèle prévoit un pic supérieur aux ventes historiques.")
    if forecast_values.min() < 1:
        messages.append("ℹ️ Certaines prévisions sont très faibles, possiblement période creuse.")
    return messages

# ---------------------
# MODELE 1 : PROPHET
# ---------------------
if model_choice == "Prophet":
    df_prophet = df_monthly.rename(columns={"order_date": "ds", "qty_ordered": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    
    # Graphique avec seaborn
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df_prophet, x="ds", y="y", label="Historique")
    sns.lineplot(data=forecast, x="ds", y="yhat", label="Prévision", color="orange")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='orange', alpha=0.2)
    plt.title(f"Prévisions des ventes pour {sku} (Prophet)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Explications
    st.markdown("### Comment lire ce graphique")
    st.write("""
    - **Courbe bleue** : ventes réelles historiques
    - **Courbe orange** : prévisions pour les prochains mois
    - **Zone orange clair** : intervalle de confiance
    """)
    
    # Analyse pics et anomalies
    messages = analyze_forecast(forecast["yhat"], df_prophet["y"].max())
    for msg in messages:
        st.warning(msg)
    
    st.subheader("Prévisions détaillées")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))

# ---------------------
# MODELE 2 : ARIMA
# ---------------------
elif model_choice == "ARIMA":
    ts = df_monthly.set_index("order_date")["qty_ordered"]
    model = ARIMA(ts, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    
    # Graphique
    plt.figure(figsize=(10,5))
    ts.plot(label="Historique")
    forecast.plot(label="Prévision", color="red")
    plt.title(f"Prévisions des ventes pour {sku} (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.markdown("### Comment lire ce graphique")
    st.write("""
    - **Courbe bleue** : ventes historiques
    - **Courbe rouge** : prévisions ARIMA
    - Ce modèle ne fournit pas d'intervalle de confiance
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
    df_rf = df_monthly.copy()
    df_rf["month"] = df_rf["order_date"].dt.month
    df_rf["year"] = df_rf["order_date"].dt.year
    
    X = df_rf[["month", "year"]]
    y = df_rf["qty_ordered"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    last_date = df_rf["order_date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(), periods=periods, freq="M")
    future_df = pd.DataFrame({"month": future_dates.month, "year": future_dates.year}, index=future_dates)
    forecast = model.predict(future_df[["month","year"]])
    
    # Graphique
    plt.figure(figsize=(10,5))
    plt.plot(df_rf["order_date"], y, label="Historique")
    plt.plot(future_dates, forecast, label="Prévision", color="red")
    plt.title(f"Prévisions des ventes pour {sku} (Random Forest)")
    plt.xlabel("Date")
    plt.ylabel("Quantité vendue")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.markdown("### Comment lire ce graphique")
    st.write("""
    - **Courbe bleue** : ventes historiques
    - **Courbe rouge** : prévisions Random Forest
    - Le modèle utilise le mois et l'année comme variables explicatives
    - Sensible aux tendances simples mais moins aux promotions ou anomalies
    """)
    
    messages = analyze_forecast(forecast, df_rf["qty_ordered"].max())
    for msg in messages:
        st.warning(msg)
    
    st.subheader("Prévisions détaillées")
    st.dataframe(pd.DataFrame({"date": future_dates, "forecast": forecast}))
