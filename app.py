import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------
# Charger les données
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("magento_fake_24months.csv")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df

df = load_data()

# ---------------------
# Sidebar
# ---------------------
st.sidebar.header("Paramètres")
sku = st.sidebar.selectbox("Choisir un produit (SKU)", df["product_sku"].unique())
periods = st.sidebar.slider("Nombre de mois à prédire", 3, 12, 6)
model_choice = st.sidebar.radio("Choisir un modèle", ["Prophet", "ARIMA", "RandomForest"])

# ---------------------
# Préparation des données
# ---------------------
df_sku = df[df["product_sku"] == sku]
df_monthly = df_sku.groupby(pd.Grouper(key="order_date", freq="M"))["qty_ordered"].sum().reset_index()

st.title("📈 Prédiction des ventes multi-modèles")
st.subheader(f"Produit sélectionné : {sku}")

# ---------------------
# MODELE 1 : PROPHET
# ---------------------
if model_choice == "Prophet":
    df_prophet = df_monthly.rename(columns={"order_date": "ds", "qty_ordered": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)
    
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))

# ---------------------
# MODELE 2 : ARIMA
# ---------------------
elif model_choice == "ARIMA":
    ts = df_monthly.set_index("order_date")["qty_ordered"]
    model = ARIMA(ts, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    
    fig, ax = plt.subplots()
    ts.plot(ax=ax, label="Historique")
    forecast.plot(ax=ax, label="Prévision", color="red")
    plt.legend()
    st.pyplot(fig)
    
    st.dataframe(pd.DataFrame({"date": pd.date_range(ts.index[-1]+pd.offsets.MonthEnd(), periods=periods, freq="M"),
                               "forecast": forecast.values}))

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
    
    # Prévision future
    last_date = df_rf["order_date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(), periods=periods, freq="M")
    future_df = pd.DataFrame({"month": future_dates.month, "year": future_dates.year}, index=future_dates)
    forecast = model.predict(future_df[["month","year"]])
    
    fig, ax = plt.subplots()
    ax.plot(df_rf["order_date"], y, label="Historique")
    ax.plot(future_dates, forecast, label="Prévision", color="red")
    plt.legend()
    st.pyplot(fig)
    
    st.dataframe(pd.DataFrame({"date": future_dates, "forecast": forecast}))
