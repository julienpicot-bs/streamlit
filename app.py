import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Prédiction de Chiffre d'Affaires",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Outil de Prédiction de Chiffre d'Affaires")
st.write("Prédiction du chiffre d'affaires basée sur les données de ventes, de trafic et les événements marketing.")

# --- Fonctions de chargement et de traitement des données ---

@st.cache_data
def load_data():
    """Charge tous les fichiers de données depuis GitHub avec les bons séparateurs."""
    try:
        sales_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_fake_24months.csv'
        traffic_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_traffic_24months.csv'
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        df_sales = pd.read_csv(sales_url)
        df_traffic = pd.read_csv(traffic_url)
        df_catalog = pd.read_csv(catalog_url, sep=';')
        df_events = pd.read_csv(events_url, sep=';')
        
        # Sécurité : nettoyer les noms de colonnes
        df_catalog.columns = df_catalog.columns.str.strip()
        df_traffic.columns = df_traffic.columns.str.strip()
        
        return df_sales, df_traffic, df_catalog, df_events
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis GitHub : {e}")
        return None, None, None, None

def clean_and_merge_data(df_sales, df_traffic):
    """Nettoie et fusionne les données de ventes et de trafic."""
    # CORRECTION : Utilisation de format='mixed' pour gérer les dates inconsistantes
    df_sales['order_date'] = pd.to_datetime(df_sales['order_date'], format='mixed', dayfirst=True)
    df_traffic['date'] = pd.to_datetime(df_traffic['date'], format='mixed', dayfirst=True)

    # Agréger les données de trafic par jour
    daily_traffic = df_traffic.groupby('date').agg(
        visits=('visits', 'sum'),
        unique_visitors=('unique_visitors', 'sum')
    ).reset_index()
    
    # Fusion des dataframes
    df = pd.merge(df_sales, daily_traffic, left_on='order_date', right_on='date', how='left')
    return df

# --- Chargement et préparation des données ---
df_sales, df_traffic, df_catalog, df_events = load_data()

if df_sales is not None:
    df_cleaned = clean_and_merge_data(df_sales, df_traffic)
    
    # --- PRÉPARATION DES FEATURES POUR PROPHET ---
    
    # CORRECTION : Utilisation de format='mixed'
    df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'], format='mixed', dayfirst=True)
    df_events['date_debut'] = pd.to_datetime(df_events['date_debut'], format='mixed', dayfirst=True)
    df_events['date_fin'] = pd.to_datetime(df_events['date_fin'], format='mixed', dayfirst=True)

    df_featured = pd.merge(df_cleaned, df_catalog, left_on='product_sku', right_on='sku', how='left')
    
    df_featured['est_en_promo'] = False
    df_featured['promo_avec_media'] = False

    for _, event in df_events.iterrows():
        event_mask = (
            (df_featured['product_sku'] == event['sku']) &
            (df_featured['order_date'] >= event['date_debut']) &
            (df_featured['order_date'] <= event['date_fin'])
        )
        if event['type_evenement'] in ['PROMOTION', 'SOLDES']:
            df_featured.loc[event_mask, 'est_en_promo'] = True
        elif event['type_evenement'] == 'PLAN_MEDIA':
            df_featured.loc[event_mask, 'promo_avec_media'] = True

    daily_df = df_featured.groupby('order_date').agg(
        revenue=('row_total', 'sum'),
        is_promo_day=('est_en_promo', 'max'),
        is_media_day=('promo_avec_media', 'max')
    ).reset_index()

    daily_df = daily_df.rename(columns={'order_date': 'ds', 'revenue': 'y'})
    
    promos = pd.DataFrame({'holiday': 'promotion', 'ds': daily_df[daily_df['is_promo_day'] == True]['ds']})
    media_plans = pd.DataFrame({'holiday': 'plan_media', 'ds': daily_df[daily_df['is_media_day'] == True]['ds']})
    holidays_df = pd.concat((promos, media_plans))

    # --- Barre latérale ---
    st.sidebar.header("Paramètres de la prédiction")
    months_to_predict = st.sidebar.slider("Nombre de mois à prédire :", 1, 24, 6)
    periods_to_predict = months_to_predict * 30

    # --- Modélisation et Prédiction ---
    if st.sidebar.button("Lancer la prédiction"):
        with st.spinner("Entraînement du modèle..."):
            model = Prophet(holidays=holidays_df, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(daily_df[['ds', 'y']])
            future = model.make_future_dataframe(periods=periods_to_predict)
            forecast = model.predict(future)

            st.subheader(f"Prédiction du CA pour les {months_to_predict} prochains mois")
            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Analyse des tendances et saisonnalités")
            fig_components = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_components, use_container_width=True)

            st.subheader("Détail des données prédites")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_predict))

            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Télécharger les prédictions en CSV",
                csv,
                f'predictions_revenue_{months_to_predict}_mois.csv',
                'text/csv',
            )
else:
    st.warning("Impossible de charger les données.")
