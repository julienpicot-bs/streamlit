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
    """Charge les 4 fichiers CSV depuis GitHub."""
    try:
        sales_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_fake_24months.csv'
        traffic_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/magento_traffic_24months.csv'
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        df_sales = pd.read_csv(sales_url)
        df_traffic = pd.read_csv(traffic_url)
        df_catalog = pd.read_csv(catalog_url)
        df_events = pd.read_csv(events_url)
        
        # Sécurité : nettoyer les noms de colonnes pour enlever les espaces
        df_catalog.columns = df_catalog.columns.str.strip()
        
        return df_sales, df_traffic, df_catalog, df_events
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis GitHub : {e}")
        return None, None, None, None

def clean_data(df_sales, df_traffic):
    """Nettoie et fusionne les données de ventes et de trafic."""
    # Conversion des colonnes de date
    df_sales['order_date'] = pd.to_datetime(df_sales['order_date'])
    df_traffic['date'] = pd.to_datetime(df_traffic['date'])

    # Nettoyage des données de trafic AVEC LES BONS NOMS DE COLONNES (Majuscules)
    df_traffic['Sessions'] = df_traffic['Sessions'].str.replace(',', '').astype(int)
    df_traffic['Pageviews'] = df_traffic['Pageviews'].str.replace(',', '').astype(int)
    df_traffic['Users'] = df_traffic['Users'].str.replace(',', '').astype(int)
    
    # Fusion des dataframes de ventes et de trafic
    df = pd.merge(df_sales, df_traffic, left_on='order_date', right_on='date', how='left')
    return df

# --- Chargement et préparation des données ---
df_sales, df_traffic, df_catalog, df_events = load_data()

if df_sales is not None:
    # Nettoyage initial des données de ventes et trafic
    df_cleaned = clean_data(df_sales, df_traffic)
    
    # --- NOUVELLE PARTIE : PRÉPARATION DES FEATURES POUR PROPHET ---
    
    # S'assurer que les colonnes de date du catalogue et des événements sont au bon format
    df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'])
    df_events['date_debut'] = pd.to_datetime(df_events['date_debut'])
    df_events['date_fin'] = pd.to_datetime(df_events['date_fin'])

    # Fusion avec le catalogue pour obtenir les infos produits
    df_featured = pd.merge(df_cleaned, df_catalog, on='product_sku', how='left')
    
    # Initialisation des colonnes de features
    df_featured['est_en_promo'] = False
    df_featured['promo_avec_media'] = False

    # Marquer les ventes qui ont eu lieu pendant un événement
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

    # Agréger les données par jour pour Prophet
    daily_df = df_featured.groupby('order_date').agg(
        revenue=('row_total', 'sum'),
        is_promo_day=('est_en_promo', 'max'),
        is_media_day=('promo_avec_media', 'max')
    ).reset_index()

    # Renommer les colonnes pour Prophet
    daily_df = daily_df.rename(columns={'order_date': 'ds', 'revenue': 'y'})
    
    # Créer le dataframe "holidays" que Prophet comprend
    promos = pd.DataFrame({
        'holiday': 'promotion',
        'ds': daily_df[daily_df['is_promo_day'] == True]['ds'],
    })
    media_plans = pd.DataFrame({
        'holiday': 'plan_media',
        'ds': daily_df[daily_df['is_media_day'] == True]['ds'],
    })
    holidays_df = pd.concat((promos, media_plans))
    # --- FIN DE LA NOUVELLE PARTIE ---


    # --- Barre latérale pour les contrôles utilisateur (INCHANGÉ) ---
    st.sidebar.header("Paramètres de la prédiction")
    months_to_predict = st.sidebar.slider(
        "Nombre de mois à prédire :", 
        min_value=1, max_value=24, value=6, step=1
    )
    periods_to_predict = months_to_predict * 30

    # --- Modélisation et Prédiction avec Prophet ---
    if st.sidebar.button("Lancer la prédiction"):
        with st.spinner("Entraînement du modèle et génération des prédictions..."):
            
            # MODIFICATION : On passe le dataframe holidays_df au modèle
            model = Prophet(holidays=holidays_df, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            
            # On entraîne le modèle sur les données journalières préparées
            model.fit(daily_df[['ds', 'y']])
            
            # Le reste est INCHANGÉ
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
                label="Télécharger les prédictions en CSV",
                data=csv,
                file_name=f'predictions_revenue_{months_to_predict}_mois.csv',
                mime='text/csv',
            )
else:
    st.warning("Impossible de charger les données. Veuillez vérifier les liens GitHub ou votre connexion internet.")

