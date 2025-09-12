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

# Utilisation du cache pour ne charger les données qu'une seule fois
@st.cache_data
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
        
        return df_sales, df_catalog, df_events
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis GitHub : {e}")
        return None, None, None

@st.cache_data
def create_features(df_sales, df_catalog, df_events):
    """Fusionne les données et crée les nouvelles features marketing."""
    # 1. Conversion des colonnes de dates
    df_sales['order_date'] = pd.to_datetime(df_sales['order_date'])
    df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'])
    df_events['date_debut'] = pd.to_datetime(df_events['date_debut'])
    df_events['date_fin'] = pd.to_datetime(df_events['date_fin'])

    # 2. Fusion du catalogue avec les ventes pour obtenir la date de lancement
    df_merged = pd.merge(df_sales, df_catalog, on='product_sku', how='left')

    # 3. Création des features au niveau de chaque ligne de vente
    df_merged['est_nouveau'] = (
        (df_merged['order_date'] - df_merged['date_lancement']).dt.days.between(0, 90)
    )

    # Initialisation des colonnes à False
    df_merged['est_en_promo'] = False
    df_merged['promo_avec_media'] = False

    # 4. Itération sur les événements pour marquer les ventes correspondantes
    # C'est plus performant que d'appliquer une fonction ligne par ligne
    for _, event in df_events.iterrows():
        # Masque pour trouver les ventes concernées par l'événement
        event_mask = (
            (df_merged['product_sku'] == event['sku']) &
            (df_merged['order_date'] >= event['date_debut']) &
            (df_merged['order_date'] <= event['date_fin'])
        )
        
        if event['type_evenement'] in ['PROMOTION', 'SOLDES']:
            df_merged.loc[event_mask, 'est_en_promo'] = True
        elif event['type_evenement'] == 'PLAN_MEDIA':
            # On vérifie aussi que la promo a lieu en même temps
            promo_mask = df_merged['est_en_promo']
            df_merged.loc[event_mask & promo_mask, 'promo_avec_media'] = True
            
    return df_merged

# --- Chargement et préparation des données ---
df_sales, df_catalog, df_events = load_data()

if df_sales is not None:
    # Création des features et agrégation par jour
    df_featured = create_features(df_sales, df_catalog, df_events)
    
    daily_df = df_featured.groupby('order_date').agg(
        revenue=('row_total', 'sum'),
        is_promo_day=('est_en_promo', 'max'), # Vrai si au moins un produit en promo ce jour-là
        is_media_day=('promo_avec_media', 'max') # Vrai si au moins un produit en plan média
    ).reset_index()

    # Renommage des colonnes pour Prophet
    daily_df = daily_df.rename(columns={'order_date': 'ds', 'revenue': 'y'})

    # --- Préparation des événements pour Prophet ---
    
    # Création d'un dataframe 'holidays' pour Prophet
    promos = pd.DataFrame({
        'holiday': 'promotion',
        'ds': daily_df[daily_df['is_promo_day'] == True]['ds'],
        'lower_window': 0,
        'upper_window': 0, # L'effet est uniquement le jour J
    })

    media_plans = pd.DataFrame({
        'holiday': 'plan_media',
        'ds': daily_df[daily_df['is_media_day'] == True]['ds'],
        'lower_window': 0,
        'upper_window': 0,
    })

    holidays_df = pd.concat((promos, media_plans))

    # --- Barre latérale pour les contrôles utilisateur ---
    st.sidebar.header("Paramètres de la prédiction")
    months_to_predict = st.sidebar.slider(
        "Nombre de mois à prédire :", 
        min_value=1, max_value=24, value=6, step=1
    )
    periods_to_predict = months_to_predict * 30

    # --- Modélisation et Prédiction avec Prophet ---
    if st.sidebar.button("Lancer la prédiction"):
        with st.spinner("Entraînement du modèle et génération des prédictions..."):
            
            # Initialisation du modèle avec les 'holidays'
            model = Prophet(holidays=holidays_df, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            
            # Entraînement
            model.fit(daily_df[['ds', 'y']])
            
            # Création du dataframe futur
            future = model.make_future_dataframe(periods=periods_to_predict)
            
            # Prédiction
            forecast = model.predict(future)

            # --- Affichage des résultats ---
            st.subheader(f"Prédiction du CA pour les {months_to_predict} prochains mois")
            
            # Graphique de prédiction
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(
                title="Prédiction du Chiffre d'Affaires vs Réalité",
                xaxis_title="Date",
                yaxis_title="Chiffre d'Affaires (€)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Graphiques des composantes
            st.subheader("Analyse des tendances et saisonnalités")
            fig_components = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_components, use_container_width=True)

            # Affichage des données de prédiction
            st.subheader("Détail des données prédites")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_predict))

            # Option pour télécharger les résultats
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les prédictions en CSV",
                data=csv,
                file_name=f'predictions_revenue_{months_to_predict}_mois.csv',
                mime='text/csv',
            )
else:
    st.warning("Impossible de charger les données. Veuillez vérifier les liens GitHub ou votre connexion internet.")
