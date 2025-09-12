def prepare_features_for_prophet(df_ventes_nettoyees):
    """
    Prend le dataframe des ventes, charge les données de catalogue/événements,
    et retourne les données prêtes pour Prophet avec un dataframe d'événements.
    """
    try:
        catalog_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/catalogue_produits.csv'
        events_url = 'https://raw.githubusercontent.com/julienpicot-bs/streamlit/main/evenements.csv'
        
        df_catalog = pd.read_csv(catalog_url)
        df_events = pd.read_csv(events_url)
        
        # S'assurer que les colonnes de date sont au bon format
        df_catalog['date_lancement'] = pd.to_datetime(df_catalog['date_lancement'])
        df_events['date_debut'] = pd.to_datetime(df_events['date_debut'])
        df_events['date_fin'] = pd.to_datetime(df_events['date_fin'])

        # Créer les flags de promo/media au niveau de la vente individuelle
        df_featured = pd.merge(df_ventes_nettoyees, df_catalog, on='product_sku', how='left')
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

        # Agréger par jour
        daily_df = df_featured.groupby('order_date').agg(
            revenue=('row_total', 'sum'),
            is_promo_day=('est_en_promo', 'max'),
            is_media_day=('promo_avec_media', 'max')
        ).reset_index()

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
        
        return daily_df, holidays_df
        
    except Exception as e:
        st.error(f"Erreur lors de la préparation des features : {e}")
        return None, None
