# 📈 Prédiction des ventes Magento

Application **Streamlit** permettant de prédire les ventes futures à partir d’exports Magento.  
Tu peux sélectionner un produit (SKU), choisir un nombre de mois à prévoir, et comparer plusieurs modèles de prédiction.

---

## ✨ Fonctionnalités
- Chargement d’un fichier d’export Magento (exemple fourni : `magento_fake_24months.csv`)
- Sélection d’un **SKU produit**
- Choix de la période de prévision (**3 à 12 mois**)
- 3 modèles de prédiction disponibles :
  - **Prophet** (Meta) → adapté aux séries temporelles
  - **ARIMA** (Statsmodels) → modèle statistique classique
  - **Random Forest** (Sklearn) → basé sur des variables explicatives
- Visualisation :
  - Courbe des ventes historiques + prévisions
  - Tableau des prévisions (valeurs numériques)

---

## 🚀 Installation locale

1. Clone le repo ou télécharge le `.zip`
2. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
3. Lance l’application :
  streamlit run app.py
4. Ouvre http://localhost:8501
 dans ton navigateur