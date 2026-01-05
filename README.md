# 🏷️ Mercari Price Predictor

Application de prédiction de prix pour produits d'occasion, basée sur le dataset Mercari (marketplace japonaise similaire à Leboncoin).

https://mercaripricepredictor-gmollot.streamlit.app/

## 📋 Description

Ce projet propose une interface simple permettant à un vendeur de :
- Sélectionner une catégorie de produit
- Indiquer l'état du produit
- Renseigner les informations de l'annonce (titre, description, marque)
- Obtenir une estimation du prix de vente optimal

## 🎯 Fonctionnalités

- **Prédiction de prix** : estimation basée sur un modèle de machine learning entraîné sur 1.4M d'annonces
- **API REST** : endpoints FastAPI pour intégration dans d'autres applications
- **Interface utilisateur** : application Streamlit intuitive

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.10+ |
| Data processing | Pandas, NumPy, Scikit-learn |
| Modèle ML | LightGBM / XGBoost |
| API | FastAPI |
| Interface | Streamlit |
| Sérialisation | Joblib |

## 📁 Structure du projet

```
mercari-price-predictor/
│
├── data/
│   ├── raw/                    # Données brutes Kaggle
│   └── processed/              # Données nettoyées
│
├── notebooks/                  # Jupyter notebooks d'exploration
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                        # Code source
│   ├── data/                   # Chargement et nettoyage
│   ├── features/               # Feature engineering
│   ├── models/                 # Entraînement et prédiction
│   └── utils/                  # Fonctions utilitaires
│
├── api/                        # API FastAPI
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│
├── app/                        # Interface Streamlit
│   └── streamlit_app.py
│
├── models/                     # Modèles sérialisés
├── tests/                      # Tests unitaires
├── configs/                    # Fichiers de configuration
└── scripts/                    # Scripts utilitaires
```

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/username/mercari-price-predictor.git
cd mercari-price-predictor
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger les données**

Télécharger le dataset depuis [Kaggle Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data) et placer `train.tsv` dans `data/raw/`.

Ou utiliser le script :
```bash
python scripts/download_data.py
```
*(Nécessite une API key Kaggle configurée)*

## 💻 Utilisation

### Entraîner le modèle

```bash
python scripts/train_model.py
```

Options disponibles :
```bash
python scripts/train_model.py --model lightgbm --cv 5
```

### Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

L'API sera accessible sur `http://localhost:8000`

Documentation Swagger : `http://localhost:8000/docs`

### Lancer l'interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'interface sera accessible sur `http://localhost:8501`

## 📡 Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Status de l'API |
| GET | `/categories` | Liste des catégories disponibles |
| POST | `/predict` | Prédiction de prix |

### Exemple de requête `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nike Air Max 90",
    "category": "Men/Shoes/Athletic",
    "brand": "Nike",
    "condition": 2,
    "description": "Worn twice, excellent condition, size 42"
  }'
```

### Exemple de réponse

```json
{
  "predicted_price": 45.99,
  "price_range": {
    "low": 38.00,
    "high": 55.00
  },
  "confidence": 0.85
}
```

## 📊 Données

### Source

Dataset Mercari Price Suggestion Challenge (Kaggle)
- 1.4 million d'annonces
- Produits variés : électronique, vêtements, maison, etc.

### Features utilisées

| Feature | Type | Description |
|---------|------|-------------|
| name | texte | Titre de l'annonce |
| category_name | catégoriel | Catégorie hiérarchique (3 niveaux) |
| brand_name | catégoriel | Marque du produit |
| item_condition_id | numérique | État du produit (1-5) |
| shipping | binaire | Frais de port inclus ou non |
| item_description | texte | Description libre |

### États du produit

| ID | Label |
|----|-------|
| 1 | Neuf |
| 2 | Comme neuf |
| 3 | Bon état |
| 4 | État correct |
| 5 | Usé |

## 🧪 Tests

Lancer les tests :
```bash
pytest tests/ -v
```

Avec couverture :
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance du modèle

| Modèle | RMSLE (validation) |
|--------|-------------------|
| Ridge Regression (baseline) | 0.46 |
| LightGBM | 0.42 |
| XGBoost | 0.41 |

*RMSLE = Root Mean Squared Logarithmic Error (plus bas = meilleur)*

## 🔧 Configuration

Les paramètres sont modifiables dans `configs/config.yaml` :

```yaml
data:
  train_path: "data/raw/train.tsv"
  test_size: 0.2
  random_state: 42

features:
  tfidf_max_features: 10000
  min_price: 1
  max_price: 2000

model:
  type: "lightgbm"
  params:
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 8
```

## 🚢 Déploiement

### Option 1 : Render (API)

1. Connecter le repo GitHub à Render
2. Configurer le build command : `pip install -r requirements.txt`
3. Configurer le start command : `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Option 2 : Streamlit Cloud (Interface)

1. Connecter le repo GitHub à Streamlit Cloud
2. Sélectionner `app/streamlit_app.py` comme fichier principal
3. Déployer

## 🤝 Contribution

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- [Mercari](https://www.mercari.com/) pour le dataset
- [Kaggle](https://www.kaggle.com/) pour l'hébergement de la compétition
- La communauté open source pour les outils utilisés

## 📬 Contact

Des questions ? Ouvrir une issue sur GitHub.

---

*Projet réalisé dans le cadre d'un portfolio de data science.*
