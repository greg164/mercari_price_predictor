"""
Mercari Price Predictor - Streamlit App
Interface utilisateur pour prédire le prix optimal de vente
"""

import sys
from pathlib import Path

# Ajouter le path racine pour les imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional

# ================================================================================
# CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Mercari Price Predictor",
    page_icon="🏷️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Chemins
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_v1.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_v1.joblib"

# Catégories principales (basées sur le dataset Mercari)
MAIN_CATEGORIES = [
    "Women",
    "Men", 
    "Beauty",
    "Electronics",
    "Kids",
    "Home",
    "Sports & Outdoors",
    "Handmade",
    "Vintage & Collectibles",
    "Other"
]

# Sous-catégories par catégorie principale
SUB_CATEGORIES = {
    "Women": ["Tops & Blouses", "Dresses", "Jeans", "Shoes", "Bags & Purses", "Jewelry", "Accessories", "Athletic Apparel", "Coats & Jackets", "Other"],
    "Men": ["Tops", "Shirts", "Jeans", "Shoes", "Accessories", "Coats & Jackets", "Athletic Apparel", "Suits", "Other"],
    "Beauty": ["Makeup", "Skin Care", "Hair Care", "Fragrance", "Tools & Accessories", "Bath & Body", "Nails", "Other"],
    "Electronics": ["Cell Phones & Accessories", "Computers & Tablets", "Video Games & Consoles", "Cameras & Photography", "TV & Audio", "Other"],
    "Kids": ["Toys", "Girls Clothing", "Boys Clothing", "Baby Gear", "Shoes", "Other"],
    "Home": ["Kitchen & Dining", "Bedding", "Bath", "Home Décor", "Furniture", "Storage", "Other"],
    "Sports & Outdoors": ["Exercise & Fitness", "Outdoor Recreation", "Sports", "Cycling", "Other"],
    "Handmade": ["Art", "Jewelry", "Knitting", "Bags & Purses", "Other"],
    "Vintage & Collectibles": ["Antiques", "Collectibles", "Art", "Other"],
    "Other": ["Other"]
}

# États du produit
CONDITION_LABELS = {
    1: "🌟 Neuf avec étiquettes",
    2: "✨ Neuf sans étiquettes", 
    3: "👍 Très bon état",
    4: "👌 Bon état",
    5: "🔧 État correct"
}

# Marques populaires (top du dataset)
TOP_BRANDS = [
    "", "Nike", "Victoria's Secret", "LuLaRoe", "Apple", "PINK", 
    "Nintendo", "Lululemon", "Michael Kors", "American Eagle",
    "Adidas", "Coach", "Rae Dunn", "Bath & Body Works", "Samsung",
    "Sony", "Disney", "Forever 21", "Kate Spade", "Carter's"
]


# ================================================================================
# CHARGEMENT DU MODÈLE (EN CACHE)
# ================================================================================

@st.cache_resource
def load_model_and_preprocessor():
    """Charge le modèle et le preprocessor une seule fois."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor, None
    except FileNotFoundError as e:
        return None, None, f"Fichiers modèle non trouvés: {e}"
    except Exception as e:
        return None, None, f"Erreur de chargement: {e}"


# ================================================================================
# FONCTIONS DE PRÉDICTION
# ================================================================================

def prepare_input_data(
    name: str,
    category_main: str,
    category_sub1: str,
    category_sub2: str,
    brand_name: str,
    item_condition_id: int,
    shipping: int,
    item_description: str
) -> pd.DataFrame:
    """
    Prépare les données d'entrée au format attendu par le preprocessor.
    """
    # Reconstruire category_name au format original
    category_name = f"{category_main}/{category_sub1}/{category_sub2}"
    
    # Créer le DataFrame avec les colonnes attendues
    data = {
        'name': [name.lower().strip()],
        'item_condition_id': [item_condition_id],
        'category_name': [category_name],
        'brand_name': [brand_name if brand_name else "unknown"],
        'shipping': [shipping],
        'item_description': [item_description.lower().strip() if item_description else ""]
    }
    
    # Ajouter les colonnes de catégories séparées (attendues par le preprocessor)
    data['cat_main'] = [category_main]
    data['cat_sub1'] = [category_sub1]
    data['cat_sub2'] = [category_sub2]
    
    return pd.DataFrame(data)


def predict_price(
    model,
    preprocessor,
    input_data: pd.DataFrame
) -> Dict:
    """
    Prédit le prix à partir des données d'entrée.
    """
    try:
        # Transformer les features
        X = preprocessor.transform(input_data)
        
        # Prédiction (en log)
        y_log_pred = model.predict(X)[0]
        
        # Revenir au prix réel
        predicted_price = np.expm1(y_log_pred)
        
        # Calculer une fourchette (approximation basée sur le RMSLE ~0.49)
        # RMSLE de 0.49 signifie ~50% d'erreur en échelle log
        price_low = predicted_price * 0.65
        price_high = predicted_price * 1.55
        
        return {
            'predicted_price': max(1, predicted_price),
            'price_low': max(1, price_low),
            'price_high': price_high,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'predicted_price': None,
            'price_low': None,
            'price_high': None,
            'success': False,
            'error': str(e)
        }


# ================================================================================
# STYLES CSS
# ================================================================================

def load_css():
    """Charge les styles CSS personnalisés."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .main-header p {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Cards */
    .form-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .form-card h3 {
        color: #ccd6f6;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 2px solid rgba(102, 126, 234, 0.4);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .price-label {
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
        position: relative;
    }
    
    .price-main {
        font-family: 'Space Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: #64ffda;
        text-shadow: 0 0 30px rgba(100, 255, 218, 0.3);
        position: relative;
        margin: 0.5rem 0;
    }
    
    .price-range {
        color: #8892b0;
        font-size: 1rem;
        position: relative;
    }
    
    /* Inputs styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ccd6f6;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ccd6f6;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Tips */
    .tips-box {
        background: rgba(100, 255, 218, 0.05);
        border-left: 3px solid #64ffda;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .tips-box h4 {
        color: #64ffda;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .tips-box p {
        color: #8892b0;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #495670;
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Radio buttons for shipping */
    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================================
# COMPOSANTS UI
# ================================================================================

def render_header():
    """Affiche le header de l'application."""
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Mercari Price Predictor</h1>
        <p>Estimez le prix optimal de vente pour votre article</p>
    </div>
    """, unsafe_allow_html=True)


def render_result(result: Dict):
    """Affiche le résultat de la prédiction."""
    if result['success']:
        st.markdown(f"""
        <div class="result-card">
            <div class="price-label">Prix suggéré</div>
            <div class="price-main">${result['predicted_price']:.0f}</div>
            <div class="price-range">
                Fourchette : ${result['price_low']:.0f} — ${result['price_high']:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips contextuels
        st.markdown("""
        <div class="tips-box">
            <h4>💡 Conseils</h4>
            <p>• Ajoutez des photos de qualité pour augmenter vos chances de vente<br>
            • Mentionnez les défauts éventuels dans la description<br>
            • Un prix légèrement en-dessous de la fourchette haute accélère la vente</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"❌ Erreur lors de la prédiction : {result['error']}")


def render_form():
    """Affiche et gère le formulaire de saisie."""
    
    # Section 1: Catégorie
    st.markdown('<div class="form-card"><h3>📦 Catégorie</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_main = st.selectbox(
            "Catégorie principale",
            options=MAIN_CATEGORIES,
            index=0,
            key="cat_main"
        )
    
    with col2:
        # Sous-catégories dynamiques
        sub_cats = SUB_CATEGORIES.get(category_main, ["Other"])
        category_sub1 = st.selectbox(
            "Sous-catégorie",
            options=sub_cats,
            index=0,
            key="cat_sub1"
        )
    
    category_sub2 = st.text_input(
        "Précision (optionnel)",
        placeholder="ex: Running Shoes, iPhone 12...",
        key="cat_sub2"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 2: Produit
    st.markdown('<div class="form-card"><h3>🎁 Produit</h3>', unsafe_allow_html=True)
    
    name = st.text_input(
        "Titre de l'annonce",
        placeholder="ex: Nike Air Max 90 - Taille 42 - Neuf",
        key="name"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox(
            "Marque",
            options=TOP_BRANDS,
            index=0,
            format_func=lambda x: "Sélectionner ou autre..." if x == "" else x,
            key="brand_select"
        )
        
        if brand == "":
            brand = st.text_input(
                "Autre marque",
                placeholder="ex: Zara, H&M...",
                key="brand_other"
            )
    
    with col2:
        condition = st.selectbox(
            "État du produit",
            options=list(CONDITION_LABELS.keys()),
            format_func=lambda x: CONDITION_LABELS[x],
            index=2,
            key="condition"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3: Description & Livraison
    st.markdown('<div class="form-card"><h3>📝 Description & Livraison</h3>', unsafe_allow_html=True)
    
    description = st.text_area(
        "Description de l'article",
        placeholder="Décrivez votre article : état, taille, couleur, défauts éventuels...",
        height=120,
        key="description"
    )
    
    shipping = st.radio(
        "Frais de livraison",
        options=[1, 0],
        format_func=lambda x: "📦 Inclus (je paye)" if x == 1 else "💵 À la charge de l'acheteur",
        horizontal=True,
        key="shipping"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'name': name,
        'category_main': category_main,
        'category_sub1': category_sub1,
        'category_sub2': category_sub2 if category_sub2 else category_sub1,
        'brand': brand,
        'condition': condition,
        'description': description,
        'shipping': shipping
    }


# ================================================================================
# APP PRINCIPALE
# ================================================================================

def main():
    # Charger les styles
    load_css()
    
    # Header
    render_header()
    
    # Charger le modèle
    model, preprocessor, error = load_model_and_preprocessor()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("Vérifiez que les fichiers `model_v1.joblib` et `preprocessor_v1.joblib` sont dans le dossier `models/`")
        return
    
    # Formulaire
    form_data = render_form()
    
    # Bouton de prédiction
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Estimer le prix", use_container_width=True):
        
        # Validation
        if not form_data['name']:
            st.warning("⚠️ Veuillez entrer un titre pour l'annonce")
            return
        
        # Préparer les données
        with st.spinner("Analyse en cours..."):
            input_df = prepare_input_data(
                name=form_data['name'],
                category_main=form_data['category_main'],
                category_sub1=form_data['category_sub1'],
                category_sub2=form_data['category_sub2'],
                brand_name=form_data['brand'],
                item_condition_id=form_data['condition'],
                shipping=form_data['shipping'],
                item_description=form_data['description']
            )
            
            # Prédiction
            result = predict_price(model, preprocessor, input_df)
        
        # Afficher le résultat
        render_result(result)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Modèle entraîné sur 1.4M d'annonces Mercari • RMSLE: 0.49</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
