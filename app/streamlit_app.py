import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional

# ================================================================================
# CONFIGURATION
# ================================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Chemin du fichier CSS (au même endroit que le script)
CSS_FILE = Path(__file__).parent / "assets/style.css"

st.set_page_config(
    page_title="Mercari Price Predictor",
    page_icon="🏷️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_v1.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_v1.joblib"

# [Vos listes MAIN_CATEGORIES, SUB_CATEGORIES, CONDITION_LABELS, TOP_BRANDS restent ici...]
MAIN_CATEGORIES = ["Women", "Men", "Beauty", "Electronics", "Kids", "Home", "Sports & Outdoors", "Handmade", "Vintage & Collectibles", "Other"]
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
CONDITION_LABELS = {1: "🌟 Neuf avec étiquettes", 2: "✨ Neuf sans étiquettes", 3: "👍 Très bon état", 4: "👌 Bon état", 5: "🔧 État correct"}
TOP_BRANDS = ["", "Nike", "Victoria's Secret", "LuLaRoe", "Apple", "PINK", "Nintendo", "Lululemon", "Michael Kors", "American Eagle", "Adidas", "Coach", "Rae Dunn", "Bath & Body Works", "Samsung", "Sony", "Disney", "Forever 21", "Kate Spade", "Carter's"]

# ================================================================================
# CHARGEMENT DU CSS
# ================================================================================

def load_css():
    """Lit le fichier CSS externe et l'applique."""
    if CSS_FILE.exists():
        with open(CSS_FILE, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Fichier style.css non trouvé.")

# ================================================================================
# FONCTIONS DE PRÉDICTION
# ================================================================================

@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

def prepare_input_data(name, category_main, category_sub1, brand_name, item_condition_id, shipping, item_description) -> pd.DataFrame:
    # On définit category_sub2 comme une chaîne vide (car on a enlevé le champ texte)
    category_sub2 = "" 
    category_name = f"{category_main}/{category_sub1}/{category_sub2}"
    
    data = {
        'name': [name.lower().strip()],
        'item_condition_id': [item_condition_id],
        'category_name': [category_name],
        'brand_name': [brand_name if brand_name else "unknown"],
        'shipping': [shipping],
        'item_description': [item_description.lower().strip() if item_description else ""],
        'cat_main': [category_main],
        'cat_sub1': [category_sub1],
        'cat_sub2': [category_sub2]
    }
    return pd.DataFrame(data)

def predict_price(model, preprocessor, input_data, inflation_coefficient=1.30) -> Dict:
    try:
        X = preprocessor.transform(input_data)
        y_log_pred = model.predict(X)[0]
        predicted_price = np.expm1(y_log_pred) * inflation_coefficient
        return {
            'predicted_price': max(1, predicted_price),
            'price_low': max(1, predicted_price * 0.65),
            'price_high': predicted_price * 1.55,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ================================================================================
# COMPOSANTS UI
# ================================================================================

def render_header():
    st.markdown('<div class="main-header"><h1>🏷️ Price Predictor</h1><p>Estimez le prix optimal de vente</p></div>', unsafe_allow_html=True)

def render_disclaimer():
    st.markdown('<div class="disclaimer">⚠️ <strong>Prix basés sur 2017-2018.</strong> Inflation de 30% appliquée.</div>', unsafe_allow_html=True)

def render_result(result: Dict):
    if result['success']:
        st.markdown(f"""
        <div class="result-card">
            <div class="price-label">Prix suggéré</div>
            <div class="price-main">${result['predicted_price']:.0f}</div>
            <div class="price-range">Fourchette : ${result['price_low']:.0f} — ${result['price_high']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"❌ Erreur : {result['error']}")

def render_form():
    # Section 1: Catégorie
    st.markdown('<div class="form-card"><h3>📦 Catégorie</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        category_main = st.selectbox("Catégorie principale", options=MAIN_CATEGORIES, key="cat_main")
    with col2:
        sub_cats = SUB_CATEGORIES.get(category_main, ["Other"])
        category_sub1 = st.selectbox("Sous-catégorie", options=sub_cats, key="cat_sub1")
    
    # LA PARTIE "PRÉCISION (OPTIONNEL)" A ÉTÉ SUPPRIMÉE ICI
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 2: Produit
    st.markdown('<div class="form-card"><h3>🎁 Produit</h3>', unsafe_allow_html=True)
    name = st.text_input("Titre de l'annonce", placeholder="ex: Nike Air Max 90", key="name")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marque", options=TOP_BRANDS, format_func=lambda x: "Autre..." if x == "" else x)
        if brand == "": brand = st.text_input("Autre marque", placeholder="ex: Zara...")
    with col2:
        condition = st.selectbox("État", options=list(CONDITION_LABELS.keys()), format_func=lambda x: CONDITION_LABELS[x], index=2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3: Description
    st.markdown('<div class="form-card"><h3>📝 Détails</h3>', unsafe_allow_html=True)
    description = st.text_area("Description", height=120)
    shipping = st.radio("Frais de livraison", options=[1, 0], format_func=lambda x: "📦 Inclus" if x == 1 else "💵 À charge acheteur", horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'name': name, 'category_main': category_main, 'category_sub1': category_sub1,
        'brand': brand, 'condition': condition, 'description': description, 'shipping': shipping
    }

# ================================================================================
# MAIN
# ================================================================================

def main():
    load_css()
    render_header()
    render_disclaimer()
    
    model, preprocessor, error = load_model_and_preprocessor()
    if error:
        st.error(f"⚠️ {error}")
        return
    
    form_data = render_form()
    
    if st.button("🔮 Estimer le prix", use_container_width=True):
        if not form_data['name']:
            st.warning("⚠️ Veuillez entrer un titre")
            return
            
        with st.spinner("Analyse..."):
            input_df = prepare_input_data(
                form_data['name'], form_data['category_main'], form_data['category_sub1'],
                form_data['brand'], form_data['condition'], form_data['shipping'], form_data['description']
            )
            result = predict_price(model, preprocessor, input_df)
            render_result(result)

if __name__ == "__main__":
    main()