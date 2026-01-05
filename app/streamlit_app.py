import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict

# ================================================================================
# CONFIGURATION
# ================================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
CONDITION_LABELS = {
    1: "🌟 Neuf avec étiquettes", 
    2: "✨ Neuf sans étiquettes", 
    3: "👍 Très bon état", 
    4: "👌 Bon état", 
    5: "🔧 État correct"
}

CONDITION_KEYWORDS = {
    1: "brand new with tags nwt unused sealed",
    2: "brand new without tags nwot unused mint condition",
    3: "excellent condition like new barely used",
    4: "good condition gently used minor wear",
    5: "fair condition used visible wear"
}

# Suggestions par catégorie pour aider l'utilisateur
CATEGORY_SUGGESTIONS = {
    "Electronics": "💡 Précisez : modèle, capacité (64gb, 128gb...), état de la batterie, accessoires inclus",
    "Women": "💡 Précisez : taille, matière, couleur, occasion portée",
    "Men": "💡 Précisez : taille, matière, couleur, coupe",
    "Beauty": "💡 Précisez : contenance, date d'ouverture, % restant",
    "Kids": "💡 Précisez : âge recommandé, taille, état",
    "Home": "💡 Précisez : dimensions, matériau, état",
    "Sports & Outdoors": "💡 Précisez : taille, marque, état d'usure",
}

TOP_BRANDS = ["", "Nike", "Victoria's Secret", "LuLaRoe", "Apple", "PINK", "Nintendo", "Lululemon", "Michael Kors", "American Eagle", "Adidas", "Coach", "Rae Dunn", "Bath & Body Works", "Samsung", "Sony", "Disney", "Forever 21", "Kate Spade", "Carter's"]

# Seuil minimum de mots pour une estimation fiable
MIN_WORDS_FOR_RELIABLE_ESTIMATE = 8

# ================================================================================
# CHARGEMENT DU CSS
# ================================================================================

def load_css():
    if CSS_FILE.exists():
        with open(CSS_FILE, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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


def enrich_text(name: str, description: str, brand: str, category_main: str, 
                category_sub1: str, condition: int) -> tuple[str, str]:
    """Enrichit le texte avec marque, catégorie et mots-clés d'état."""
    
    brand_text = brand.lower() if brand and brand != "unknown" else ""
    category_text = f"{category_main} {category_sub1}".lower()
    condition_text = CONDITION_KEYWORDS.get(condition, "")
    
    # Enrichir le titre
    enriched_name = name.lower().strip()
    if brand_text and brand_text not in enriched_name:
        enriched_name = f"{brand_text} {enriched_name}"
    
    # Enrichir la description
    desc_clean = description.lower().strip() if description else ""
    
    parts = [desc_clean] if desc_clean else [enriched_name]
    
    if category_text not in (desc_clean + enriched_name):
        parts.append(category_text)
    if brand_text and brand_text not in desc_clean:
        parts.append(brand_text)
    parts.append(condition_text)
    
    enriched_description = " ".join(parts)
    
    return enriched_name, enriched_description


def count_meaningful_words(name: str, description: str) -> int:
    """Compte les mots significatifs (hors mots très courts)."""
    text = f"{name} {description}".lower()
    words = [w for w in text.split() if len(w) > 2]
    return len(words)


def prepare_input_data(name, category_main, category_sub1, brand_name, 
                       item_condition_id, shipping, item_description) -> pd.DataFrame:
    
    enriched_name, enriched_description = enrich_text(
        name=name,
        description=item_description,
        brand=brand_name,
        category_main=category_main,
        category_sub1=category_sub1,
        condition=item_condition_id
    )
    
    category_sub2 = ""
    category_name = f"{category_main}/{category_sub1}/{category_sub2}"
    
    data = {
        'name': [enriched_name],
        'item_condition_id': [item_condition_id],
        'category_name': [category_name],
        'brand_name': [brand_name if brand_name else "unknown"],
        'shipping': [shipping],
        'item_description': [enriched_description],
        'cat_main': [category_main],
        'cat_sub1': [category_sub1],
        'cat_sub2': [category_sub2]
    }
    return pd.DataFrame(data)


def predict_price(model, preprocessor, input_data, inflation_coefficient=1.30) -> Dict:
    try:
        X = preprocessor.transform(input_data)
        
        if isinstance(model, dict) and model.get('type') == 'ensemble':
            pred_ridge = model['ridge'].predict(X)[0]
            pred_lgbm = model['lgbm'].predict(X)[0]
            y_log_pred = model['weights'][0] * pred_ridge + model['weights'][1] * pred_lgbm
        else:
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

def render_result(result: Dict, is_low_confidence: bool = False):
    if result['success']:
        confidence_warning = ""
        if is_low_confidence:
            confidence_warning = '<div style="color: #ffc107; font-size: 0.85rem; margin-top: 1rem;">⚠️ Estimation approximative - ajoutez plus de détails pour plus de précision</div>'
        
        st.markdown(f"""
        <div class="result-card">
            <div class="price-label">Prix suggéré</div>
            <div class="price-main">${result['predicted_price']:.0f}</div>
            <div class="price-range">Fourchette : ${result['price_low']:.0f} — ${result['price_high']:.0f}</div>
            {confidence_warning}
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 2: Produit
    st.markdown('<div class="form-card"><h3>🎁 Produit</h3>', unsafe_allow_html=True)
    name = st.text_input("Titre de l'annonce", placeholder="ex: iPhone 12 64GB débloqué", key="name")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marque", options=TOP_BRANDS, format_func=lambda x: "Sélectionner ou autre..." if x == "" else x)
        if brand == "": 
            brand = st.text_input("Autre marque", placeholder="ex: Zara...")
    with col2:
        condition = st.selectbox("État", options=list(CONDITION_LABELS.keys()), format_func=lambda x: CONDITION_LABELS[x], index=2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3: Description
    st.markdown('<div class="form-card"><h3>📝 Détails</h3>', unsafe_allow_html=True)
    
    # Afficher suggestion selon catégorie
    suggestion = CATEGORY_SUGGESTIONS.get(category_main, "💡 Plus vous détaillez, plus l'estimation sera précise")
    st.caption(suggestion)
    
    description = st.text_area("Description", height=120, 
                               placeholder="Ex: iPhone 12 64GB, couleur noir, batterie 89%, débloqué tout opérateur, vendu avec boîte et chargeur d'origine")
    shipping = st.radio("Frais de livraison", options=[1, 0], format_func=lambda x: "📦 Inclus" if x == 1 else "💵 À charge acheteur", horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'name': name, 
        'category_main': category_main, 
        'category_sub1': category_sub1,
        'brand': brand, 
        'condition': condition, 
        'description': description, 
        'shipping': shipping
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
        
        # Vérifier si assez d'informations
        word_count = count_meaningful_words(form_data['name'], form_data['description'])
        is_low_confidence = word_count < MIN_WORDS_FOR_RELIABLE_ESTIMATE
        
        if is_low_confidence:
            st.warning(f"⚠️ **Description courte détectée** ({word_count} mots). Pour une estimation plus précise, ajoutez des détails : modèle exact, capacité, état, accessoires inclus...")
            
        with st.spinner("Analyse en cours..."):
            input_df = prepare_input_data(
                form_data['name'], 
                form_data['category_main'], 
                form_data['category_sub1'],
                form_data['brand'], 
                form_data['condition'], 
                form_data['shipping'], 
                form_data['description']
            )
            result = predict_price(model, preprocessor, input_df)
            render_result(result, is_low_confidence)

if __name__ == "__main__":
    main()