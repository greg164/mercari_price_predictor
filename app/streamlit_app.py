import sys
import json
import requests
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict
from deep_translator import GoogleTranslator

# ================================================================================
# CONFIGURATION
# ================================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Chemins relatifs
ASSETS_DIR = Path(__file__).parent / "assets"
CSS_FILE = ASSETS_DIR / "style.css"
TRANSLATIONS_FILE = ASSETS_DIR / "translations.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_v1.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_v1.joblib"

st.set_page_config(
    page_title="Mercari Price Predictor",
    page_icon="🏷️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CONSTANTES MODÈLE (Restent en anglais) ---
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

# Clés entières pour le modèle
CONDITION_IDS = [1, 2, 3, 4, 5]

# Mots clés pour enrichir le texte (anglais car modèle anglais)
CONDITION_KEYWORDS = {
    1: "brand new with tags nwt unused sealed",
    2: "brand new without tags nwot unused mint condition",
    3: "excellent condition like new barely used",
    4: "good condition gently used minor wear",
    5: "fair condition used visible wear"
}

TOP_BRANDS = ["", "Nike", "Victoria's Secret", "LuLaRoe", "Apple", "PINK", "Nintendo", "Lululemon", "Michael Kors", "American Eagle", "Adidas", "Coach", "Rae Dunn", "Bath & Body Works", "Samsung", "Sony", "Disney", "Forever 21", "Kate Spade", "Carter's"]

MIN_WORDS_FOR_RELIABLE_ESTIMATE = 8

# ================================================================================
# GESTION DES TRADUCTIONS (I18N)
# ================================================================================

@st.cache_data
def load_translations():
    if TRANSLATIONS_FILE.exists():
        with open(TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_text(key: str) -> str:
    """Récupère un texte simple."""
    translations = load_translations()
    lang = st.session_state.get('lang_code', 'fr')
    text = translations.get(lang, {}).get(key)
    if not text:
        text = translations.get('en', {}).get(key, key)
    return text

def get_category_display(cat_english: str) -> str:
    """Traduit une catégorie principale."""
    translations = load_translations()
    lang = st.session_state.get('lang_code', 'fr')
    
    # Si anglais, on retourne direct (car JSON "categories" est vide en EN)
    if lang == 'en': return cat_english
    
    cat_translated = translations.get(lang, {}).get("categories", {}).get(cat_english)
    return cat_translated if cat_translated else cat_english

def get_subcategory_display(subcat_english: str) -> str:
    """Traduit une sous-catégorie."""
    translations = load_translations()
    lang = st.session_state.get('lang_code', 'fr')

    if lang == 'en': return subcat_english

    # Cherche dans "sub_categories"
    sub_translated = translations.get(lang, {}).get("sub_categories", {}).get(subcat_english)
    return sub_translated if sub_translated else subcat_english

def get_condition_display(condition_id: int) -> str:
    """Traduit un état (1-5)."""
    translations = load_translations()
    lang = st.session_state.get('lang_code', 'fr')
    
    # Les clés JSON sont des strings "1", "2"...
    cond_translated = translations.get(lang, {}).get("conditions", {}).get(str(condition_id))
    
    # Fallback anglais si pas trouvé
    if not cond_translated:
        cond_translated = translations.get("en", {}).get("conditions", {}).get(str(condition_id), str(condition_id))
        
    return cond_translated

def translate_text(text: str, source_lang: str) -> str:
    """
    Traduit le texte vers l'anglais via Google Translate (deep-translator).
    Beaucoup plus fiable que les instances publiques LibreTranslate.
    """
    if not text or source_lang == 'en':
        return text
    
    # Mapping des codes langues si nécessaire (Google utilise 'fr', 'es', 'en'...)
    # Ton sélecteur renvoie déjà 'fr', 'es', 'en', donc c'est parfait.
    
    try:
        translator = GoogleTranslator(source=source_lang, target='en')
        translated = translator.translate(text)
        return translated
    except Exception as e:
        # En cas d'erreur (pas de connexion internet par ex), on log et on renvoie l'original
        print(f"Erreur traduction: {e}")
        return text

# ================================================================================
# LOGIQUE MODÈLE
# ================================================================================

@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

def enrich_text(name, description, brand, category_main, category_sub1, condition):
    brand_text = brand.lower() if brand and brand != "unknown" else ""
    category_text = f"{category_main} {category_sub1}".lower()
    condition_text = CONDITION_KEYWORDS.get(condition, "")
    
    enriched_name = name.lower().strip()
    if brand_text and brand_text not in enriched_name:
        enriched_name = f"{brand_text} {enriched_name}"
    
    desc_clean = description.lower().strip() if description else ""
    parts = [desc_clean] if desc_clean else [enriched_name]
    
    if category_text not in (desc_clean + enriched_name): parts.append(category_text)
    if brand_text and brand_text not in desc_clean: parts.append(brand_text)
    parts.append(condition_text)
    
    return enriched_name, " ".join(parts)

def count_meaningful_words(name, description):
    text = f"{name} {description}".lower()
    return len([w for w in text.split() if len(w) > 2])

def prepare_input_data(name, category_main, category_sub1, brand_name, 
                       item_condition_id, shipping, item_description):
    
    enriched_name, enriched_description = enrich_text(
        name, item_description, brand_name, category_main, category_sub1, item_condition_id
    )
    
    data = {
        'name': [enriched_name],
        'item_condition_id': [item_condition_id],
        'category_name': [f"{category_main}/{category_sub1}/"],
        'brand_name': [brand_name if brand_name else "unknown"],
        'shipping': [shipping],
        'item_description': [enriched_description],
        'cat_main': [category_main],
        'cat_sub1': [category_sub1],
        'cat_sub2': [""]
    }
    return pd.DataFrame(data)

def predict_price(model, preprocessor, input_data):
    try:
        X = preprocessor.transform(input_data)
        if isinstance(model, dict) and model.get('type') == 'ensemble':
            pred = (model['weights'][0] * model['ridge'].predict(X)[0] + 
                    model['weights'][1] * model['lgbm'].predict(X)[0])
        else:
            pred = model.predict(X)[0]
        
        price = np.expm1(pred) * 1.30 
        return {
            'predicted_price': max(1, price),
            'price_low': max(1, price * 0.65),
            'price_high': price * 1.55,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ================================================================================
# UI
# ================================================================================

def load_css():
    if CSS_FILE.exists():
        with open(CSS_FILE, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_form():
    # Catégorie
    st.markdown(f'<div class="form-card"><h3>{get_text("cat_title")}</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        # Affiche traduit, retourne Anglais
        category_main = st.selectbox(get_text("cat_main_label"), options=MAIN_CATEGORIES, 
                                     format_func=get_category_display, key="cat_main")
    with c2:
        sub_cats = SUB_CATEGORIES.get(category_main, ["Other"])
        # ICI: on utilise get_subcategory_display pour traduire
        category_sub1 = st.selectbox(get_text("cat_sub_label"), options=sub_cats, 
                                     format_func=get_subcategory_display, key="cat_sub1")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Produit
    st.markdown(f'<div class="form-card"><h3>{get_text("prod_title")}</h3>', unsafe_allow_html=True)
    name = st.text_input(get_text("prod_name_label"), placeholder=get_text("prod_name_placeholder"))
    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox(get_text("brand_label"), options=TOP_BRANDS, 
                             format_func=lambda x: get_text("brand_select_default") if x == "" else x)
        if brand == "": brand = st.text_input(get_text("brand_placeholder"))
    with c2:
        # ICI: on utilise get_condition_display pour traduire
        cond = st.selectbox(get_text("condition_label"), options=CONDITION_IDS, 
                            format_func=get_condition_display, index=2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Détails
    st.markdown(f'<div class="form-card"><h3>{get_text("details_title")}</h3>', unsafe_allow_html=True)
    desc = st.text_area(get_text("desc_label"), height=120, placeholder=get_text("desc_placeholder"))
    ship = st.radio(get_text("shipping_label"), options=[1, 0], horizontal=True,
                    format_func=lambda x: get_text("shipping_included") if x == 1 else get_text("shipping_paid"))
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {'name': name, 'cat_main': category_main, 'cat_sub1': category_sub1, 
            'brand': brand, 'condition': cond, 'desc': desc, 'shipping': ship}

def main():
    load_css()
    
    # Sélecteur langue
    col1, col2 = st.columns([6, 1])
    with col2:
        options = {"🇫🇷 FR": "fr", "🇺🇸 EN": "en", "🇪🇸 ES": "es"}
        sel = st.selectbox("Lang", options=options.keys(), label_visibility="collapsed")
        st.session_state['lang_code'] = options[sel]

    st.markdown(f'<div class="main-header"><h1>{get_text("header_title")}</h1><p>{get_text("header_subtitle")}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">{get_text("disclaimer")}</div>', unsafe_allow_html=True)
    
    model, preprocessor, error = load_model_and_preprocessor()
    if error:
        st.error(f"⚠️ {error}")
        return
    
    data = render_form()
    
    if st.button(get_text("btn_estimate"), use_container_width=True):
        if not data['name']:
            st.warning(get_text("warn_title"))
            return
            
        wc = count_meaningful_words(data['name'], data['desc'])
        is_low_confidence = wc < MIN_WORDS_FOR_RELIABLE_ESTIMATE
        if is_low_confidence:
            st.warning(get_text("warn_short_desc").format(count=wc))
            
        with st.spinner(get_text("translating")):
            lang = st.session_state.get('lang_code', 'fr')
            final_name = data['name']
            final_desc = data['desc']
            
            # On traduit si la langue n'est pas l'anglais
            if lang != 'en':
                try:
                    # Appel de la nouvelle fonction
                    final_name = translate_text(data['name'], lang)
                    final_desc = translate_text(data['desc'], lang)
                except Exception as e:
                    st.warning(get_text("warn_api_fail"))

            input_df = prepare_input_data(final_name, data['cat_main'], data['cat_sub1'], 
                                          data['brand'], data['condition'], data['shipping'], final_desc)
            
            res = predict_price(model, preprocessor, input_df)
            
            if res['success']:
                warn = f'<div style="color: #ffc107; font-size: 0.85rem; margin-top: 1rem;">{get_text("result_low_conf")}</div>' if is_low_confidence else ""
                st.markdown(f"""
                <div class="result-card">
                    <div class="price-label">{get_text("result_label")}</div>
                    <div class="price-main">${res['predicted_price']:.0f}</div>
                    <div class="price-range">{get_text("result_range")} : ${res['price_low']:.0f} — ${res['price_high']:.0f}</div>
                    {warn}
                </div>""", unsafe_allow_html=True)
                
                # if lang != 'en':
                #     with st.expander(get_text("debug_translation")):
                #         st.code(f"Title: {final_name}\nDesc:  {final_desc}")

                # # --- DEBUG : AFFICHER LA VÉRITÉ ---
                # # On récupère ce qui est vraiment dans le DataFrame envoyé au modèle
                # real_model_input = input_df['item_description'][0]
                
                # with st.expander("🔍 Debug : Ce que le modèle voit (Final Feature)"):
                #     st.markdown("**1. Traduction simple (DeepTranslator) :**")
                #     st.text(f"Titre : {final_name}")
                #     st.text(f"Desc  : {final_desc}")
                    
                #     st.markdown("**2. Texte Enrichi final (Envoyé au Vectorizer) :**")
                #     st.caption("Contient : Traduction + Marque + Catégorie + Mots-clés d'état")
                #     st.code(real_model_input, language="text")

            else:
                st.error(f"{get_text('error')} {res['error']}")

if __name__ == "__main__":
    main()