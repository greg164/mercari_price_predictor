# tests/debug_prediction.py
import sys
from pathlib import Path

# Ajouter le chemin racine au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

model = joblib.load(PROJECT_ROOT / "models/model_v1.joblib")
preprocessor = joblib.load(PROJECT_ROOT / "models/preprocessor_v1.joblib")

# Test avec iPhone
test_data = pd.DataFrame({
    'name': ['iphone x 64gb unlocked mint condition'],
    'item_condition_id': [2],
    'category_name': ['Electronics/Cell Phones & Accessories/Cell Phones'],
    'brand_name': ['Apple'],
    'shipping': [1],
    'item_description': ['iphone x 64gb space gray unlocked works perfectly no scratches'],
    'cat_main': ['Electronics'],
    'cat_sub1': ['Cell Phones & Accessories'],
    'cat_sub2': ['Cell Phones']
})

print("=== DEBUG PREDICTION ===")
print(f"Preprocessor type: {type(preprocessor)}")
print(f"Model type: {type(model)}")

X = preprocessor.transform(test_data)
print(f"Shape features: {X.shape}")
print(f"Features non-nulles: {X.nnz}")

# Prédiction
if isinstance(model, dict):
    pred_ridge = model['ridge'].predict(X)[0]
    pred_lgbm = model['lgbm'].predict(X)[0]
    print(f"Pred Ridge (log): {pred_ridge:.3f} -> ${np.expm1(pred_ridge):.2f}")
    print(f"Pred LightGBM (log): {pred_lgbm:.3f} -> ${np.expm1(pred_lgbm):.2f}")
    y_log = 0.3 * pred_ridge + 0.7 * pred_lgbm
else:
    y_log = model.predict(X)[0]

print(f"\nPred finale (log): {y_log:.3f}")
print(f"Prix prédit: ${np.expm1(y_log):.2f}")
print(f"Prix avec inflation 30%: ${np.expm1(y_log) * 1.30:.2f}")

# Ajoute ça à ton script debug

# Test avec enrichissement (comme la nouvelle version Streamlit)
CONDITION_KEYWORDS = {
    1: "brand new with tags nwt unused sealed original packaging",
    2: "brand new without tags nwot unused mint condition never worn",
    3: "excellent condition like new barely used very good",
    4: "good condition gently used minor wear clean",
    5: "fair condition used visible wear some flaws"
}

name = "iphone"
brand = "Apple"
category_main = "Electronics"
category_sub1 = "Cell Phones & Accessories"
condition = 2

# Enrichissement
enriched_name = f"{brand.lower()} {name.lower()}"
enriched_desc = f"{enriched_name} {category_main.lower()} {category_sub1.lower()} {brand.lower()} {CONDITION_KEYWORDS[condition]}"

print(f"\n=== TEST ENRICHI ===")
print(f"Titre enrichi: {enriched_name}")
print(f"Description enrichie: {enriched_desc}")

test_enriched = pd.DataFrame({
    'name': [enriched_name],
    'item_condition_id': [condition],
    'category_name': [f'{category_main}/{category_sub1}/'],
    'brand_name': [brand],
    'shipping': [1],
    'item_description': [enriched_desc],
    'cat_main': [category_main],
    'cat_sub1': [category_sub1],
    'cat_sub2': ['']
})

X_enriched = preprocessor.transform(test_enriched)
print(f"Features non-nulles: {X_enriched.nnz} / {X_enriched.shape[1]}")

if isinstance(model, dict):
    pred_ridge = model['ridge'].predict(X_enriched)[0]
    pred_lgbm = model['lgbm'].predict(X_enriched)[0]
    print(f"Ridge: ${np.expm1(pred_ridge):.2f}")
    print(f"LightGBM: ${np.expm1(pred_lgbm):.2f}")
    y_log = 0.3 * pred_ridge + 0.7 * pred_lgbm
else:
    y_log = model.predict(X_enriched)[0]

print(f"Prix prédit: ${np.expm1(y_log):.2f}")

# Ajoute ce test - simuler une description plus spécifique

test_specific = pd.DataFrame({
    'name': ['apple iphone 64gb unlocked'],
    'item_condition_id': [2],
    'category_name': ['Electronics/Cell Phones & Accessories/'],
    'brand_name': ['Apple'],
    'shipping': [1],
    'item_description': ['apple iphone 64gb unlocked smartphone cell phone excellent condition works perfectly'],
    'cat_main': ['Electronics'],
    'cat_sub1': ['Cell Phones & Accessories'],
    'cat_sub2': ['']
})

X_spec = preprocessor.transform(test_specific)
print(f"\n=== TEST SPÉCIFIQUE (64gb unlocked) ===")
print(f"Features non-nulles: {X_spec.nnz}")

if isinstance(model, dict):
    y_log = 0.3 * model['ridge'].predict(X_spec)[0] + 0.7 * model['lgbm'].predict(X_spec)[0]
else:
    y_log = model.predict(X_spec)[0]

print(f"Prix prédit: ${np.expm1(y_log):.2f}")