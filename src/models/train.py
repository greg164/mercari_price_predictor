"""
Module: train.py
Description: Entraînement des modèles de prédiction de prix Mercari
Author: Greg
Date: 2025-01

Usage:
    python src/models/train.py                      # Entraînement par défaut (LightGBM)
    python src/models/train.py --model ridge        # Ridge baseline
    python src/models/train.py --model lgbm         # LightGBM
    python src/models/train.py --nrows 50000        # Limiter les données
    python src/models/train.py --cv 5               # Cross-validation 5-fold
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import sys
import json
import logging
import argparse
from typing import Optional, Dict, Tuple, Any
from pathlib import Path
from datetime import datetime

# Ajouter le path racine pour les imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third party
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import joblib

# LightGBM (optionnel mais recommandé)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM non installé. Utiliser: pip install lightgbm")

# Local imports
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.features.pipeline import (
    MercariPreprocessor,
    prepare_data,
    transform_target,
    save_preprocessor
)

# ================================================================================
# CONFIGURATION
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
MODELS_DIR = PROJECT_ROOT / "models"

# Hyperparamètres par défaut
DEFAULT_PARAMS = {
    'ridge': {
        'alpha': 1.0
    },
    'lgbm': {
        'n_estimators': 1000, #1000
        'learning_rate': 0.07,
        'max_depth': 8,
        'num_leaves': 127, #31
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
}


# ================================================================================
# MÉTRIQUES
# ================================================================================

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le Root Mean Squared Logarithmic Error.
    
    Note: Si y est déjà en log (log1p), on calcule simplement le RMSE.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
    
    Returns:
        Score RMSLE.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_model(
    model,
    X: csr_matrix,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, float]:
    """
    Évalue un modèle avec cross-validation.
    
    Args:
        model: Modèle sklearn-compatible.
        X: Features (matrice sparse).
        y: Cible (log-transformée).
        cv: Nombre de folds.
    
    Returns:
        Dict avec les métriques.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation avec neg_mean_squared_error
    scores = cross_val_score(
        model, X, y,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Convertir en RMSLE
    rmsle_scores = np.sqrt(-scores)
    
    return {
        'rmsle_mean': rmsle_scores.mean(),
        'rmsle_std': rmsle_scores.std(),
        'rmsle_scores': rmsle_scores.tolist()
    }


# ================================================================================
# MODÈLES
# ================================================================================

def train_ridge(
    X_train: csr_matrix,
    y_train: np.ndarray,
    params: Dict = None
) -> Ridge:
    """
    Entraîne un modèle Ridge Regression (baseline).
    
    Args:
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        params: Hyperparamètres (optionnel).
    
    Returns:
        Modèle entraîné.
    """
    params = params or DEFAULT_PARAMS['ridge']
    
    logger.info(f"Entraînement Ridge avec alpha={params['alpha']}")
    
    model = Ridge(**params)
    model.fit(X_train, y_train)
    
    logger.info("Ridge entraîné")
    
    return model


def train_lightgbm(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix = None,
    y_val: np.ndarray = None,
    params: Dict = None
) -> Any:
    """
    Entraîne un modèle LightGBM.
    
    Args:
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        X_val: Features de validation (optionnel, pour early stopping).
        y_val: Cible de validation.
        params: Hyperparamètres (optionnel).
    
    Returns:
        Modèle entraîné.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM non installé. Utiliser: pip install lightgbm")
    
    params = params or DEFAULT_PARAMS['lgbm'].copy()
    
    logger.info(f"Entraînement LightGBM avec {params['n_estimators']} estimateurs")
    
    model = lgb.LGBMRegressor(**params)
    
    # Entraînement avec early stopping si validation fournie
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        logger.info(f"Meilleur n_estimators: {model.best_iteration_}")
    else:
        model.fit(X_train, y_train)
    
    logger.info("LightGBM entraîné")
    
    return model

# Après la fonction train_lightgbm(), ajouter :

def train_ensemble(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix = None,
    y_val: np.ndarray = None,
    ridge_params: Dict = None,
    lgbm_params: Dict = None,
    weights: Tuple[float, float] = (0.3, 0.7)
) -> Dict:
    """
    Entraîne un ensemble Ridge + LightGBM.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        ridge_params: Hyperparamètres Ridge
        lgbm_params: Hyperparamètres LightGBM
        weights: Pondération (ridge_weight, lgbm_weight)
    
    Returns:
        Dict avec les deux modèles et les poids
    """
    ridge_params = ridge_params or DEFAULT_PARAMS['ridge']
    lgbm_params = lgbm_params or DEFAULT_PARAMS['lgbm']
    
    logger.info("Entraînement de l'ensemble Ridge + LightGBM")
    
    # Entraîner Ridge
    model_ridge = train_ridge(X_train, y_train, ridge_params)
    
    # Entraîner LightGBM
    model_lgbm = train_lightgbm(X_train, y_train, X_val, y_val, lgbm_params)
    
    # Évaluer chaque modèle sur validation
    if X_val is not None and y_val is not None:
        pred_ridge = model_ridge.predict(X_val)
        pred_lgbm = model_lgbm.predict(X_val)
        
        rmsle_ridge = rmsle(y_val, pred_ridge)
        rmsle_lgbm = rmsle(y_val, pred_lgbm)
        
        # Prédiction ensemble
        pred_ensemble = weights[0] * pred_ridge + weights[1] * pred_lgbm
        rmsle_ensemble = rmsle(y_val, pred_ensemble)
        
        logger.info(f"  Ridge RMSLE: {rmsle_ridge:.4f}")
        logger.info(f"  LightGBM RMSLE: {rmsle_lgbm:.4f}")
        logger.info(f"  Ensemble RMSLE: {rmsle_ensemble:.4f}")
    
    return {
        'ridge': model_ridge,
        'lgbm': model_lgbm,
        'weights': weights,
        'type': 'ensemble'
    }

# ================================================================================
# SAUVEGARDE
# ================================================================================

def save_model(
    model,
    filepath: Optional[Path] = None,
    version: str = "v1"
) -> Path:
    """
    Sauvegarde le modèle avec joblib.
    
    Args:
        model: Modèle entraîné.
        filepath: Chemin de sauvegarde (optionnel).
        version: Version du modèle.
    
    Returns:
        Chemin du fichier sauvegardé.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if filepath is None:
        filepath = MODELS_DIR / f"model_{version}.joblib"
    
    filepath = Path(filepath)
    
    joblib.dump(model, filepath)
    
    logger.info(f"Modèle sauvegardé: {filepath}")
    
    return filepath


def save_metadata(
    metrics: Dict,
    model_type: str,
    params: Dict,
    version: str = "v1"
) -> Path:
    """
    Sauvegarde les métadonnées du modèle.
    
    Args:
        metrics: Métriques d'évaluation.
        model_type: Type de modèle (ridge, lgbm).
        params: Hyperparamètres utilisés.
        version: Version du modèle.
    
    Returns:
        Chemin du fichier sauvegardé.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'version': version,
        'model_type': model_type,
        'created_at': datetime.now().isoformat(),
        'metrics': metrics,
        'params': params
    }
    
    filepath = MODELS_DIR / "metadata.json"
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Métadonnées sauvegardées: {filepath}")
    
    return filepath


def load_model(filepath: Optional[Path] = None, version: str = "v1"):
    """
    Charge un modèle sauvegardé.
    
    Args:
        filepath: Chemin du fichier (optionnel).
        version: Version à charger.
    
    Returns:
        Modèle chargé.
    """
    if filepath is None:
        filepath = MODELS_DIR / f"model_{version}.joblib"
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {filepath}")
    
    model = joblib.load(filepath)
    
    logger.info(f"Modèle chargé: {filepath}")
    
    return model

# ================================================================================
# SELECTION DES TOP N FEATURES
# ================================================================================

def select_top_features(
    model,
    X: csr_matrix,
    feature_names: list[str],
    top_n: int = 1000
) -> Tuple[csr_matrix, list[str], np.ndarray]:
    """
    Sélectionne les top N features selon l'importance du modèle.
    
    Args:
        model: Modèle LightGBM entraîné
        X: Matrice de features
        feature_names: Noms des features
        top_n: Nombre de features à garder
    
    Returns:
        Tuple (X réduit, noms des features gardées, indices des features)
    """
    importances = model.feature_importances_
    
    # Indices des top N features
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_indices = np.sort(top_indices)  # Remettre dans l'ordre original
    
    # Filtrer
    X_reduced = X[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]
    
    logger.info(f"Features réduites: {X.shape[1]} → {X_reduced.shape[1]}")
    
    return X_reduced, selected_names, top_indices

# ================================================================================
# PIPELINE COMPLET
# ================================================================================

def train_full_pipeline(
    nrows: Optional[int] = None,
    model_type: str = 'lgbm',
    tfidf_max_features: int = 10000,
    cv: int = 5,
    version: str = "v1",
    save: bool = True, 
    feature_selection: bool = False, 
    top_n_features: int = 1000
) -> Tuple[Any, MercariPreprocessor, Dict]:
    """
    Pipeline complet d'entraînement.
    
    1. Charge et nettoie les données
    2. Prépare les features
    3. Entraîne le modèle
    4. Évalue avec cross-validation
    5. Sauvegarde modèle + preprocessor
    
    Args:
        nrows: Nombre de lignes à charger (None = tout).
        model_type: Type de modèle ('ridge' ou 'lgbm').
        tfidf_max_features: Nombre max de features TF-IDF.
        cv: Nombre de folds pour cross-validation.
        version: Version pour la sauvegarde.
        save: Si True, sauvegarde le modèle et le preprocessor.
    
    Returns:
        Tuple (modèle, preprocessor, métriques).
    """
    print(f"\n{'='*60}")
    print("PIPELINE D'ENTRAÎNEMENT MERCARI")
    print(f"{'='*60}")
    print(f"  Modèle: {model_type}")
    print(f"  TF-IDF features: {tfidf_max_features}")
    print(f"  Cross-validation: {cv} folds")
    if nrows:
        print(f"  Données: {nrows:,} lignes")
    else:
        print(f"  Données: dataset complet")
    
    # -------------------------------------------------------------------------
    # 1. Chargement et nettoyage
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("1. CHARGEMENT DES DONNÉES")
    print(f"{'='*60}")
    
    df_raw = load_raw_data(nrows=nrows)
    df_clean = clean_data(df_raw)
    
    print(f"  Données nettoyées: {len(df_clean):,} lignes")
    
    # -------------------------------------------------------------------------
    # 2. Split train/validation
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("2. SPLIT TRAIN/VALIDATION")
    print(f"{'='*60}")
    
    X_train, X_val, y_train, y_val = prepare_data(
        df_clean,
        test_size=0.2,
        stratify_col='cat_main'
    )
    
    print(f"  Train: {len(X_train):,} lignes")
    print(f"  Validation: {len(X_val):,} lignes")
    
    # -------------------------------------------------------------------------
    # 3. Feature engineering
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("3. FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    # Transformer la cible
    y_train_log = transform_target(y_train)
    y_val_log = transform_target(y_val)
    
    print(f"  y_train: min={y_train.min():.0f}, max={y_train.max():.0f}, median={y_train.median():.0f}")
    print(f"  y_train_log: min={y_train_log.min():.2f}, max={y_train_log.max():.2f}")
    
    # Créer le preprocessor
    preprocessor = MercariPreprocessor(
        tfidf_max_features=tfidf_max_features,
        include_tfidf=True,
        include_keywords=True
    )
    
    # Fit sur train, transform train et val
    X_train_features = preprocessor.fit_transform(X_train, y_train_log)
    X_val_features = preprocessor.transform(X_val)
    
    print(f"  Features train: {X_train_features.shape}")
    print(f"  Features val: {X_val_features.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Entraînement
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("4. ENTRAÎNEMENT")
    print(f"{'='*60}")
    
    if model_type == 'ridge':
        params = DEFAULT_PARAMS['ridge']
        model = train_ridge(X_train_features, y_train_log, params)
        
    elif model_type == 'lgbm':
        if not HAS_LIGHTGBM:
            print("  ⚠ LightGBM non disponible, fallback sur Ridge")
            model_type = 'ridge'
            params = DEFAULT_PARAMS['ridge']
            model = train_ridge(X_train_features, y_train_log, params)
        else:
            params = DEFAULT_PARAMS['lgbm']
            model = train_lightgbm(
                X_train_features, y_train_log,
                X_val_features, y_val_log,
                params
            )
            
    elif model_type == 'ensemble':
        if not HAS_LIGHTGBM:
            print("  ⚠ LightGBM non disponible, fallback sur Ridge")
            model_type = 'ridge'
            params = DEFAULT_PARAMS['ridge']
            model = train_ridge(X_train_features, y_train_log, params)
        else:
            params = {'ridge': DEFAULT_PARAMS['ridge'], 'lgbm': DEFAULT_PARAMS['lgbm']}
            model = train_ensemble(
                X_train_features, y_train_log,
                X_val_features, y_val_log,
                weights=(0.3, 0.7)
            )
    else:
        raise ValueError(f"Modèle inconnu: {model_type}")
    
    # -------------------------------------------------------------------------
    # 5. ÉVALUATION
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("5. ÉVALUATION")
    print(f"{'='*60}")
    
    # Prédiction selon le type de modèle
    if model_type == 'ensemble':
        y_pred_ridge = model['ridge'].predict(X_val_features)
        y_pred_lgbm = model['lgbm'].predict(X_val_features)
        y_val_pred = model['weights'][0] * y_pred_ridge + model['weights'][1] * y_pred_lgbm
    else:
        y_val_pred = model.predict(X_val_features)
    
    val_rmsle = rmsle(y_val_log, y_val_pred)
    print(f"  RMSLE validation: {val_rmsle:.4f}")
    
    # Cross-validation (sur train seulement)
    if cv > 1:
        print(f"\n  Cross-validation {cv}-fold...")
        cv_metrics = evaluate_model(model, X_train_features, y_train_log, cv=cv)
        print(f"  RMSLE CV: {cv_metrics['rmsle_mean']:.4f} ± {cv_metrics['rmsle_std']:.4f}")
    else:
        cv_metrics = {}
    
    # Métriques complètes
    metrics = {
        'val_rmsle': val_rmsle,
        'cv_rmsle_mean': cv_metrics.get('rmsle_mean'),
        'cv_rmsle_std': cv_metrics.get('rmsle_std'),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_features': preprocessor.n_features_
    }
    
    # Exemple de prédiction
    print(f"\n  Exemple de prédiction:")
    sample_idx = 0
    y_true = y_val.iloc[sample_idx]
    y_pred = np.expm1(y_val_pred[sample_idx])  # Revenir au prix réel
    print(f"    Prix réel: ${y_true:.2f}")
    print(f"    Prix prédit: ${y_pred:.2f}")
    print(f"    Erreur: ${abs(y_true - y_pred):.2f}")
    
    # -------------------------------------------------------------------------
    # 6. Sauvegarde
    # -------------------------------------------------------------------------
    if save:
        print(f"\n{'='*60}")
        print("6. SAUVEGARDE")
        print(f"{'='*60}")
        
        model_path = save_model(model, version=version)
        preprocessor_path = save_preprocessor(preprocessor, version=version)
        metadata_path = save_metadata(metrics, model_type, params, version)
        
        print(f"  ✓ Modèle: {model_path}")
        print(f"  ✓ Preprocessor: {preprocessor_path}")
        print(f"  ✓ Métadonnées: {metadata_path}")
    
    # -------------------------------------------------------------------------
    # Résumé
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  Modèle: {model_type}")
    print(f"  RMSLE validation: {val_rmsle:.4f}")
    if cv_metrics:
        print(f"  RMSLE CV: {cv_metrics['rmsle_mean']:.4f} ± {cv_metrics['rmsle_std']:.4f}")
    print(f"  Features: {metrics['n_features']}")
    
    return model, preprocessor, metrics


# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle Mercari Price Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/models/train.py                      # LightGBM par défaut
  python src/models/train.py --model ridge        # Ridge baseline
  python src/models/train.py --nrows 50000        # Sur 50k lignes
  python src/models/train.py --cv 5               # Cross-validation 5-fold
  python src/models/train.py --no-save            # Sans sauvegarde
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['ridge', 'lgbm', 'ensemble'],
        default='lgbm',
        help='Type de modèle (défaut: lgbm)'
    )
    
    parser.add_argument(
        '--nrows', '-n',
        type=int,
        default=None,
        help='Nombre de lignes à charger (défaut: tout)'
    )
    
    parser.add_argument(
        '--tfidf',
        type=int,
        default=10000,
        help='Nombre max de features TF-IDF (défaut: 10000)'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Nombre de folds pour cross-validation (défaut: 5, 0=désactivé)'
    )
    
    parser.add_argument(
        '--version', '-v',
        default='v1',
        help='Version du modèle (défaut: v1)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Ne pas sauvegarder le modèle'
    )
    
    args = parser.parse_args()
    
    try:
        model, preprocessor, metrics = train_full_pipeline(
            nrows=args.nrows,
            model_type=args.model,
            tfidf_max_features=args.tfidf,
            cv=args.cv,
            version=args.version,
            save=not args.no_save
        )
        
        print(f"\n✓ Entraînement terminé avec succès!")
        
    except FileNotFoundError as e:
        logger.error(f"Erreur: {e}")
        print("\n💡 Télécharge d'abord les données:")
        print("   python scripts/download_data.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise


if __name__ == "__main__":
    main()