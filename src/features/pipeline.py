"""
Module: pipeline.py
Description: Pipeline complet de preprocessing pour le modèle Mercari
Author: Greg
Date: 2025-01

Ce module combine les features textuelles et catégorielles en un pipeline
sklearn reproductible et sauvegardable.
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import sys
import logging
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

# Ajouter le path racine pour les imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third party
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import joblib

# Local imports
from src.features.text_features import (
    TextFeatureExtractor,
    extract_keyword_features,
    create_tfidf_features
)
from src.features.categorical_features import (
    CategoricalEncoder,
    create_categorical_features
)

# ================================================================================
# CONFIGURATION
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Colonnes requises
REQUIRED_COLUMNS = [
    'name', 'item_description', 'item_condition_id', 
    'category_name', 'brand_name', 'shipping'
]


# ================================================================================
# PIPELINE COMPLET
# ================================================================================

class MercariPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline de preprocessing complet pour le dataset Mercari.
    
    Combine:
    - Features textuelles (TF-IDF sur name + description)
    - Features mots-clés (qualité, luxe)
    - Features catégorielles (one-hot, target encoding)
    - Features numériques (condition, shipping, longueurs)
    
    Compatible avec sklearn et sauvegardable avec joblib.
    """
    
    def __init__(self,
                 tfidf_max_features: int = 10000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 2),
                 onehot_columns: List[str] = None,
                 target_columns: List[str] = None,
                 passthrough_columns: List[str] = None,
                 include_tfidf: bool = True,
                 include_keywords: bool = True,
                 target_smoothing: float = 10.0):
        """
        Args:
            tfidf_max_features: Nombre max de features TF-IDF.
            tfidf_ngram_range: Plage de n-grams pour TF-IDF.
            onehot_columns: Colonnes pour one-hot encoding.
            target_columns: Colonnes pour target encoding.
            passthrough_columns: Colonnes numériques à passer telles quelles.
            include_tfidf: Inclure les features TF-IDF.
            include_keywords: Inclure les features de mots-clés.
            target_smoothing: Lissage pour target encoding.
        """
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.onehot_columns = onehot_columns or ['cat_main']
        self.target_columns = target_columns or ['brand_name', 'cat_sub1', 'cat_sub2']
        self.passthrough_columns = passthrough_columns or ['item_condition_id', 'shipping']
        self.include_tfidf = include_tfidf
        self.include_keywords = include_keywords
        self.target_smoothing = target_smoothing
        
        # Encodeurs (initialisés lors du fit)
        self.tfidf_encoder_ = None
        self.categorical_encoder_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self._is_fitted = False
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Vérifie que les colonnes requises sont présentes."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
    
    def _extract_numeric_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrait les features numériques additionnelles.
        
        Features créées:
        - name_len: longueur du titre
        - desc_len: longueur de la description
        - has_description: présence de description
        """
        features = []
        feature_names = []
        
        # Longueur du titre
        name_len = df['name'].fillna('').str.split().str.len().fillna(0).values
        features.append(name_len.reshape(-1, 1))
        feature_names.append('name_len')
        
        # Longueur de la description
        desc_len = df['item_description'].fillna('').str.split().str.len().fillna(0).values
        features.append(desc_len.reshape(-1, 1))
        feature_names.append('desc_len')
        
        # Indicateur de description
        has_desc = (desc_len > 0).astype(int).reshape(-1, 1)
        features.append(has_desc)
        feature_names.append('has_description')
        
        return np.hstack(features), feature_names
    
    def _extract_keyword_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extrait les features de mots-clés.
        
        Retourne un array avec les scores de qualité/luxe.
        """
        df_kw = extract_keyword_features(df, 'item_description')
        
        kw_columns = [
            'kw_excellent', 'kw_good', 'kw_fair', 'kw_poor',
            'kw_luxury', 'kw_budget', 'quality_score',
            'has_quality_keywords', 'has_luxury_keywords'
        ]
        
        return df_kw[kw_columns].values, kw_columns
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Entraîne le pipeline de preprocessing.
        
        Args:
            X: DataFrame avec les données (doit contenir les colonnes requises).
            y: Series avec la cible (prix, requis pour target encoding).
        
        Returns:
            self
        """
        logger.info(f"Fit du preprocessor sur {len(X):,} lignes")
        
        self._validate_columns(X)
        
        if y is None:
            raise ValueError("y (prix) requis pour le target encoding")
        
        self.feature_names_ = []
        
        # 1. TF-IDF
        if self.include_tfidf:
            self.tfidf_encoder_ = TextFeatureExtractor(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                text_columns=['name', 'item_description']
            )
            self.tfidf_encoder_.fit(X)
            self.feature_names_.extend(
                [f"tfidf_{name}" for name in self.tfidf_encoder_.get_feature_names()]
            )
            logger.info(f"  TF-IDF: {len(self.tfidf_encoder_.get_feature_names())} features")
        
        # 2. Categorical encoding
        self.categorical_encoder_ = CategoricalEncoder(
            onehot_columns=self.onehot_columns,
            target_columns=self.target_columns,
            passthrough_columns=self.passthrough_columns,
            target_smoothing=self.target_smoothing
        )
        self.categorical_encoder_.fit(X, y)
        self.feature_names_.extend(self.categorical_encoder_.get_feature_names())
        logger.info(f"  Categorical: {len(self.categorical_encoder_.get_feature_names())} features")
        
        # 3. Numeric features (pas de fit nécessaire)
        _, num_names = self._extract_numeric_features(X)
        self.feature_names_.extend(num_names)
        logger.info(f"  Numeric: {len(num_names)} features")
        
        # 4. Keyword features (pas de fit nécessaire)
        if self.include_keywords:
            _, kw_names = self._extract_keyword_features(X)
            self.feature_names_.extend(kw_names)
            logger.info(f"  Keywords: {len(kw_names)} features")
        
        self.n_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        logger.info(f"Preprocessor entraîné: {self.n_features_} features totales")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> csr_matrix:
        """
        Transforme les données.
        
        Args:
            X: DataFrame à transformer.
        
        Returns:
            Matrice sparse avec toutes les features.
        """
        if not self._is_fitted:
            raise ValueError("Le preprocessor n'a pas été entraîné. Appeler fit() d'abord.")
        
        self._validate_columns(X)
        
        logger.info(f"Transform sur {len(X):,} lignes")
        
        parts = []
        
        # 1. TF-IDF
        if self.include_tfidf and self.tfidf_encoder_ is not None:
            tfidf_matrix = self.tfidf_encoder_.transform(X)
            parts.append(tfidf_matrix)
        
        # 2. Categorical
        cat_matrix = self.categorical_encoder_.transform(X)
        parts.append(csr_matrix(cat_matrix))
        
        # 3. Numeric
        num_matrix, _ = self._extract_numeric_features(X)
        parts.append(csr_matrix(num_matrix))
        
        # 4. Keywords
        if self.include_keywords:
            kw_matrix, _ = self._extract_keyword_features(X)
            parts.append(csr_matrix(kw_matrix))
        
        # Combiner toutes les parties
        result = hstack(parts, format='csr')
        
        logger.info(f"Résultat: {result.shape}")
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> csr_matrix:
        """Fit et transform en une seule étape."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms de toutes les features."""
        return self.feature_names_ or []
    
    def get_params(self, deep: bool = True) -> Dict:
        """Retourne les paramètres du preprocessor."""
        return {
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range,
            'onehot_columns': self.onehot_columns,
            'target_columns': self.target_columns,
            'passthrough_columns': self.passthrough_columns,
            'include_tfidf': self.include_tfidf,
            'include_keywords': self.include_keywords,
            'target_smoothing': self.target_smoothing
        }


# ================================================================================
# FONCTIONS UTILITAIRES
# ================================================================================

def transform_target(y: pd.Series, inverse: bool = False) -> np.ndarray:
    """
    Transforme la cible (prix) avec log(1 + price).
    
    La distribution des prix suit une loi log-normale, cette transformation
    la rend plus proche d'une distribution normale.
    
    Args:
        y: Series des prix.
        inverse: Si True, applique la transformation inverse (exp(y) - 1).
    
    Returns:
        Array transformé.
    """
    if inverse:
        return np.expm1(y)  # exp(y) - 1
    else:
        return np.log1p(y)  # log(1 + y)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prépare les données pour l'entraînement (split train/validation).
    
    Args:
        df: DataFrame nettoyé avec la colonne 'price'.
        test_size: Proportion pour la validation.
        random_state: Seed pour reproductibilité.
        stratify_col: Colonne pour stratification (optionnel).
    
    Returns:
        Tuple (X_train, X_val, y_train, y_val).
    """
    if 'price' not in df.columns:
        raise ValueError("Colonne 'price' manquante")
    
    # Séparer features et cible
    y = df['price']
    X = df.drop(columns=['price'])
    
    # Supprimer les colonnes non nécessaires
    cols_to_drop = ['train_id'] if 'train_id' in X.columns else []
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Stratification optionnelle
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    logger.info(f"Split: {len(X_train):,} train, {len(X_val):,} validation")
    
    return X_train, X_val, y_train, y_val


def create_full_pipeline(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    tfidf_max_features: int = 10000,
    include_tfidf: bool = True,
    include_keywords: bool = True
) -> Tuple[csr_matrix, MercariPreprocessor]:
    """
    Crée et entraîne le pipeline complet.
    
    Args:
        df_train: DataFrame d'entraînement.
        y_train: Series cible (prix bruts, sera transformée en log).
        tfidf_max_features: Nombre max de features TF-IDF.
        include_tfidf: Inclure TF-IDF.
        include_keywords: Inclure mots-clés.
    
    Returns:
        Tuple (matrice features, preprocessor entraîné).
    """
    # Transformer la cible
    y_log = transform_target(y_train)
    
    # Créer et entraîner le preprocessor
    preprocessor = MercariPreprocessor(
        tfidf_max_features=tfidf_max_features,
        include_tfidf=include_tfidf,
        include_keywords=include_keywords
    )
    
    X_features = preprocessor.fit_transform(df_train, y_log)
    
    return X_features, preprocessor


# ================================================================================
# SAUVEGARDE / CHARGEMENT
# ================================================================================

def save_preprocessor(
    preprocessor: MercariPreprocessor,
    filepath: Optional[Path] = None,
    version: str = "v1"
) -> Path:
    """
    Sauvegarde le preprocessor avec joblib.
    
    Args:
        preprocessor: Preprocessor entraîné.
        filepath: Chemin de sauvegarde (optionnel).
        version: Version du preprocessor.
    
    Returns:
        Chemin du fichier sauvegardé.
    """
    if filepath is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = MODELS_DIR / f"preprocessor_{version}.joblib"
    
    filepath = Path(filepath)
    
    joblib.dump(preprocessor, filepath)
    
    logger.info(f"Preprocessor sauvegardé: {filepath}")
    
    return filepath


def load_preprocessor(filepath: Optional[Path] = None, version: str = "v1") -> MercariPreprocessor:
    """
    Charge un preprocessor sauvegardé.
    
    Args:
        filepath: Chemin du fichier (optionnel).
        version: Version à charger.
    
    Returns:
        Preprocessor chargé.
    """
    if filepath is None:
        filepath = MODELS_DIR / f"preprocessor_{version}.joblib"
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor non trouvé: {filepath}")
    
    preprocessor = joblib.load(filepath)
    
    logger.info(f"Preprocessor chargé: {filepath}")
    
    return preprocessor


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean_data
    
    logger.info("Test du module pipeline")
    
    try:
        # 1. Charger et nettoyer les données
        print(f"\n{'='*60}")
        print("1. CHARGEMENT DES DONNÉES")
        print(f"{'='*60}")
        
        df_raw = load_raw_data(nrows=5000)
        df_clean = clean_data(df_raw)
        
        print(f"Données nettoyées: {df_clean.shape}")
        
        # 2. Préparer le split
        print(f"\n{'='*60}")
        print("2. SPLIT TRAIN/VALIDATION")
        print(f"{'='*60}")
        
        X_train, X_val, y_train, y_val = prepare_data(
            df_clean, 
            test_size=0.2,
            stratify_col='cat_main'
        )
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        # 3. Créer le pipeline
        print(f"\n{'='*60}")
        print("3. CRÉATION DU PIPELINE")
        print(f"{'='*60}")
        
        X_train_features, preprocessor = create_full_pipeline(
            X_train, 
            y_train,
            tfidf_max_features=5000
        )
        
        print(f"Features train: {X_train_features.shape}")
        print(f"Nombre de features: {preprocessor.n_features_}")
        
        # 4. Transformer la validation
        print(f"\n{'='*60}")
        print("4. TRANSFORMATION VALIDATION")
        print(f"{'='*60}")
        
        X_val_features = preprocessor.transform(X_val)
        y_train_log = transform_target(y_train)
        y_val_log = transform_target(y_val)
        
        print(f"Features val: {X_val_features.shape}")
        print(f"y_train_log: min={y_train_log.min():.2f}, max={y_train_log.max():.2f}")
        
        # 5. Sauvegarder
        print(f"\n{'='*60}")
        print("5. SAUVEGARDE")
        print(f"{'='*60}")
        
        filepath = save_preprocessor(preprocessor, version="test")
        print(f"Sauvegardé: {filepath}")
        
        # 6. Recharger et tester
        print(f"\n{'='*60}")
        print("6. TEST DE RECHARGEMENT")
        print(f"{'='*60}")
        
        preprocessor_loaded = load_preprocessor(version="test")
        X_val_reload = preprocessor_loaded.transform(X_val)
        
        # Vérifier que c'est identique
        diff = np.abs(X_val_features.toarray() - X_val_reload.toarray()).max()
        print(f"Différence max après reload: {diff}")
        
        # Résumé
        print(f"\n{'='*60}")
        print("RÉSUMÉ DU PIPELINE")
        print(f"{'='*60}")
        print(f"  - TF-IDF features: {preprocessor.tfidf_max_features}")
        print(f"  - Total features: {preprocessor.n_features_}")
        print(f"  - Matrice sparse: {X_train_features.nnz:,} éléments non-nuls")
        print(f"  - Densité: {X_train_features.nnz / (X_train_features.shape[0] * X_train_features.shape[1]) * 100:.2f}%")
        
        # Aperçu des noms de features
        print(f"\n  Premiers noms de features:")
        for name in preprocessor.get_feature_names()[:10]:
            print(f"    - {name}")
        print(f"    ... ({len(preprocessor.get_feature_names()) - 10} autres)")
        
    except FileNotFoundError as e:
        logger.error(e)
        print("\n💡 Télécharge d'abord les données:")
        print("   python scripts/download_data.py")
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise