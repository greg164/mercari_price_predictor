"""
Module: categorical_features.py
Description: Encodage des variables catégorielles (Label, Target, One-Hot)
Author: Greg
Date: 2025-01
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import logging
from typing import Optional, List, Dict, Tuple, Union

# Third party
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ================================================================================
# CONFIGURATION
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Valeur par défaut pour les catégories inconnues
UNKNOWN_VALUE = "unknown"


# ================================================================================
# LABEL ENCODING
# ================================================================================

class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label Encoder qui gère les valeurs inconnues (unseen) lors du transform.
    
    Les valeurs inconnues sont encodées avec une valeur spéciale (-1 ou max+1).
    """
    
    def __init__(self, unknown_value: int = -1):
        """
        Args:
            unknown_value: Valeur à attribuer aux catégories inconnues.
                          -1 ou 'max' (max_label + 1).
        """
        self.unknown_value = unknown_value
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.class_to_idx_ = None
    
    def fit(self, X: pd.Series, y=None):
        """
        Entraîne l'encodeur.
        
        Args:
            X: Series avec les valeurs à encoder.
            y: Ignoré.
        
        Returns:
            self
        """
        # Remplir les NaN
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str)
        
        self.encoder.fit(X_filled)
        self.classes_ = self.encoder.classes_
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        
        logger.info(f"LabelEncoder entraîné: {len(self.classes_)} classes")
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Transforme les valeurs en labels.
        
        Args:
            X: Series à transformer.
        
        Returns:
            Array d'entiers.
        """
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str)
        
        # Déterminer la valeur pour les inconnus
        if self.unknown_value == 'max':
            unk_val = len(self.classes_)
        else:
            unk_val = self.unknown_value
        
        # Encoder avec gestion des inconnus
        result = np.array([
            self.class_to_idx_.get(val, unk_val) for val in X_filled
        ])
        
        n_unknown = (result == unk_val).sum()
        if n_unknown > 0:
            logger.warning(f"  {n_unknown} valeurs inconnues encodées avec {unk_val}")
        
        return result
    
    def fit_transform(self, X: pd.Series, y=None) -> np.ndarray:
        """Fit et transform en une étape."""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse la transformation.
        
        Args:
            X: Array d'entiers.
        
        Returns:
            Array de strings.
        """
        unk_val = len(self.classes_) if self.unknown_value == 'max' else self.unknown_value
        
        result = []
        for val in X:
            if val == unk_val or val < 0 or val >= len(self.classes_):
                result.append(UNKNOWN_VALUE)
            else:
                result.append(self.classes_[val])
        
        return np.array(result)


# ================================================================================
# TARGET ENCODING
# ================================================================================

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoding : remplace chaque catégorie par la moyenne de la cible.
    
    Utile pour les variables avec beaucoup de modalités (ex: brand_name).
    Inclut un lissage pour éviter l'overfitting sur les catégories rares.
    """
    
    def __init__(self, 
                 smoothing: float = 10.0,
                 min_samples: int = 5,
                 fill_value: Optional[float] = None):
        """
        Args:
            smoothing: Paramètre de lissage (plus élevé = plus conservateur).
            min_samples: Nombre min d'échantillons pour utiliser la moyenne locale.
            fill_value: Valeur pour les catégories inconnues (défaut: moyenne globale).
        """
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.fill_value = fill_value
        
        self.encoding_map_ = None
        self.global_mean_ = None
    
    def fit(self, X: pd.Series, y: pd.Series):
        """
        Calcule les moyennes par catégorie.
        
        Args:
            X: Series avec les catégories.
            y: Series avec la cible (prix).
        
        Returns:
            self
        """
        # Remplir les NaN
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str)
        
        # Moyenne globale
        self.global_mean_ = y.mean()
        
        # Calcul des stats par catégorie
        df_temp = pd.DataFrame({'category': X_filled, 'target': y})
        stats = df_temp.groupby('category')['target'].agg(['mean', 'count'])
        
        # Lissage bayésien : weighted average entre moyenne locale et globale
        # smoothed_mean = (count * local_mean + smoothing * global_mean) / (count + smoothing)
        smoothed_means = (
            (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_) /
            (stats['count'] + self.smoothing)
        )
        
        # Pour les catégories avec trop peu d'échantillons, utiliser la moyenne globale
        smoothed_means[stats['count'] < self.min_samples] = self.global_mean_
        
        self.encoding_map_ = smoothed_means.to_dict()
        
        logger.info(f"TargetEncoder entraîné: {len(self.encoding_map_)} catégories, "
                    f"moyenne globale={self.global_mean_:.2f}")
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Applique le target encoding.
        
        Args:
            X: Series à transformer.
        
        Returns:
            Array de floats.
        """
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str)
        
        # Valeur pour les inconnus
        default = self.fill_value if self.fill_value is not None else self.global_mean_
        
        result = X_filled.map(self.encoding_map_).fillna(default).values
        
        n_unknown = X_filled.map(lambda x: x not in self.encoding_map_).sum()
        if n_unknown > 0:
            logger.warning(f"  {n_unknown} valeurs inconnues encodées avec {default:.2f}")
        
        return result
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> np.ndarray:
        """Fit et transform en une étape."""
        return self.fit(X, y).transform(X)


# ================================================================================
# ONE-HOT ENCODING
# ================================================================================

class SafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-Hot Encoder qui gère les valeurs inconnues.
    
    Recommandé pour les variables avec peu de modalités (< 15).
    """
    
    def __init__(self, 
                 sparse_output: bool = False,
                 handle_unknown: str = 'ignore'):
        """
        Args:
            sparse_output: Si True, retourne une matrice sparse.
            handle_unknown: 'ignore' ou 'error'.
        """
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        
        self.encoder = None
        self.categories_ = None
        self.feature_names_ = None
    
    def fit(self, X: pd.Series, y=None):
        """
        Entraîne l'encodeur.
        
        Args:
            X: Series avec les catégories.
            y: Ignoré.
        
        Returns:
            self
        """
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str).values.reshape(-1, 1)
        
        self.encoder = OneHotEncoder(
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown
        )
        
        self.encoder.fit(X_filled)
        self.categories_ = self.encoder.categories_[0]
        self.feature_names_ = [f"{X.name}_{cat}" for cat in self.categories_]
        
        logger.info(f"OneHotEncoder entraîné: {len(self.categories_)} catégories")
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Applique le one-hot encoding.
        
        Args:
            X: Series à transformer.
        
        Returns:
            Array 2D (n_samples, n_categories).
        """
        X_filled = X.fillna(UNKNOWN_VALUE).astype(str).values.reshape(-1, 1)
        
        return self.encoder.transform(X_filled)
    
    def fit_transform(self, X: pd.Series, y=None) -> np.ndarray:
        """Fit et transform en une étape."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features."""
        return self.feature_names_ or []


# ================================================================================
# ENCODEUR MULTI-COLONNES
# ================================================================================

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodeur qui applique différentes stratégies selon les colonnes.
    
    - One-hot pour les colonnes avec peu de modalités
    - Target encoding pour les colonnes avec beaucoup de modalités
    - Label encoding optionnel
    """
    
    def __init__(self,
                 onehot_columns: List[str] = None,
                 target_columns: List[str] = None,
                 label_columns: List[str] = None,
                 passthrough_columns: List[str] = None,
                 target_smoothing: float = 10.0):
        """
        Args:
            onehot_columns: Colonnes pour one-hot encoding.
            target_columns: Colonnes pour target encoding.
            label_columns: Colonnes pour label encoding.
            passthrough_columns: Colonnes numériques à garder telles quelles.
            target_smoothing: Paramètre de lissage pour target encoding.
        """
        self.onehot_columns = onehot_columns or []
        self.target_columns = target_columns or []
        self.label_columns = label_columns or []
        self.passthrough_columns = passthrough_columns or []
        self.target_smoothing = target_smoothing
        
        self.encoders_ = {}
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Entraîne tous les encodeurs.
        
        Args:
            X: DataFrame avec les colonnes à encoder.
            y: Series cible (requis pour target encoding).
        
        Returns:
            self
        """
        self.encoders_ = {}
        self.feature_names_ = []
        
        # One-hot encoding
        for col in self.onehot_columns:
            if col in X.columns:
                enc = SafeOneHotEncoder()
                enc.fit(X[col])
                self.encoders_[col] = ('onehot', enc)
                self.feature_names_.extend(enc.get_feature_names())
                logger.info(f"  {col}: OneHot ({len(enc.categories_)} cat)")
        
        # Target encoding
        if y is not None:
            for col in self.target_columns:
                if col in X.columns:
                    enc = TargetEncoder(smoothing=self.target_smoothing)
                    enc.fit(X[col], y)
                    self.encoders_[col] = ('target', enc)
                    self.feature_names_.append(f"{col}_encoded")
                    logger.info(f"  {col}: TargetEnc ({len(enc.encoding_map_)} cat)")
        
        # Label encoding
        for col in self.label_columns:
            if col in X.columns:
                enc = SafeLabelEncoder()
                enc.fit(X[col])
                self.encoders_[col] = ('label', enc)
                self.feature_names_.append(f"{col}_label")
                logger.info(f"  {col}: LabelEnc ({len(enc.classes_)} cat)")
        
        # Passthrough
        for col in self.passthrough_columns:
            if col in X.columns:
                self.encoders_[col] = ('passthrough', None)
                self.feature_names_.append(col)
        
        logger.info(f"CategoricalEncoder entraîné: {len(self.feature_names_)} features totales")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applique tous les encodages.
        
        Args:
            X: DataFrame à transformer.
        
        Returns:
            Array 2D avec toutes les features encodées.
        """
        result_parts = []
        
        for col, (enc_type, encoder) in self.encoders_.items():
            if col not in X.columns:
                logger.warning(f"Colonne {col} manquante")
                continue
            
            if enc_type == 'onehot':
                encoded = encoder.transform(X[col])
                result_parts.append(encoded)
            
            elif enc_type == 'target':
                encoded = encoder.transform(X[col]).reshape(-1, 1)
                result_parts.append(encoded)
            
            elif enc_type == 'label':
                encoded = encoder.transform(X[col]).reshape(-1, 1)
                result_parts.append(encoded)
            
            elif enc_type == 'passthrough':
                encoded = X[col].values.reshape(-1, 1)
                result_parts.append(encoded)
        
        if not result_parts:
            raise ValueError("Aucune colonne à encoder")
        
        return np.hstack(result_parts)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit et transform en une étape."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms de toutes les features."""
        return self.feature_names_


# ================================================================================
# FONCTIONS DE HAUT NIVEAU
# ================================================================================

def create_categorical_features(
    df: pd.DataFrame,
    y: pd.Series = None,
    onehot_columns: List[str] = None,
    target_columns: List[str] = None,
    passthrough_columns: List[str] = None
) -> Tuple[np.ndarray, CategoricalEncoder]:
    """
    Crée les features catégorielles avec la stratégie par défaut du projet.
    
    Stratégie par défaut:
    - One-hot: cat_main (catégorie principale, ~11 valeurs)
    - Target encoding: brand_name, cat_sub1, cat_sub2 (beaucoup de valeurs)
    - Passthrough: item_condition_id, shipping (déjà numériques)
    
    Args:
        df: DataFrame source.
        y: Series cible (prix, requis pour target encoding).
        onehot_columns: Override des colonnes one-hot.
        target_columns: Override des colonnes target.
        passthrough_columns: Override des colonnes passthrough.
    
    Returns:
        Tuple (array encodé, encoder).
    """
    # Valeurs par défaut
    if onehot_columns is None:
        onehot_columns = ['cat_main']
    
    if target_columns is None:
        target_columns = ['brand_name', 'cat_sub1', 'cat_sub2']
    
    if passthrough_columns is None:
        passthrough_columns = ['item_condition_id', 'shipping']
    
    encoder = CategoricalEncoder(
        onehot_columns=onehot_columns,
        target_columns=target_columns,
        passthrough_columns=passthrough_columns
    )
    
    encoded = encoder.fit_transform(df, y)
    
    return encoded, encoder


def get_category_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Retourne des statistiques sur les colonnes catégorielles.
    
    Args:
        df: DataFrame à analyser.
    
    Returns:
        Dict avec les stats par colonne.
    """
    cat_columns = ['cat_main', 'cat_sub1', 'cat_sub2', 'brand_name', 
                   'item_condition_id', 'shipping']
    
    stats = {}
    
    for col in cat_columns:
        if col in df.columns:
            stats[col] = {
                'n_unique': df[col].nunique(),
                'n_missing': df[col].isnull().sum(),
                'top_5': df[col].value_counts().head(5).to_dict()
            }
    
    return stats


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Ajouter le path pour les imports
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean_data
    
    logger.info("Test du module categorical_features")
    
    try:
        # Charger et nettoyer un échantillon
        df_raw = load_raw_data(nrows=5000)
        df_clean = clean_data(df_raw)
        
        # Cible (log price)
        y = np.log1p(df_clean['price'])
        
        # Stats des catégories
        print(f"\n{'='*60}")
        print("STATISTIQUES DES CATÉGORIES")
        print(f"{'='*60}")
        
        stats = get_category_stats(df_clean)
        for col, col_stats in stats.items():
            print(f"\n{col}:")
            print(f"  - Unique: {col_stats['n_unique']}")
            print(f"  - Top 3: {list(col_stats['top_5'].keys())[:3]}")
        
        # Encoder
        print(f"\n{'='*60}")
        print("ENCODAGE DES CATÉGORIES")
        print(f"{'='*60}")
        
        encoded, encoder = create_categorical_features(df_clean, y)
        
        print(f"\nRésultat: {encoded.shape}")
        print(f"Features: {encoder.get_feature_names()[:10]}...")
        
        # Test sur les premières lignes
        print(f"\nAperçu (5 premières lignes, 8 premières features):")
        print(encoded[:5, :8])
        
    except FileNotFoundError as e:
        logger.error(e)