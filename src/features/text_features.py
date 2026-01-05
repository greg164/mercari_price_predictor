"""
Module: text_features.py
Description: Extraction de features textuelles (TF-IDF, mots-clés, statistiques)
Author: Greg
Date: 2025-01
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import re
import logging
from typing import Optional, List, Tuple, Union

# Third party
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# ================================================================================
# CONFIGURATION
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mots-clés indicateurs de qualité/état
QUALITY_KEYWORDS = {
    'excellent': ['new', 'brand new', 'bnwt', 'bnib', 'nwt', 'nib', 'mint', 
                  'perfect', 'excellent', 'pristine', 'unworn', 'unused'],
    'good': ['good', 'great', 'nice', 'clean', 'gently used', 'like new',
             'barely used', 'lightly used', 'worn once', 'used once'],
    'fair': ['fair', 'okay', 'ok', 'decent', 'average', 'normal wear',
             'some wear', 'minor', 'small'],
    'poor': ['worn', 'used', 'damaged', 'broken', 'stain', 'stained', 
             'hole', 'tear', 'rip', 'flaw', 'defect', 'scratch', 'scratched',
             'crack', 'cracked', 'missing']
}

# Mots-clés indicateurs de valeur
VALUE_KEYWORDS = {
    'luxury': ['authentic', 'genuine', 'real', 'original', 'designer',
               'luxury', 'premium', 'rare', 'limited edition', 'vintage',
               'collector', 'exclusive'],
    'budget': ['cheap', 'budget', 'affordable', 'bargain', 'deal',
               'discount', 'sale', 'clearance', 'bundle', 'lot']
}

# Stopwords simples (on peut utiliser NLTK pour plus complet)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us',
    'them', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if'
}


# ================================================================================
# FONCTIONS UTILITAIRES
# ================================================================================

def combine_text_columns(df: pd.DataFrame, 
                         columns: List[str] = ['name', 'item_description']) -> pd.Series:
    """
    Combine plusieurs colonnes texte en une seule.
    
    Args:
        df: DataFrame source.
        columns: Liste des colonnes à combiner.
    
    Returns:
        Series avec le texte combiné.
    """
    combined = df[columns[0]].fillna('').astype(str)
    
    for col in columns[1:]:
        combined = combined + ' ' + df[col].fillna('').astype(str)
    
    return combined.str.strip()


def preprocess_text(text: str) -> str:
    """
    Prétraitement basique du texte.
    
    Args:
        text: Texte à nettoyer.
    
    Returns:
        Texte nettoyé.
    """
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    
    # Supprimer les caractères spéciaux (garder lettres, chiffres, espaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ================================================================================
# FEATURES BASÉES SUR LES MOTS-CLÉS
# ================================================================================

def count_keywords(text: str, keywords: List[str]) -> int:
    """
    Compte le nombre de mots-clés présents dans un texte.
    
    Args:
        text: Texte à analyser.
        keywords: Liste de mots-clés à chercher.
    
    Returns:
        Nombre de mots-clés trouvés.
    """
    if pd.isna(text) or not text:
        return 0
    
    text = text.lower()
    count = 0
    
    for keyword in keywords:
        if keyword in text:
            count += 1
    
    return count


def extract_keyword_features(df: pd.DataFrame, 
                             text_col: str = 'item_description') -> pd.DataFrame:
    """
    Extrait des features basées sur les mots-clés de qualité et valeur.
    
    Args:
        df: DataFrame avec colonne texte.
        text_col: Nom de la colonne texte à analyser.
    
    Returns:
        DataFrame avec les nouvelles features.
    """
    df = df.copy()
    text = df[text_col].fillna('').str.lower()
    
    # Features de qualité
    df['kw_excellent'] = text.apply(lambda x: count_keywords(x, QUALITY_KEYWORDS['excellent']))
    df['kw_good'] = text.apply(lambda x: count_keywords(x, QUALITY_KEYWORDS['good']))
    df['kw_fair'] = text.apply(lambda x: count_keywords(x, QUALITY_KEYWORDS['fair']))
    df['kw_poor'] = text.apply(lambda x: count_keywords(x, QUALITY_KEYWORDS['poor']))
    
    # Features de valeur
    df['kw_luxury'] = text.apply(lambda x: count_keywords(x, VALUE_KEYWORDS['luxury']))
    df['kw_budget'] = text.apply(lambda x: count_keywords(x, VALUE_KEYWORDS['budget']))
    
    # Score de qualité global (-1 à 1)
    df['quality_score'] = (
        (df['kw_excellent'] * 1.0 + df['kw_good'] * 0.5) - 
        (df['kw_fair'] * 0.5 + df['kw_poor'] * 1.0)
    )
    
    # Indicateurs binaires
    df['has_quality_keywords'] = ((df['kw_excellent'] + df['kw_good'] + 
                                   df['kw_fair'] + df['kw_poor']) > 0).astype(int)
    df['has_luxury_keywords'] = (df['kw_luxury'] > 0).astype(int)
    
    logger.info(f"Features mots-clés extraites: {df['has_quality_keywords'].sum():,} "
                f"avec mots qualité, {df['has_luxury_keywords'].sum():,} avec mots luxe")
    
    return df


# ================================================================================
# TF-IDF VECTORIZER
# ================================================================================

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracteur de features textuelles combinant TF-IDF et features statistiques.
    
    Compatible avec sklearn Pipeline.
    """
    
    def __init__(self,
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 max_df: float = 0.95,
                 use_idf: bool = True,
                 sublinear_tf: bool = True,
                 text_columns: List[str] = None):
        """
        Initialise l'extracteur.
        
        Args:
            max_features: Nombre max de features TF-IDF.
            ngram_range: Plage de n-grams (ex: (1,2) pour unigrams et bigrams).
            min_df: Fréquence minimum pour inclure un terme.
            max_df: Fréquence maximum (filtre les termes trop fréquents).
            use_idf: Utiliser IDF.
            sublinear_tf: Utiliser log(1+tf) au lieu de tf.
            text_columns: Colonnes à combiner ['name', 'item_description'].
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.text_columns = text_columns or ['name', 'item_description']
        
        self.tfidf_vectorizer = None
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Entraîne le vectorizer TF-IDF.
        
        Args:
            X: DataFrame avec les colonnes texte.
            y: Ignoré (compatibilité sklearn).
        
        Returns:
            self
        """
        # Combiner les colonnes texte
        combined_text = combine_text_columns(X, self.text_columns)
        combined_text = combined_text.apply(preprocess_text)
        
        # Créer et entraîner le vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words=list(STOPWORDS)
        )
        
        self.tfidf_vectorizer.fit(combined_text)
        self.feature_names_ = self.tfidf_vectorizer.get_feature_names_out()
        
        logger.info(f"TF-IDF entraîné: {len(self.feature_names_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> csr_matrix:
        """
        Transforme le texte en matrice TF-IDF.
        
        Args:
            X: DataFrame avec les colonnes texte.
        
        Returns:
            Matrice sparse TF-IDF.
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("L'extracteur n'a pas été entraîné. Appeler fit() d'abord.")
        
        # Combiner et prétraiter
        combined_text = combine_text_columns(X, self.text_columns)
        combined_text = combined_text.apply(preprocess_text)
        
        # Transformer
        tfidf_matrix = self.tfidf_vectorizer.transform(combined_text)
        
        logger.info(f"Texte transformé: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> csr_matrix:
        """Fit et transform en une seule étape."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features TF-IDF."""
        if self.feature_names_ is None:
            return []
        return list(self.feature_names_)


# ================================================================================
# FONCTIONS DE HAUT NIVEAU
# ================================================================================

def create_tfidf_features(df: pd.DataFrame,
                          max_features: int = 10000,
                          ngram_range: Tuple[int, int] = (1, 2),
                          text_columns: List[str] = None,
                          fit: bool = True,
                          vectorizer: TextFeatureExtractor = None
                          ) -> Tuple[csr_matrix, TextFeatureExtractor]:
    """
    Crée les features TF-IDF à partir du DataFrame.
    
    Args:
        df: DataFrame source.
        max_features: Nombre max de features.
        ngram_range: Plage de n-grams.
        text_columns: Colonnes à utiliser.
        fit: Si True, entraîne un nouveau vectorizer.
        vectorizer: Vectorizer pré-entraîné (si fit=False).
    
    Returns:
        Tuple (matrice TF-IDF, vectorizer).
    """
    if text_columns is None:
        text_columns = ['name', 'item_description']
    
    if fit:
        vectorizer = TextFeatureExtractor(
            max_features=max_features,
            ngram_range=ngram_range,
            text_columns=text_columns
        )
        tfidf_matrix = vectorizer.fit_transform(df)
    else:
        if vectorizer is None:
            raise ValueError("vectorizer requis si fit=False")
        tfidf_matrix = vectorizer.transform(df)
    
    return tfidf_matrix, vectorizer


def extract_all_text_features(df: pd.DataFrame,
                              tfidf_max_features: int = 10000,
                              include_keywords: bool = True,
                              include_tfidf: bool = True
                              ) -> Tuple[pd.DataFrame, Optional[csr_matrix], Optional[TextFeatureExtractor]]:
    """
    Extrait toutes les features textuelles.
    
    Args:
        df: DataFrame source (doit contenir 'name' et 'item_description').
        tfidf_max_features: Nombre max de features TF-IDF.
        include_keywords: Inclure les features de mots-clés.
        include_tfidf: Inclure TF-IDF.
    
    Returns:
        Tuple (DataFrame avec features, matrice TF-IDF, vectorizer).
    """
    df_features = df.copy()
    tfidf_matrix = None
    vectorizer = None
    
    # Features mots-clés
    if include_keywords:
        df_features = extract_keyword_features(df_features, 'item_description')
    
    # TF-IDF
    if include_tfidf:
        tfidf_matrix, vectorizer = create_tfidf_features(
            df_features,
            max_features=tfidf_max_features
        )
    
    return df_features, tfidf_matrix, vectorizer


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
    
    logger.info("Test du module text_features")
    
    try:
        # Charger et nettoyer un échantillon
        df_raw = load_raw_data(nrows=5000)
        df_clean = clean_data(df_raw)
        
        # Extraire les features texte
        df_features, tfidf_matrix, vectorizer = extract_all_text_features(
            df_clean,
            tfidf_max_features=5000
        )
        
        # Afficher les résultats
        print(f"\n{'='*60}")
        print("FEATURES TEXTUELLES EXTRAITES")
        print(f"{'='*60}")
        
        print(f"\nDataFrame: {df_features.shape}")
        print(f"TF-IDF: {tfidf_matrix.shape}")
        
        # Features mots-clés
        keyword_cols = [c for c in df_features.columns if c.startswith('kw_') or 
                        c in ['quality_score', 'has_quality_keywords', 'has_luxury_keywords']]
        print(f"\nFeatures mots-clés: {keyword_cols}")
        
        # Stats mots-clés
        print(f"\nStatistiques mots-clés:")
        print(f"  - Avec mots qualité: {df_features['has_quality_keywords'].sum():,}")
        print(f"  - Avec mots luxe: {df_features['has_luxury_keywords'].sum():,}")
        print(f"  - Score qualité moyen: {df_features['quality_score'].mean():.3f}")
        
        # Top termes TF-IDF
        print(f"\nTop 20 termes TF-IDF:")
        feature_names = vectorizer.get_feature_names()
        print(f"  {feature_names[:20]}")
        
    except FileNotFoundError as e:
        logger.error(e)