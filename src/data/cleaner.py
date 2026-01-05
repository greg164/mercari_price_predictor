"""
Module: cleaner.py
Description: Nettoyage et préparation des données Mercari
Author: Greg
Date: 2025-01
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import re
import logging
from typing import Optional, Tuple

# Third party
import pandas as pd
import numpy as np

# ================================================================================
# CONFIGURATION
# ================================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_MIN_PRICE = 1
DEFAULT_MAX_PRICE = 2000
UNKNOWN_BRAND = "unknown"
UNKNOWN_CATEGORY = "unknown"


# ================================================================================
# FONCTIONS DE NETTOYAGE
# ================================================================================

def remove_price_outliers(
    df: pd.DataFrame,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE
) -> pd.DataFrame:
    """
    Supprime les lignes avec des prix aberrants (0 ou trop élevés).
    
    Args:
        df: DataFrame avec une colonne 'price'.
        min_price: Prix minimum acceptable.
        max_price: Prix maximum acceptable.
    
    Returns:
        DataFrame filtré.
    """
    n_before = len(df)
    
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)].copy()
    
    n_removed = n_before - len(df)
    logger.info(f"Prix outliers supprimés: {n_removed:,} lignes ({n_removed/n_before*100:.2f}%)")
    
    return df


def fill_missing_brands(df: pd.DataFrame, fill_value: str = UNKNOWN_BRAND) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes de brand_name.
    
    Args:
        df: DataFrame avec une colonne 'brand_name'.
        fill_value: Valeur de remplacement.
    
    Returns:
        DataFrame avec les marques manquantes remplacées.
    """
    n_missing = df['brand_name'].isnull().sum()
    
    df = df.copy()
    df['brand_name'] = df['brand_name'].fillna(fill_value)
    
    logger.info(f"Marques manquantes remplacées: {n_missing:,} ({n_missing/len(df)*100:.2f}%)")
    
    return df


def split_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sépare category_name en 3 colonnes : cat_main, cat_sub1, cat_sub2.
    
    Exemple: "Electronics/Computers/Laptops" -> ["Electronics", "Computers", "Laptops"]
    
    Args:
        df: DataFrame avec une colonne 'category_name'.
    
    Returns:
        DataFrame avec 3 nouvelles colonnes de catégorie.
    """
    df = df.copy()
    
    # Remplacer les valeurs manquantes
    df['category_name'] = df['category_name'].fillna(f"{UNKNOWN_CATEGORY}/{UNKNOWN_CATEGORY}/{UNKNOWN_CATEGORY}")
    
    # Séparer en 3 colonnes
    categories = df['category_name'].str.split('/', n=2, expand=True)
    
    df['cat_main'] = categories[0].fillna(UNKNOWN_CATEGORY)
    df['cat_sub1'] = categories[1].fillna(UNKNOWN_CATEGORY) if 1 in categories.columns else UNKNOWN_CATEGORY
    df['cat_sub2'] = categories[2].fillna(UNKNOWN_CATEGORY) if 2 in categories.columns else UNKNOWN_CATEGORY
    
    logger.info(f"Catégories séparées: {df['cat_main'].nunique()} principales, "
                f"{df['cat_sub1'].nunique()} sous-cat1, {df['cat_sub2'].nunique()} sous-cat2")
    
    return df


def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la colonne item_description.
    
    - Remplace les valeurs manquantes par une chaîne vide
    - Supprime "[rm]" (prix censurés par Mercari)
    - Supprime les caractères spéciaux excessifs
    - Convertit en minuscules
    
    Args:
        df: DataFrame avec une colonne 'item_description'.
    
    Returns:
        DataFrame avec les descriptions nettoyées.
    """
    df = df.copy()
    
    # Remplacer les valeurs manquantes et "No description yet"
    df['item_description'] = df['item_description'].fillna('')
    df['item_description'] = df['item_description'].replace('No description yet', '')
    
    # Supprimer [rm] (prix censurés)
    df['item_description'] = df['item_description'].str.replace(r'\[rm\]', '', regex=True)
    
    # Convertir en minuscules
    df['item_description'] = df['item_description'].str.lower()
    
    # Supprimer les espaces multiples
    df['item_description'] = df['item_description'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    logger.info("Descriptions nettoyées")
    
    return df


def clean_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la colonne name (titre de l'annonce).
    
    Args:
        df: DataFrame avec une colonne 'name'.
    
    Returns:
        DataFrame avec les titres nettoyés.
    """
    df = df.copy()
    
    # Remplacer les valeurs manquantes
    df['name'] = df['name'].fillna('')
    
    # Supprimer [rm]
    df['name'] = df['name'].str.replace(r'\[rm\]', '', regex=True)
    
    # Convertir en minuscules
    df['name'] = df['name'].str.lower()
    
    # Supprimer les espaces multiples
    df['name'] = df['name'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    logger.info("Titres nettoyés")
    
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features basées sur le texte.
    
    - name_len: nombre de mots dans le titre
    - desc_len: nombre de mots dans la description
    - has_description: 1 si description présente, 0 sinon
    
    Args:
        df: DataFrame avec 'name' et 'item_description'.
    
    Returns:
        DataFrame avec les nouvelles features.
    """
    df = df.copy()
    
    # Longueur du titre (nombre de mots)
    df['name_len'] = df['name'].str.split().str.len().fillna(0).astype(int)
    
    # Longueur de la description (nombre de mots)
    df['desc_len'] = df['item_description'].str.split().str.len().fillna(0).astype(int)
    
    # Indicateur de présence de description
    df['has_description'] = (df['desc_len'] > 0).astype(int)
    
    logger.info(f"Features texte ajoutées: name_len (moy={df['name_len'].mean():.1f}), "
                f"desc_len (moy={df['desc_len'].mean():.1f})")
    
    return df


def add_brand_features(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Ajoute un indicateur pour les marques populaires.
    
    Args:
        df: DataFrame avec 'brand_name'.
        top_n: Nombre de marques à considérer comme "populaires".
    
    Returns:
        DataFrame avec la feature 'is_top_brand'.
    """
    df = df.copy()
    
    # Top N marques (hors unknown)
    brand_counts = df[df['brand_name'] != UNKNOWN_BRAND]['brand_name'].value_counts()
    top_brands = set(brand_counts.head(top_n).index)
    
    df['is_top_brand'] = df['brand_name'].isin(top_brands).astype(int)
    
    logger.info(f"Feature marque ajoutée: {df['is_top_brand'].sum():,} produits avec top brand")
    
    return df


# ================================================================================
# PIPELINE COMPLET
# ================================================================================

def clean_data(
    df: pd.DataFrame,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
    add_features: bool = True
) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage des données.
    
    Args:
        df: DataFrame brut.
        min_price: Prix minimum acceptable.
        max_price: Prix maximum acceptable.
        add_features: Si True, ajoute les features dérivées.
    
    Returns:
        DataFrame nettoyé.
    """
    logger.info(f"Début du nettoyage - {len(df):,} lignes")
    
    # 1. Filtrer les prix aberrants
    df = remove_price_outliers(df, min_price, max_price)
    
    # 2. Remplir les marques manquantes
    df = fill_missing_brands(df)
    
    # 3. Séparer les catégories
    df = split_category(df)
    
    # 4. Nettoyer le texte
    df = clean_name(df)
    df = clean_description(df)
    
    # 5. Ajouter les features (optionnel)
    if add_features:
        df = add_text_features(df)
        df = add_brand_features(df)
    
    logger.info(f"Nettoyage terminé - {len(df):,} lignes, {len(df.columns)} colonnes")
    
    return df


def get_clean_stats(df: pd.DataFrame) -> dict:
    """
    Retourne des statistiques sur les données nettoyées.
    
    Args:
        df: DataFrame nettoyé.
    
    Returns:
        Dictionnaire de statistiques.
    """
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'price_stats': {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
            'median': df['price'].median()
        },
        'n_categories_main': df['cat_main'].nunique() if 'cat_main' in df.columns else None,
        'n_brands': df['brand_name'].nunique(),
        'pct_unknown_brand': (df['brand_name'] == UNKNOWN_BRAND).mean() * 100,
        'avg_name_len': df['name_len'].mean() if 'name_len' in df.columns else None,
        'avg_desc_len': df['desc_len'].mean() if 'desc_len' in df.columns else None
    }
    
    return stats


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    from src.data.loader import load_raw_data
    
    logger.info("Test du module cleaner")
    
    try:
        # Charger un échantillon
        df_raw = load_raw_data(nrows=10000)
        
        # Nettoyer
        df_clean = clean_data(df_raw)
        
        # Afficher les stats
        stats = get_clean_stats(df_clean)
        
        print(f"\n{'='*50}")
        print("STATISTIQUES APRÈS NETTOYAGE")
        print(f"{'='*50}")
        print(f"Lignes: {stats['n_rows']:,}")
        print(f"Colonnes: {stats['n_cols']}")
        print(f"\nPrix:")
        print(f"  - Min: ${stats['price_stats']['min']:.2f}")
        print(f"  - Max: ${stats['price_stats']['max']:.2f}")
        print(f"  - Moyenne: ${stats['price_stats']['mean']:.2f}")
        print(f"  - Médiane: ${stats['price_stats']['median']:.2f}")
        print(f"\nCatégories principales: {stats['n_categories_main']}")
        print(f"Marques uniques: {stats['n_brands']}")
        print(f"Marques inconnues: {stats['pct_unknown_brand']:.1f}%")
        print(f"\nLongueur moyenne titre: {stats['avg_name_len']:.1f} mots")
        print(f"Longueur moyenne description: {stats['avg_desc_len']:.1f} mots")
        
        print(f"\n{'='*50}")
        print("APERÇU DES DONNÉES")
        print(f"{'='*50}")
        print(df_clean[['name', 'cat_main', 'brand_name', 'price', 'name_len', 'desc_len']].head(10))
        
    except FileNotFoundError as e:
        logger.error(e)