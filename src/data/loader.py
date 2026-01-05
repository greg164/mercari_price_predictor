"""
Module: loader.py
Description: Chargement des données Mercari depuis les fichiers TSV
Author: Greg
Date: 2025-01
"""

# ================================================================================
# IMPORTS
# ================================================================================

# Standard library
import logging
from pathlib import Path
from typing import Optional

# Third party
import pandas as pd

# ================================================================================
# CONFIGURATION
# ================================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins par défaut
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ================================================================================
# FONCTIONS
# ================================================================================

def load_raw_data(filepath: Optional[Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Charge les données brutes depuis le fichier train.tsv.
    
    Args:
        filepath: Chemin vers le fichier TSV. Si None, utilise le chemin par défaut.
        nrows: Nombre de lignes à charger. Si None, charge tout le fichier.
    
    Returns:
        DataFrame avec les données brutes.
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if filepath is None:
        filepath = DATA_RAW_DIR / "train.tsv"
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {filepath}\n"
            f"Télécharge les données avec: python scripts/download_data.py"
        )
    
    logger.info(f"Chargement des données depuis {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep='\t',
        nrows=nrows,
        dtype={
            'train_id': 'int32',
            'item_condition_id': 'int8',
            'shipping': 'int8'
        }
    )
    
    logger.info(f"Données chargées: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    return df


def load_processed_data(filename: str = "train_clean.csv") -> pd.DataFrame:
    """
    Charge les données nettoyées depuis le dossier processed.
    
    Args:
        filename: Nom du fichier à charger.
    
    Returns:
        DataFrame avec les données nettoyées.
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = DATA_PROCESSED_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {filepath}\n"
            f"Lance d'abord le preprocessing pour générer ce fichier."
        )
    
    logger.info(f"Chargement des données depuis {filepath}")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"Données chargées: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    return df


def get_sample(n: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Charge un échantillon des données pour les tests et l'exploration.
    
    Args:
        n: Nombre de lignes à échantillonner.
        random_state: Seed pour la reproductibilité.
    
    Returns:
        DataFrame échantillonné.
    """
    df = load_raw_data()
    
    if n >= len(df):
        logger.warning(f"n ({n}) >= taille du dataset ({len(df)}), retourne tout le dataset")
        return df
    
    sample = df.sample(n=n, random_state=random_state)
    logger.info(f"Échantillon créé: {len(sample):,} lignes")
    
    return sample


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Retourne des informations sur le DataFrame.
    
    Args:
        df: DataFrame à analyser.
    
    Returns:
        Dictionnaire avec les informations.
    """
    info = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    return info


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    # Test du module
    logger.info("Test du module loader")
    
    try:
        # Charger un échantillon
        df = load_raw_data(nrows=1000)
        
        # Afficher les infos
        info = get_data_info(df)
        print(f"\n{'='*50}")
        print("INFORMATIONS SUR LES DONNÉES")
        print(f"{'='*50}")
        print(f"Lignes: {info['n_rows']:,}")
        print(f"Colonnes: {info['n_cols']}")
        print(f"Mémoire: {info['memory_mb']:.2f} MB")
        print(f"\nColonnes: {info['columns']}")
        print(f"\nValeurs manquantes (%):")
        for col, pct in info['missing_pct'].items():
            if pct > 0:
                print(f"  - {col}: {pct}%")
        
        print(f"\n{'='*50}")
        print("APERÇU DES DONNÉES")
        print(f"{'='*50}")
        print(df.head())
        
    except FileNotFoundError as e:
        logger.error(e)