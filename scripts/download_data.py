"""
Module: download_data.py
Description: Téléchargement du dataset Mercari depuis Kaggle
Author: Greg
Date: 2025-01
"""

# ================================================================================
# IMPORTS
# ================================================================================

import os
import sys
import logging
import zipfile
import argparse
from pathlib import Path

# ================================================================================
# CONFIGURATION
# ================================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Kaggle competition
COMPETITION_NAME = "mercari-price-suggestion-challenge"


# ================================================================================
# FONCTIONS
# ================================================================================

def check_kaggle_credentials() -> bool:
    """
    Vérifie que les credentials Kaggle sont configurés.
    
    Le fichier kaggle.json doit être dans :
    - Windows: C:/Users/<username>/.kaggle/kaggle.json
    - Linux/Mac: ~/.kaggle/kaggle.json
    
    Returns:
        bool: True si les credentials existent
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error(
            f"Fichier kaggle.json non trouvé dans {kaggle_dir}\n"
            "Pour configurer Kaggle:\n"
            "1. Va sur https://www.kaggle.com/settings\n"
            "2. Section 'API' > 'Create New Token'\n"
            "3. Place le fichier kaggle.json dans ~/.kaggle/\n"
            "4. Assure-toi que les permissions sont correctes (chmod 600 sur Linux/Mac)"
        )
        return False
    
    logger.info("Credentials Kaggle trouvés")
    return True


def accept_competition_rules():
    """
    Rappelle à l'utilisateur d'accepter les règles de la compétition.
    """
    logger.warning(
        f"Assure-toi d'avoir accepté les règles de la compétition sur:\n"
        f"https://www.kaggle.com/c/{COMPETITION_NAME}/rules"
    )


def download_from_kaggle() -> bool:
    """
    Télécharge le dataset depuis Kaggle.
    
    Returns:
        bool: True si le téléchargement a réussi
    """
    try:
        # Import kaggle ici pour éviter l'erreur si non installé
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authentification
        api = KaggleApi()
        api.authenticate()
        logger.info("Authentification Kaggle réussie")
        
        # Créer le dossier si nécessaire
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        # Télécharger les données
        logger.info(f"Téléchargement du dataset {COMPETITION_NAME}...")
        logger.info("Cela peut prendre quelques minutes...")
        
        api.competition_download_files(
            competition=COMPETITION_NAME,
            path=DATA_RAW_DIR,
            quiet=False
        )
        
        logger.info(f"Téléchargement terminé dans {DATA_RAW_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return False


def extract_zip_files():
    """
    Extrait les fichiers ZIP téléchargés.
    """
    zip_files = list(DATA_RAW_DIR.glob("*.zip"))
    
    if not zip_files:
        logger.info("Aucun fichier ZIP à extraire")
        return
    
    for zip_file in zip_files:
        logger.info(f"Extraction de {zip_file.name}...")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW_DIR)
            
            # Supprimer le ZIP après extraction
            zip_file.unlink()
            logger.info(f"Extraction terminée, ZIP supprimé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de {zip_file}: {e}")


def verify_download() -> bool:
    """
    Vérifie que les fichiers attendus sont présents.
    
    Returns:
        bool: True si les fichiers sont présents
    """
    expected_files = ["train.tsv"]  # test.tsv n'est pas nécessaire pour l'entraînement
    
    missing = []
    for filename in expected_files:
        filepath = DATA_RAW_DIR / filename
        if not filepath.exists():
            missing.append(filename)
        else:
            # Afficher la taille du fichier
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {filename} ({size_mb:.1f} MB)")
    
    if missing:
        logger.error(f"Fichiers manquants: {missing}")
        return False
    
    logger.info("Tous les fichiers sont présents")
    return True


def download_manual_instructions():
    """
    Affiche les instructions pour un téléchargement manuel.
    """
    print("\n" + "="*60)
    print("TÉLÉCHARGEMENT MANUEL")
    print("="*60)
    print(f"""
Si le téléchargement automatique échoue, tu peux télécharger manuellement:

1. Va sur: https://www.kaggle.com/c/{COMPETITION_NAME}/data

2. Clique sur "Download All"

3. Extrais le ZIP dans: {DATA_RAW_DIR}

4. Vérifie que tu as bien le fichier train.tsv
""")
    print("="*60 + "\n")


# ================================================================================
# MAIN
# ================================================================================

def main():
    """
    Fonction principale de téléchargement.
    """
    parser = argparse.ArgumentParser(
        description="Télécharge le dataset Mercari depuis Kaggle"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Affiche les instructions de téléchargement manuel"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Vérifie seulement si les fichiers sont présents"
    )
    
    args = parser.parse_args()
    
    # Instructions manuelles
    if args.manual:
        download_manual_instructions()
        return
    
    # Vérification seule
    if args.verify_only:
        if verify_download():
            logger.info("Dataset prêt à l'utilisation")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Vérifier si déjà téléchargé
    if verify_download():
        logger.info("Dataset déjà présent, pas besoin de retélécharger")
        logger.info("Utilise --verify-only pour juste vérifier les fichiers")
        return
    
    # Vérifier les credentials
    if not check_kaggle_credentials():
        download_manual_instructions()
        sys.exit(1)
    
    # Rappel des règles
    accept_competition_rules()
    
    # Télécharger
    if not download_from_kaggle():
        download_manual_instructions()
        sys.exit(1)
    
    # Extraire les ZIP
    extract_zip_files()
    
    # Vérifier
    if verify_download():
        logger.info("✓ Dataset téléchargé et prêt à l'utilisation")
    else:
        logger.error("✗ Problème avec le téléchargement")
        sys.exit(1)


if __name__ == "__main__":
    main()