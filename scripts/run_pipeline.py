"""
Module: run_pipeline.py
Description: Orchestration du pipeline complet (chargement, nettoyage, tests)
Author: Greg
Date: 2025-01

Usage:
    python scripts/run_pipeline.py                    # Pipeline complet
    python scripts/run_pipeline.py --test             # Tests uniquement
    python scripts/run_pipeline.py --step load        # Étape spécifique
    python scripts/run_pipeline.py --nrows 10000      # Limiter les données
"""

# ================================================================================
# IMPORTS
# ================================================================================

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Ajouter le dossier racine au path pour les imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data, get_data_info
from src.data.cleaner import clean_data, get_clean_stats

# ================================================================================
# CONFIGURATION
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ================================================================================
# FONCTIONS DE TEST
# ================================================================================

def test_loader() -> bool:
    """Test du module loader."""
    print("\n" + "="*60)
    print("TEST: src/data/loader.py")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Chargement avec nrows
    tests_total += 1
    try:
        df = load_raw_data(nrows=100)
        assert len(df) == 100, f"Attendu 100 lignes, obtenu {len(df)}"
        assert 'price' in df.columns, "Colonne 'price' manquante"
        assert 'name' in df.columns, "Colonne 'name' manquante"
        print("  ✓ Test chargement (nrows=100)")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test chargement: {e}")
    
    # Test 2: Colonnes attendues
    tests_total += 1
    try:
        expected_cols = ['train_id', 'name', 'item_condition_id', 'category_name', 
                         'brand_name', 'price', 'shipping', 'item_description']
        df = load_raw_data(nrows=10)
        for col in expected_cols:
            assert col in df.columns, f"Colonne '{col}' manquante"
        print("  ✓ Test colonnes attendues")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test colonnes: {e}")
    
    # Test 3: get_data_info
    tests_total += 1
    try:
        df = load_raw_data(nrows=100)
        info = get_data_info(df)
        assert info['n_rows'] == 100
        assert 'missing_values' in info
        print("  ✓ Test get_data_info()")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test get_data_info: {e}")
    
    print(f"\n  Résultat: {tests_passed}/{tests_total} tests passés")
    return tests_passed == tests_total


def test_cleaner() -> bool:
    """Test du module cleaner."""
    print("\n" + "="*60)
    print("TEST: src/data/cleaner.py")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Charger des données pour les tests
    df_raw = load_raw_data(nrows=1000)
    
    # Test 1: Pipeline complet
    tests_total += 1
    try:
        df_clean = clean_data(df_raw)
        assert len(df_clean) <= len(df_raw), "Le nettoyage ne devrait pas ajouter de lignes"
        assert 'cat_main' in df_clean.columns, "Colonne 'cat_main' manquante"
        print("  ✓ Test pipeline clean_data()")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test pipeline: {e}")
    
    # Test 2: Filtrage des prix
    tests_total += 1
    try:
        df_clean = clean_data(df_raw, min_price=5, max_price=500)
        assert df_clean['price'].min() >= 5, "Prix min non respecté"
        assert df_clean['price'].max() <= 500, "Prix max non respecté"
        print("  ✓ Test filtrage prix")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test filtrage prix: {e}")
    
    # Test 3: Pas de valeurs manquantes pour brand_name
    tests_total += 1
    try:
        df_clean = clean_data(df_raw)
        assert df_clean['brand_name'].isnull().sum() == 0, "brand_name a encore des nulls"
        print("  ✓ Test remplissage brand_name")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test brand_name: {e}")
    
    # Test 4: Séparation des catégories
    tests_total += 1
    try:
        df_clean = clean_data(df_raw)
        assert 'cat_main' in df_clean.columns
        assert 'cat_sub1' in df_clean.columns
        assert 'cat_sub2' in df_clean.columns
        print("  ✓ Test séparation catégories")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test catégories: {e}")
    
    # Test 5: Features texte ajoutées
    tests_total += 1
    try:
        df_clean = clean_data(df_raw, add_features=True)
        assert 'name_len' in df_clean.columns
        assert 'desc_len' in df_clean.columns
        assert 'has_description' in df_clean.columns
        assert 'is_top_brand' in df_clean.columns
        print("  ✓ Test features texte")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test features: {e}")
    
    # Test 6: Nettoyage [rm]
    tests_total += 1
    try:
        df_clean = clean_data(df_raw)
        has_rm = df_clean['item_description'].str.contains(r'\[rm\]', regex=True).any()
        assert not has_rm, "[rm] encore présent dans les descriptions"
        print("  ✓ Test suppression [rm]")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test [rm]: {e}")
    
    print(f"\n  Résultat: {tests_passed}/{tests_total} tests passés")
    return tests_passed == tests_total


def run_tests() -> bool:
    """Lance tous les tests."""
    print("\n" + "="*60)
    print("LANCEMENT DES TESTS")
    print("="*60)
    
    results = []
    
    try:
        results.append(("loader", test_loader()))
    except FileNotFoundError as e:
        print(f"\n⚠ Impossible de tester loader: {e}")
        results.append(("loader", False))
    
    try:
        results.append(("cleaner", test_cleaner()))
    except FileNotFoundError as e:
        print(f"\n⚠ Impossible de tester cleaner: {e}")
        results.append(("cleaner", False))
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    all_passed = True
    for module, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {module}")
        if not passed:
            all_passed = False
    
    return all_passed


# ================================================================================
# FONCTIONS DU PIPELINE
# ================================================================================

def step_load(nrows: int = None) -> dict:
    """Étape 1: Chargement des données."""
    print("\n" + "="*60)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("="*60)
    
    df = load_raw_data(nrows=nrows)
    info = get_data_info(df)
    
    print(f"\n  Lignes chargées: {info['n_rows']:,}")
    print(f"  Colonnes: {info['n_cols']}")
    print(f"  Mémoire: {info['memory_mb']:.2f} MB")
    print(f"\n  Valeurs manquantes:")
    for col, pct in info['missing_pct'].items():
        if pct > 0:
            print(f"    - {col}: {pct}%")
    
    return {'df': df, 'info': info}


def step_clean(df, min_price: float = 1, max_price: float = 2000) -> dict:
    """Étape 2: Nettoyage des données."""
    print("\n" + "="*60)
    print("ÉTAPE 2: NETTOYAGE DES DONNÉES")
    print("="*60)
    
    df_clean = clean_data(df, min_price=min_price, max_price=max_price)
    stats = get_clean_stats(df_clean)
    
    print(f"\n  Lignes après nettoyage: {stats['n_rows']:,}")
    print(f"  Colonnes: {stats['n_cols']}")
    print(f"\n  Prix:")
    print(f"    - Moyenne: ${stats['price_stats']['mean']:.2f}")
    print(f"    - Médiane: ${stats['price_stats']['median']:.2f}")
    print(f"\n  Catégories principales: {stats['n_categories_main']}")
    print(f"  Marques inconnues: {stats['pct_unknown_brand']:.1f}%")
    
    return {'df': df_clean, 'stats': stats}


def step_save(df, filename: str = None) -> str:
    """Étape 3: Sauvegarde des données nettoyées."""
    print("\n" + "="*60)
    print("ÉTAPE 3: SAUVEGARDE DES DONNÉES")
    print("="*60)
    
    # Créer le dossier si nécessaire
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_clean_{timestamp}.csv"
    
    filepath = DATA_PROCESSED_DIR / filename
    
    # Sauvegarder
    df.to_csv(filepath, index=False)
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\n  Fichier sauvegardé: {filepath}")
    print(f"  Taille: {size_mb:.2f} MB")
    
    # Créer aussi une version "latest"
    latest_path = DATA_PROCESSED_DIR / "train_clean.csv"
    df.to_csv(latest_path, index=False)
    print(f"  Copie créée: {latest_path}")
    
    return str(filepath)


def run_pipeline(nrows: int = None, save: bool = True) -> dict:
    """Lance le pipeline complet."""
    print("\n" + "="*60)
    print("PIPELINE MERCARI PRICE PREDICTOR")
    print("="*60)
    print(f"  Démarré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if nrows:
        print(f"  Mode: échantillon ({nrows:,} lignes)")
    else:
        print(f"  Mode: dataset complet")
    
    results = {}
    
    # Étape 1: Chargement
    load_result = step_load(nrows=nrows)
    results['load'] = load_result
    
    # Étape 2: Nettoyage
    clean_result = step_clean(load_result['df'])
    results['clean'] = clean_result
    
    # Étape 3: Sauvegarde
    if save:
        filepath = step_save(clean_result['df'])
        results['save'] = {'filepath': filepath}
    
    # Résumé
    print("\n" + "="*60)
    print("PIPELINE TERMINÉ")
    print("="*60)
    print(f"  Lignes initiales: {load_result['info']['n_rows']:,}")
    print(f"  Lignes finales: {clean_result['stats']['n_rows']:,}")
    print(f"  Colonnes: {clean_result['stats']['n_cols']}")
    
    return results


# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Mercari Price Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/run_pipeline.py                  # Pipeline complet
  python scripts/run_pipeline.py --test           # Tests uniquement
  python scripts/run_pipeline.py --nrows 10000    # Échantillon de 10k lignes
  python scripts/run_pipeline.py --step load      # Étape spécifique
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Lancer les tests uniquement'
    )
    
    parser.add_argument(
        '--nrows', '-n',
        type=int,
        default=None,
        help='Nombre de lignes à charger (défaut: tout)'
    )
    
    parser.add_argument(
        '--step', '-s',
        choices=['load', 'clean', 'all'],
        default='all',
        help='Étape à exécuter (défaut: all)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Ne pas sauvegarder les données nettoyées'
    )
    
    args = parser.parse_args()
    
    # Mode test
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    
    # Mode pipeline
    try:
        if args.step == 'load':
            step_load(nrows=args.nrows)
        elif args.step == 'clean':
            result = step_load(nrows=args.nrows)
            step_clean(result['df'])
        else:
            run_pipeline(nrows=args.nrows, save=not args.no_save)
            
    except FileNotFoundError as e:
        logger.error(f"Erreur: {e}")
        print("\n💡 Télécharge d'abord les données:")
        print("   python scripts/download_data.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise


if __name__ == "__main__":
    main()