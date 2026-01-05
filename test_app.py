#!/usr/bin/env python
"""
Script de test pour vérifier que l'app Streamlit peut se lancer.
Usage: python test_app.py
"""

import sys
from pathlib import Path

# Ajouter les paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test que tous les imports fonctionnent."""
    print("Test des imports...")
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import joblib
        print("  ✓ joblib")
    except ImportError as e:
        print(f"  ✗ joblib: {e}")
        return False
    
    try:
        import streamlit as st
        print("  ✓ streamlit")
    except ImportError as e:
        print(f"  ✗ streamlit: {e}")
        return False
    
    try:
        from scipy.sparse import csr_matrix
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    return True


def test_model_files():
    """Test que les fichiers modèle existent."""
    print("\nTest des fichiers modèle...")
    
    model_path = PROJECT_ROOT / "models" / "model_v1.joblib"
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor_v1.joblib"
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ model_v1.joblib ({size:.1f} MB)")
    else:
        print(f"  ✗ model_v1.joblib manquant")
        print(f"    Attendu: {model_path}")
        return False
    
    if preprocessor_path.exists():
        size = preprocessor_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ preprocessor_v1.joblib ({size:.1f} MB)")
    else:
        print(f"  ✗ preprocessor_v1.joblib manquant")
        print(f"    Attendu: {preprocessor_path}")
        return False
    
    return True


def test_model_loading():
    """Test le chargement du modèle."""
    print("\nTest du chargement du modèle...")
    
    import joblib
    
    model_path = PROJECT_ROOT / "models" / "model_v1.joblib"
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor_v1.joblib"
    
    if not model_path.exists():
        print("  ⊘ Skipped (fichiers manquants)")
        return True
    
    try:
        model = joblib.load(model_path)
        print(f"  ✓ Modèle chargé: {type(model).__name__}")
    except Exception as e:
        print(f"  ✗ Erreur modèle: {e}")
        return False
    
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"  ✓ Preprocessor chargé: {type(preprocessor).__name__}")
        if hasattr(preprocessor, 'n_features_'):
            print(f"    Features: {preprocessor.n_features_}")
    except Exception as e:
        print(f"  ✗ Erreur preprocessor: {e}")
        return False
    
    return True


def main():
    print("=" * 50)
    print("TEST MERCARI PRICE PREDICTOR APP")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Fichiers modèle", test_model_files()))
    results.append(("Chargement modèle", test_model_loading()))
    
    print("\n" + "=" * 50)
    print("RÉSUMÉ")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Tous les tests passent!")
        print("\nPour lancer l'app:")
        print("  streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️ Certains tests ont échoué")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
