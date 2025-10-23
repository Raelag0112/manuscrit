#!/usr/bin/env python3
"""
Test script pour vérifier que tous les imports du notebook fonctionnent
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent))

print("🧪 Test des imports pour le notebook EGNN...")

try:
    # Test imports de base
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
    print("✅ Imports de base OK")
    
    # Test imports du projet
    from synthetic.generator import SyntheticOrganoidGenerator
    print("✅ SyntheticOrganoidGenerator importé")
    
    from models.egnn import EGNNClassifier
    print("✅ EGNNClassifier importé")
    
    # Test création d'un générateur
    generator = SyntheticOrganoidGenerator(
        radius=100.0,
        edge_method='knn',
        k_neighbors=8,
        seed=42
    )
    print("✅ Générateur créé")
    
    # Test génération d'un organoïde simple
    data = generator.generate_organoid(
        num_cells=50,
        process_type='poisson',
        label=0,
        feature_dim=5
    )
    print(f"✅ Organoïde généré: {data.num_nodes} nœuds, {data.num_edges} arêtes")
    
    print("\n🎉 Tous les imports fonctionnent correctement!")
    print("Le notebook devrait pouvoir s'exécuter sans problème.")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les modules sont correctement installés.")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    sys.exit(1)
