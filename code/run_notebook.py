#!/usr/bin/env python3
"""
Script pour exécuter le notebook EGNN regression training
"""

import subprocess
import sys
from pathlib import Path

def run_notebook():
    """Exécute le notebook EGNN regression training"""
    
    notebook_path = Path("notebooks/egnn_regression_training.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook non trouvé: {notebook_path}")
        return False
    
    print("🚀 Exécution du notebook EGNN regression training...")
    
    try:
        # Exécuter le notebook avec nbconvert
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--execute", 
            "--to", "notebook",
            "--inplace",
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook exécuté avec succès!")
            return True
        else:
            print(f"❌ Erreur lors de l'exécution:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = run_notebook()
    sys.exit(0 if success else 1)
