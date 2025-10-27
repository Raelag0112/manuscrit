"""
Script pour calculer les coefficients de clustering de tous les graphes
dans D:\data_prepared et créer un histogramme.
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from tqdm import tqdm

def load_graph(file_path):
    """Charge un graphe depuis un fichier .pt"""
    try:
        data = torch.load(file_path, weights_only=False)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None

def compute_clustering_coefficient(data):
    """
    Calcule le coefficient de clustering moyen d'un graphe PyTorch Geometric.
    
    Args:
        data: torch_geometric.data.Data object
        
    Returns:
        float: coefficient de clustering moyen
    """
    try:
        # Extraire edge_index
        edge_index = data.edge_index.numpy()
        
        # Créer un graphe NetworkX
        G = nx.Graph()
        
        # Ajouter les nœuds
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else edge_index.max() + 1
        G.add_nodes_from(range(num_nodes))
        
        # Ajouter les arêtes
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Calculer le coefficient de clustering moyen
        clustering_coef = nx.average_clustering(G)
        
        return clustering_coef
    except Exception as e:
        print(f"Erreur lors du calcul du clustering: {e}")
        return None

def analyze_all_graphs(data_dir):
    """
    Analyse tous les graphes dans le répertoire data_dir.
    
    Args:
        data_dir: chemin vers le répertoire contenant les graphes
        
    Returns:
        dict: dictionnaire avec les informations sur chaque graphe
    """
    results = []
    
    # Parcourir tous les sous-dossiers (train, test, val)
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Le répertoire {split_dir} n'existe pas.")
            continue
        
        print(f"\nTraitement du split: {split}")
        
        # Lister tous les fichiers .pt
        graph_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.pt')])
        
        print(f"Nombre de graphes trouvés: {len(graph_files)}")
        
        # Traiter chaque graphe
        for graph_file in tqdm(graph_files, desc=f"Analyse {split}"):
            file_path = os.path.join(split_dir, graph_file)
            
            # Charger le graphe
            data = load_graph(file_path)
            if data is None:
                continue
            
            # Calculer le coefficient de clustering
            clustering_coef = compute_clustering_coefficient(data)
            if clustering_coef is None:
                continue
            
            # Déterminer le type d'organoïde
            if 'Cystiques' in graph_file or 'cystic' in graph_file:
                organoid_type = 'Cystiques'
            elif 'Chouxfleurs' in graph_file or 'cauliflower' in graph_file:
                organoid_type = 'Choux-fleurs'
            else:
                organoid_type = 'Unknown'
            
            # Stocker les résultats
            results.append({
                'file': graph_file,
                'split': split,
                'type': organoid_type,
                'clustering_coefficient': clustering_coef,
                'num_nodes': data.num_nodes if hasattr(data, 'num_nodes') else data.edge_index.max().item() + 1,
                'num_edges': data.edge_index.shape[1]
            })
    
    return results

def plot_histogram(results, output_file='clustering_histogram.png'):
    """
    Crée un histogramme des coefficients de clustering.
    
    Args:
        results: liste de dictionnaires contenant les résultats
        output_file: nom du fichier de sortie
    """
    # Extraire les coefficients de clustering
    clustering_coeffs = [r['clustering_coefficient'] for r in results]
    
    # Séparer par type
    cystiques_coeffs = [r['clustering_coefficient'] for r in results if r['type'] == 'Cystiques']
    chouxfleurs_coeffs = [r['clustering_coefficient'] for r in results if r['type'] == 'Choux-fleurs']
    
    # Statistiques globales
    print("\n" + "="*60)
    print("STATISTIQUES GLOBALES")
    print("="*60)
    print(f"Nombre total de graphes: {len(clustering_coeffs)}")
    print(f"Coefficient de clustering moyen: {np.mean(clustering_coeffs):.4f}")
    print(f"Écart-type: {np.std(clustering_coeffs):.4f}")
    print(f"Minimum: {np.min(clustering_coeffs):.4f}")
    print(f"Maximum: {np.max(clustering_coeffs):.4f}")
    print(f"Médiane: {np.median(clustering_coeffs):.4f}")
    
    # Statistiques par type
    print("\n" + "="*60)
    print("STATISTIQUES PAR TYPE D'ORGANOÏDE")
    print("="*60)
    if cystiques_coeffs:
        print(f"\nCYSTIQUES (n={len(cystiques_coeffs)}):")
        print(f"  Moyenne: {np.mean(cystiques_coeffs):.4f}")
        print(f"  Écart-type: {np.std(cystiques_coeffs):.4f}")
        print(f"  Min: {np.min(cystiques_coeffs):.4f}")
        print(f"  Max: {np.max(cystiques_coeffs):.4f}")
        print(f"  Médiane: {np.median(cystiques_coeffs):.4f}")
    
    if chouxfleurs_coeffs:
        print(f"\nCHOUX-FLEURS (n={len(chouxfleurs_coeffs)}):")
        print(f"  Moyenne: {np.mean(chouxfleurs_coeffs):.4f}")
        print(f"  Écart-type: {np.std(chouxfleurs_coeffs):.4f}")
        print(f"  Min: {np.min(chouxfleurs_coeffs):.4f}")
        print(f"  Max: {np.max(chouxfleurs_coeffs):.4f}")
        print(f"  Médiane: {np.median(chouxfleurs_coeffs):.4f}")
    
    # Créer la figure avec 3 lignes et 2 colonnes
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Couleurs pour les types
    type_colors = {'Cystiques': '#e74c3c', 'Choux-fleurs': '#3498db'}
    
    # 1. Histogramme global par type
    ax1 = fig.add_subplot(gs[0, 0])
    if cystiques_coeffs:
        ax1.hist(cystiques_coeffs, bins=40, alpha=0.6, label=f'Cystiques (n={len(cystiques_coeffs)})', 
                color=type_colors['Cystiques'], edgecolor='black')
    if chouxfleurs_coeffs:
        ax1.hist(chouxfleurs_coeffs, bins=40, alpha=0.6, label=f'Choux-fleurs (n={len(chouxfleurs_coeffs)})', 
                color=type_colors['Choux-fleurs'], edgecolor='black')
    ax1.set_xlabel('Coefficient de clustering', fontsize=12)
    ax1.set_ylabel('Fréquence', fontsize=12)
    ax1.set_title('Distribution par type d\'organoïde', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot par type
    ax2 = fig.add_subplot(gs[0, 1])
    type_data = []
    type_labels = []
    type_cols = []
    if cystiques_coeffs:
        type_data.append(cystiques_coeffs)
        type_labels.append(f'Cystiques\n(n={len(cystiques_coeffs)})')
        type_cols.append(type_colors['Cystiques'])
    if chouxfleurs_coeffs:
        type_data.append(chouxfleurs_coeffs)
        type_labels.append(f'Choux-fleurs\n(n={len(chouxfleurs_coeffs)})')
        type_cols.append(type_colors['Choux-fleurs'])
    
    if type_data:
        bp = ax2.boxplot(type_data, tick_labels=type_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], type_cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Coefficient de clustering', fontsize=12)
    ax2.set_title('Box plot par type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Scatter plot: clustering vs taille par type
    ax3 = fig.add_subplot(gs[1, :])
    for org_type, color in type_colors.items():
        type_results = [r for r in results if r['type'] == org_type]
        if type_results:
            x = [r['num_nodes'] for r in type_results]
            y = [r['clustering_coefficient'] for r in type_results]
            ax3.scatter(x, y, alpha=0.5, label=f'{org_type} (n={len(type_results)})', 
                       color=color, s=40, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Nombre de nœuds', fontsize=12)
    ax3.set_ylabel('Coefficient de clustering', fontsize=12)
    ax3.set_title('Clustering vs Taille du graphe par type', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution par type et par split (Cystiques)
    ax4 = fig.add_subplot(gs[2, 0])
    splits = ['train', 'test', 'val']
    split_colors = {'train': '#9b59b6', 'test': '#e67e22', 'val': '#16a085'}
    for split in splits:
        split_cyst = [r['clustering_coefficient'] for r in results 
                     if r['type'] == 'Cystiques' and r['split'] == split]
        if split_cyst:
            ax4.hist(split_cyst, bins=20, alpha=0.5, label=f'{split} (n={len(split_cyst)})', 
                    color=split_colors[split], edgecolor='black')
    ax4.set_xlabel('Coefficient de clustering', fontsize=12)
    ax4.set_ylabel('Fréquence', fontsize=12)
    ax4.set_title('Cystiques - Distribution par split', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution par type et par split (Choux-fleurs)
    ax5 = fig.add_subplot(gs[2, 1])
    for split in splits:
        split_chou = [r['clustering_coefficient'] for r in results 
                     if r['type'] == 'Choux-fleurs' and r['split'] == split]
        if split_chou:
            ax5.hist(split_chou, bins=20, alpha=0.5, label=f'{split} (n={len(split_chou)})', 
                    color=split_colors[split], edgecolor='black')
    ax5.set_xlabel('Coefficient de clustering', fontsize=12)
    ax5.set_ylabel('Fréquence', fontsize=12)
    ax5.set_title('Choux-fleurs - Distribution par split', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHistogramme sauvegardé dans: {output_file}")
    plt.show()
    
    return fig

def save_results_to_csv(results, output_file='clustering_results.csv'):
    """Sauvegarde les résultats dans un fichier CSV."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'split', 'type', 'clustering_coefficient', 
                                                'num_nodes', 'num_edges'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Résultats sauvegardés dans: {output_file}")

def main():
    # Chemin vers le répertoire des données
    data_dir = r'D:\data_prepared'
    
    if not os.path.exists(data_dir):
        print(f"ERREUR: Le répertoire {data_dir} n'existe pas.")
        return
    
    print("="*60)
    print("ANALYSE DES COEFFICIENTS DE CLUSTERING")
    print("="*60)
    print(f"Répertoire: {data_dir}")
    
    # Analyser tous les graphes
    results = analyze_all_graphs(data_dir)
    
    if not results:
        print("Aucun graphe n'a pu être analysé.")
        return
    
    # Sauvegarder les résultats
    save_results_to_csv(results, 'clustering_results.csv')
    
    # Créer l'histogramme
    plot_histogram(results, 'clustering_histogram.png')
    
    # Statistiques par split
    print("\n" + "="*60)
    print("STATISTIQUES PAR SPLIT")
    print("="*60)
    for split in ['train', 'test', 'val']:
        split_coeffs = [r['clustering_coefficient'] for r in results if r['split'] == split]
        if split_coeffs:
            print(f"\n{split.upper()}:")
            print(f"  Nombre de graphes: {len(split_coeffs)}")
            print(f"  Moyenne: {np.mean(split_coeffs):.4f}")
            print(f"  Écart-type: {np.std(split_coeffs):.4f}")
            print(f"  Min: {np.min(split_coeffs):.4f}")
            print(f"  Max: {np.max(split_coeffs):.4f}")
    
    # Statistiques croisées: par split et par type
    print("\n" + "="*60)
    print("STATISTIQUES DÉTAILLÉES PAR SPLIT ET TYPE")
    print("="*60)
    for split in ['train', 'test', 'val']:
        print(f"\n{split.upper()}:")
        for org_type in ['Cystiques', 'Choux-fleurs']:
            type_split_coeffs = [r['clustering_coefficient'] for r in results 
                                if r['split'] == split and r['type'] == org_type]
            if type_split_coeffs:
                print(f"  {org_type} (n={len(type_split_coeffs)}):")
                print(f"    Moyenne: {np.mean(type_split_coeffs):.4f}")
                print(f"    Écart-type: {np.std(type_split_coeffs):.4f}")
                print(f"    Min-Max: [{np.min(type_split_coeffs):.4f}, {np.max(type_split_coeffs):.4f}]")

if __name__ == '__main__':
    main()

