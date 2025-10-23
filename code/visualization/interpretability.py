"""
Interpretability and explainability for GNN models
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GNNExplainer:
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_node_importances_gradcam(
        self,
        data: Data,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        data = data.to(self.device)
        self.model.zero_grad()
        
        activations = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        hook_handle = None
        for name, module in self.model.named_modules():
            if 'conv' in name.lower() or 'layer' in name.lower():
                hook_handle = module.register_forward_hook(forward_hook)
                break
        
        data.x.requires_grad = True
        
        output = self.model(data.x, data.edge_index, data.batch)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target_score = output[0, target_class]
        target_score.backward()
        
        if hook_handle is not None:
            hook_handle.remove()
        
        if len(activations) > 0 and data.x.grad is not None:
            node_importances = torch.abs(data.x.grad).sum(dim=1).cpu().numpy()
        else:
            node_importances = np.ones(data.num_nodes)
        
        node_importances = node_importances / (node_importances.max() + 1e-8)
        
        return node_importances
    
    def compute_node_importances_attention(
        self,
        data: Data,
    ) -> np.ndarray:
        data = data.to(self.device)
        
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'alpha'):
                attention_weights.append(module.alpha.detach())
        
        hooks = []
        for module in self.model.modules():
            if 'GAT' in module.__class__.__name__:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(data.x, data.edge_index, data.batch)
        
        for hook in hooks:
            hook.remove()
        
        if len(attention_weights) == 0:
            return np.ones(data.num_nodes)
        
        node_importances = torch.zeros(data.num_nodes, device=self.device)
        
        for attn in attention_weights:
            edge_index = data.edge_index
            for i in range(edge_index.size(1)):
                src, dst = edge_index[:, i]
                node_importances[dst] += attn[i].mean()
        
        node_importances = node_importances.cpu().numpy()
        node_importances = node_importances / (node_importances.max() + 1e-8)
        
        return node_importances
    
    def compute_node_importances_perturbation(
        self,
        data: Data,
        num_samples: int = 100,
    ) -> np.ndarray:
        data = data.to(self.device)
        
        with torch.no_grad():
            baseline_output = self.model(data.x, data.edge_index, data.batch)
            baseline_pred = baseline_output.argmax(dim=1).item()
        
        node_importances = np.zeros(data.num_nodes)
        
        for node_idx in range(data.num_nodes):
            perturbed_data = data.clone()
            
            perturbed_data.x[node_idx] = 0
            
            with torch.no_grad():
                perturbed_output = self.model(
                    perturbed_data.x,
                    perturbed_data.edge_index,
                    perturbed_data.batch
                )
            
            importance = torch.abs(
                baseline_output[0, baseline_pred] - perturbed_output[0, baseline_pred]
            ).item()
            
            node_importances[node_idx] = importance
        
        node_importances = node_importances / (node_importances.max() + 1e-8)
        
        return node_importances
    
    def explain_prediction(
        self,
        data: Data,
        method: str = 'gradcam',
        top_k: int = 10,
    ) -> Dict:
        if method == 'gradcam':
            importances = self.compute_node_importances_gradcam(data)
        elif method == 'attention':
            importances = self.compute_node_importances_attention(data)
        elif method == 'perturbation':
            importances = self.compute_node_importances_perturbation(data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        with torch.no_grad():
            output = self.model(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                data.batch.to(self.device)
            )
            predicted_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, predicted_class].item()
        
        top_indices = np.argsort(importances)[-top_k:][::-1]
        top_importances = importances[top_indices]
        
        explanation = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'node_importances': importances,
            'top_k_nodes': top_indices,
            'top_k_importances': top_importances,
            'method': method,
        }
        
        return explanation
    
    def get_important_cells(
        self,
        data: Data,
        threshold: float = 0.5,
        method: str = 'gradcam',
    ) -> Tuple[np.ndarray, np.ndarray]:
        explanation = self.explain_prediction(data, method=method)
        
        importances = explanation['node_importances']
        important_mask = importances >= threshold
        
        important_indices = np.where(important_mask)[0]
        important_scores = importances[important_mask]
        
        return important_indices, important_scores


class SaliencyMapper:
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.explainer = GNNExplainer(model, device)
    
    def create_saliency_map(
        self,
        data: Data,
        method: str = 'gradcam',
    ) -> np.ndarray:
        positions = data.pos.cpu().numpy()
        importances = self.explainer.compute_node_importances_gradcam(data)
        
        return positions, importances
    
    def visualize_saliency_3d(
        self,
        data: Data,
        method: str = 'gradcam',
        save_path: Optional[str] = None,
    ):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("matplotlib required for visualization")
            return
        
        positions, importances = self.create_saliency_map(data, method)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=importances,
            cmap='hot',
            s=50,
            alpha=0.8,
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cell Importance ({method})')
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Importance')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved saliency map to {save_path}")
        else:
            plt.show()
        
        plt.close()


def explain_batch_predictions(
    model,
    dataloader,
    device: str = 'cuda',
    method: str = 'gradcam',
    output_dir: Optional[str] = None,
) -> List[Dict]:
    from pathlib import Path
    
    explainer = GNNExplainer(model, device)
    mapper = SaliencyMapper(model, device)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    explanations = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        num_graphs = batch.num_graphs
        ptr = batch.ptr
        
        for graph_idx in range(num_graphs):
            start_idx = ptr[graph_idx]
            end_idx = ptr[graph_idx + 1]
            
            sub_data = Data(
                x=batch.x[start_idx:end_idx],
                edge_index=batch.edge_index[:, (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx)] - start_idx,
                pos=batch.pos[start_idx:end_idx],
                batch=torch.zeros(end_idx - start_idx, dtype=torch.long, device=device),
            )
            
            explanation = explainer.explain_prediction(sub_data, method=method)
            explanations.append(explanation)
            
            if output_dir:
                save_path = output_path / f"saliency_batch{batch_idx}_graph{graph_idx}.png"
                mapper.visualize_saliency_3d(sub_data, method=method, save_path=str(save_path))
    
    logger.info(f"Generated explanations for {len(explanations)} graphs")
    
    return explanations


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from models.classifier import OrganoidClassifier
    from data.loader import get_dataloaders
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'attention', 'perturbation'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    model = OrganoidClassifier.load_from_checkpoint(args.model_path, args.device)
    
    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    explanations = explain_batch_predictions(
        model,
        test_loader,
        device=args.device,
        method=args.method,
        output_dir=args.output_dir,
    )
    
    logger.info("Explanation generation complete!")
