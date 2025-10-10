"""
3D visualization of organoids using PyVista
"""

import numpy as np
import torch
from torch_geometric.data import Data


try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")


def plot_organoid_3d(
    data: Data,
    output_path: str = None,
    point_size: int = 10,
    show_edges: bool = True,
):
    """
    Interactive 3D visualization of organoid
    
    Args:
        data: PyG Data object
        output_path: Path to save HTML (None = show interactive)
        point_size: Size of cell points
        show_edges: Show edges
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available!")
        return
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add points (cells)
    pos = data.pos.cpu().numpy()
    point_cloud = pv.PolyData(pos)
    plotter.add_mesh(
        point_cloud,
        color='steelblue',
        point_size=point_size,
        render_points_as_spheres=True,
    )
    
    # Add edges
    if show_edges:
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            line = pv.Line(pos[src], pos[dst])
            plotter.add_mesh(line, color='gray', opacity=0.3, line_width=1)
    
    # Configure
    plotter.add_axes()
    plotter.set_background('white')
    plotter.camera_position = 'isometric'
    
    if output_path:
        plotter.export_html(output_path)
        print(f"Saved to {output_path}")
    else:
        plotter.show()


def interactive_organoid_viewer(
    data_list: list,
    labels: list = None,
):
    """
    Interactive viewer for multiple organoids
    
    Args:
        data_list: List of PyG Data objects
        labels: Optional labels for each organoid
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available!")
        return
    
    print(f"Interactive viewer for {len(data_list)} organoids")
    print("Use left/right arrows to navigate")
    
    current_idx = [0]  # Use list to make it mutable in callback
    
    def update_view():
        plotter.clear()
        
        data = data_list[current_idx[0]]
        pos = data.pos.cpu().numpy()
        
        # Add points
        point_cloud = pv.PolyData(pos)
        plotter.add_mesh(
            point_cloud,
            color='steelblue',
            point_size=15,
            render_points_as_spheres=True,
        )
        
        # Add title
        title = f"Organoid {current_idx[0] + 1}/{len(data_list)}"
        if labels:
            title += f" | Label: {labels[current_idx[0]]}"
        title += f" | {data.num_nodes} cells"
        plotter.add_text(title, position='upper_left', font_size=12)
        
        plotter.add_axes()
        plotter.render()
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Key callbacks
    def next_organoid():
        current_idx[0] = (current_idx[0] + 1) % len(data_list)
        update_view()
    
    def prev_organoid():
        current_idx[0] = (current_idx[0] - 1) % len(data_list)
        update_view()
    
    plotter.add_key_event('Right', next_organoid)
    plotter.add_key_event('Left', prev_organoid)
    
    # Initial view
    update_view()
    
    plotter.show()


if __name__ == "__main__":
    # Test
    num_nodes = 100
    pos = torch.randn(num_nodes, 3) * 50
    edge_index = torch.randint(0, num_nodes, (2, 300))
    
    data = Data(pos=pos, edge_index=edge_index)
    
    print("Creating 3D visualization...")
    plot_organoid_3d(data, 'organoid_3d.html')

