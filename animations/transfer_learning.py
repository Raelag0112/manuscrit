"""
Animation: Transfer Learning from Synthetic to Real Data
Shows the pre-training and fine-tuning process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

# Configuration
np.random.seed(42)
FIG_SIZE = (14, 7)
DPI = 150
SAVE_GIF = True


def animate_transfer_learning():
    """Create animation showing transfer learning process."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Title
    title = ax.text(7, 6.5, 'Transfer Learning: Synthetic → Real', 
                   ha='center', fontsize=16, fontweight='bold', color='white')
    
    # Phase indicator
    phase_text = ax.text(7, 6.0, '', ha='center', fontsize=12, color='#888888')
    
    # ===== LEFT SIDE: Synthetic Data =====
    synthetic_box = FancyBboxPatch((0.5, 2.5), 3, 3, boxstyle="round,pad=0.1",
                                    facecolor='#1E3A5F', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(synthetic_box)
    ax.text(2, 5.2, 'Synthetic Data', ha='center', fontsize=11, 
           fontweight='bold', color='#4CAF50')
    ax.text(2, 4.7, '100,000 organoids', ha='center', fontsize=9, color='white')
    
    # Synthetic data icons (small circles)
    synthetic_dots = []
    for i in range(25):
        x = 0.8 + (i % 5) * 0.5
        y = 2.8 + (i // 5) * 0.4
        dot = Circle((x, y), 0.12, color='#4CAF50', alpha=0.6)
        ax.add_patch(dot)
        synthetic_dots.append(dot)
    
    # ===== CENTER: Neural Network =====
    nn_box = FancyBboxPatch((5, 1.5), 4, 4.5, boxstyle="round,pad=0.1",
                             facecolor='#2D2D44', edgecolor='#2196F3', linewidth=2)
    ax.add_patch(nn_box)
    ax.text(7, 5.7, 'Graph Neural Network', ha='center', fontsize=11, 
           fontweight='bold', color='#2196F3')
    
    # Network layers
    layer_boxes = []
    layer_labels = ['Input', 'GAT×5', 'Pool', 'MLP', 'Output']
    layer_colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800', '#F44336']
    for i, (label, color) in enumerate(zip(layer_labels, layer_colors)):
        y = 4.8 - i * 0.8
        box = FancyBboxPatch((5.3, y - 0.25), 3.4, 0.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1, alpha=0.7)
        ax.add_patch(box)
        layer_boxes.append(box)
        ax.text(7, y, label, ha='center', va='center', fontsize=9, 
               fontweight='bold', color='white')
    
    # Weights indicator
    weights_text = ax.text(7, 1.3, 'Weights: Random', ha='center', 
                          fontsize=10, color='#888888')
    
    # ===== RIGHT SIDE: Real Data =====
    real_box = FancyBboxPatch((10.5, 2.5), 3, 3, boxstyle="round,pad=0.1",
                               facecolor='#3D1F1F', edgecolor='#FF9800', linewidth=2)
    ax.add_patch(real_box)
    ax.text(12, 5.2, 'Real Data', ha='center', fontsize=11, 
           fontweight='bold', color='#FF9800')
    ax.text(12, 4.7, '500 organoids', ha='center', fontsize=9, color='white')
    
    # Real data icons
    real_dots = []
    for i in range(12):
        x = 10.8 + (i % 3) * 0.8
        y = 2.8 + (i // 3) * 0.5
        dot = Circle((x, y), 0.15, color='#FF9800', alpha=0.6)
        ax.add_patch(dot)
        real_dots.append(dot)
    
    # ===== ARROWS =====
    # Synthetic to NN
    arrow1 = ax.annotate('', xy=(5, 4), xytext=(3.5, 4),
                        arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=3))
    ax.text(4.2, 4.3, 'Pre-train', ha='center', fontsize=9, color='#4CAF50')
    
    # NN to Real
    arrow2 = ax.annotate('', xy=(10.5, 4), xytext=(9, 4),
                        arrowprops=dict(arrowstyle='->', color='#FF9800', lw=3, alpha=0))
    fine_tune_text = ax.text(9.7, 4.3, 'Fine-tune', ha='center', 
                            fontsize=9, color='#FF9800', alpha=0)
    
    # ===== RESULTS BOX =====
    results_box = FancyBboxPatch((5, 0.3), 4, 0.8, boxstyle="round,pad=0.1",
                                  facecolor='#1a1a2e', edgecolor='#4CAF50', 
                                  linewidth=2, alpha=0)
    ax.add_patch(results_box)
    results_text = ax.text(7, 0.7, '', ha='center', fontsize=11, 
                          fontweight='bold', color='white')
    
    # Progress bar
    progress_bg = Rectangle((0.5, 0.5), 3.5, 0.3, fill=True, 
                            facecolor='#333333', edgecolor='white', linewidth=1)
    ax.add_patch(progress_bg)
    progress_fill = Rectangle((0.5, 0.5), 0, 0.3, fill=True, 
                              facecolor='#4CAF50', alpha=0.8)
    ax.add_patch(progress_fill)
    progress_label = ax.text(2.25, 0.35, '', ha='center', fontsize=9, color='white')
    
    def init():
        return [title, phase_text, weights_text, results_text, progress_label]
    
    def animate(frame):
        total_frames = 240
        
        # Phase 1: Pre-training on synthetic (frames 0-100)
        if frame < 100:
            phase_text.set_text('Phase 1: Pre-training on Synthetic Data')
            
            progress = frame / 100
            progress_fill.set_width(3.5 * progress)
            progress_label.set_text(f'Epoch {int(progress * 200)}/200')
            progress_fill.set_facecolor('#4CAF50')
            
            # Animate synthetic data flowing
            for i, dot in enumerate(synthetic_dots):
                phase = (frame * 0.1 + i * 0.3) % (2 * np.pi)
                alpha = 0.4 + 0.4 * np.sin(phase)
                dot.set_alpha(alpha)
            
            # Animate network layers
            for i, box in enumerate(layer_boxes):
                phase = (frame * 0.15 + i * 0.5) % (2 * np.pi)
                alpha = 0.5 + 0.3 * np.sin(phase)
                box.set_alpha(alpha)
            
            # Update weights text
            if frame < 30:
                weights_text.set_text('Weights: Initializing...')
            elif frame < 70:
                weights_text.set_text('Weights: Learning patterns...')
            else:
                weights_text.set_text('Weights: Pre-trained ✓')
                weights_text.set_color('#4CAF50')
        
        # Phase 2: Transition (frames 100-120)
        elif frame < 120:
            phase_text.set_text('Transferring learned weights...')
            
            progress = (frame - 100) / 20
            
            # Fade out synthetic activity
            for dot in synthetic_dots:
                dot.set_alpha(0.3)
            
            # Show fine-tune arrow
            arrow2.arrow_patch.set_alpha(progress)
            fine_tune_text.set_alpha(progress)
            
            # Reset progress bar
            progress_fill.set_width(0)
            progress_fill.set_facecolor('#FF9800')
            progress_label.set_text('')
        
        # Phase 3: Fine-tuning on real (frames 120-200)
        elif frame < 200:
            phase_text.set_text('Phase 2: Fine-tuning on Real Data')
            
            progress = (frame - 120) / 80
            progress_fill.set_width(3.5 * progress)
            progress_label.set_text(f'Epoch {int(progress * 100)}/100')
            
            # Animate real data
            for i, dot in enumerate(real_dots):
                phase = (frame * 0.1 + i * 0.5) % (2 * np.pi)
                alpha = 0.4 + 0.5 * np.sin(phase)
                dot.set_alpha(alpha)
            
            # Animate network layers (gentler, fine-tuning)
            for i, box in enumerate(layer_boxes):
                phase = (frame * 0.08 + i * 0.3) % (2 * np.pi)
                alpha = 0.7 + 0.2 * np.sin(phase)
                box.set_alpha(alpha)
            
            weights_text.set_text('Weights: Fine-tuning...')
            weights_text.set_color('#FF9800')
        
        # Phase 4: Results (frames 200+)
        else:
            phase_text.set_text('Training Complete!')
            phase_text.set_color('#4CAF50')
            
            weights_text.set_text('Weights: Optimized ✓')
            weights_text.set_color('#4CAF50')
            
            # Show results
            results_box.set_alpha(1)
            results_text.set_text('Accuracy: 84% (+8% vs from-scratch)')
            
            # Pulse effect
            pulse = 0.8 + 0.2 * np.sin((frame - 200) * 0.2)
            results_box.set_edgecolor('#4CAF50')
            
            progress_fill.set_width(3.5)
            progress_label.set_text('Complete!')
        
        return [title, phase_text, weights_text, results_text, progress_label]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=240, interval=50, blit=False)
    
    if SAVE_GIF:
        print("Saving transfer_learning.gif...")
        anim.save('animations/transfer_learning.gif', writer='pillow', fps=20, dpi=DPI)
        print("Saved!")
    
    plt.show()
    return anim


if __name__ == "__main__":
    import os
    os.makedirs('animations', exist_ok=True)
    animate_transfer_learning()
