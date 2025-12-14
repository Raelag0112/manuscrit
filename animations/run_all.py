"""
Run all animation scripts to generate GIFs for the presentation.
"""

import os
import sys

# Ensure animations folder exists
os.makedirs('animations', exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Generating animations for thesis presentation")
print("=" * 60)

animations = [
    ("message_passing.py", "Message Passing in GNNs"),
    ("point_processes.py", "Poisson vs Matérn Point Processes"),
    ("graph_construction.py", "Graph Construction from Cells"),
    ("rotation_invariance.py", "E(3) Rotation Invariance"),
    ("attention_mechanism.py", "GAT Attention Mechanism"),
    ("transfer_learning.py", "Transfer Learning Process"),
]

for script, description in animations:
    print(f"\n{'=' * 60}")
    print(f"Generating: {description}")
    print(f"Script: {script}")
    print("=" * 60)
    
    try:
        # Import and run each module
        module_name = script.replace('.py', '')
        exec(f"import {module_name}")
        print(f"✓ Successfully generated {module_name}.gif")
    except Exception as e:
        print(f"✗ Error generating {script}: {e}")

print("\n" + "=" * 60)
print("Animation generation complete!")
print("GIF files are saved in the 'animations' folder.")
print("=" * 60)
