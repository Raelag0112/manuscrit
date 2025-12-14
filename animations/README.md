# Animations for Thesis Presentation

This folder contains Python scripts to generate animated GIFs for the thesis defense presentation.

## Animations

| Script | Description | Slide(s) |
|--------|-------------|----------|
| `message_passing.py` | Message passing in GNNs | Slide 12 |
| `point_processes.py` | Poisson vs Matérn processes | Slide 29 |
| `graph_construction.py` | Graph construction from cells | Slides 11, 28 |
| `rotation_invariance.py` | E(3) rotation invariance | Slides 17-19 |
| `attention_mechanism.py` | GAT attention weights | Slide 15 |
| `transfer_learning.py` | Pre-training + fine-tuning | Slides 30, 42 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate all animations
```bash
python run_all.py
```

### Generate individual animation
```bash
python message_passing.py
python point_processes.py
# etc.
```

## Output

All GIFs are saved in the `animations/` folder:
- `message_passing.gif`
- `point_processes.gif`
- `graph_construction.gif`
- `rotation_invariance.gif`
- `attention_mechanism.gif`
- `transfer_learning.gif`

## Including in LaTeX Presentation

To include animated GIFs in the Beamer presentation, you'll need to convert them to a format Beamer can use. Options:

### Option 1: Use `animate` package (PDF)
```latex
\usepackage{animate}
% Then use \animategraphics command
```

### Option 2: Extract key frames as static images
Use the GIFs for PowerPoint/HTML presentations, or extract key frames:
```bash
# Using ImageMagick
convert animation.gif frame_%03d.png
```

### Option 3: Use in PowerPoint
Import the GIFs directly into PowerPoint slides for presentations.

## Customization

Each script has configuration variables at the top:
- `FIG_SIZE`: Figure dimensions
- `DPI`: Resolution (higher = larger file)
- `SAVE_GIF`: Set to `False` to preview without saving

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- NetworkX
- Pillow (for GIF export)
