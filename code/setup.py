"""
Setup script for organoid analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Deep Learning for Organoid Analysis with Graph Neural Networks"

setup(
    name="organoid-gnn",
    version="1.0.0",
    author="Alexandre Martin",
    author_email="",
    description="Graph Neural Networks for 3D Organoid Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/organoid-gnn",
    packages=find_packages(exclude=['tests', 'notebooks', 'scripts']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "cellpose>=2.2.0",
        "scikit-image>=0.20.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "plotly>=5.13.0",
            "pyvista>=0.38.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'organoid-train=scripts.train:main',
            'organoid-evaluate=scripts.evaluate:main',
            'organoid-generate=scripts.generate_data:main',
        ],
    },
)

