"""
PAR-VAE: Physics-Attribute-Regularized Variational Autoencoder

A physics-informed generative model for COVID-19 CT severity classification
with interpretable latent representations grounded in biomedically-meaningful features.

Key modules:
- models: VAE architecture and physics regularizers
- data: Dataset classes and data loaders
- utils: Physics feature extraction, metrics, visualization
- evaluation: Classification and transfer evaluation protocols
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from . import models, data, utils, evaluation

__all__ = ["models", "data", "utils", "evaluation"]
