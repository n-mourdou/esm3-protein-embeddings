"""
ESM3 Protein Analysis

Tools for protein analysis using ESM3 embeddings.
"""

# Import modules for convenient access
from . import esm3_utils
from . import data_processing  
from . import classification


# Import the main classes for direct access
from .esm3_utils import ESM3EmbeddingGenerator

# Import utility functions
from .esm3_utils import (
    truncate_sequence,
    embeddings_to_matrix,
) 