"""
ESM3 Utilities Module

This module contains utilities for working with the ESM3 model,
including model initialization, embedding generation, and sequence processing.
"""

import os
import torch
import numpy as np
from typing import List, Union, Optional
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
import huggingface_hub
from tqdm import tqdm

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ESM3EmbeddingGenerator:
    """
    A class for generating protein embeddings using ESM3 models.
    
    This class encapsulates the ESM3 model and provides methods for generating
    embeddings from protein sequences with optimized memory management.
    """
    
    def __init__(
        self, 
        model_name: str = "esm3_sm_open_v1", 
        device: Optional[str] = None,
        login_required: bool = True
    ):
        """
        Initialize the ESM3 embedding generator.
        
        Args:
            model_name: Name of the ESM3 model to load
            device: Device to load the model on (cuda/cpu). Auto-detected if None.
            login_required: Whether to require HuggingFace login
        """
        if login_required:
            huggingface_hub.login()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_name = model_name
        self.model = ESM3.from_pretrained(model_name, device=device)
    
    def get_embedding(self, sequence: str, truncate: bool = True) -> np.ndarray:
        """
        Generate embedding for a single protein sequence.
        
        Args:
            sequence: Protein sequence string
            truncate: Whether to truncate long sequences
            
        Returns:
            Embedding as numpy array
        """
        if truncate:
            sequence = truncate_sequence(sequence)
        
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        
        with torch.no_grad():
            output = self.model.forward_and_sample(
                protein_tensor, 
                SamplingConfig(return_mean_embedding=True)
            )
        
        embedding = output.mean_embedding.detach().cpu().numpy()
        
        # Clear GPU cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embedding
    
    def generate_embeddings(
        self, 
        sequences: List[str], 
        truncate: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            truncate: Whether to truncate long sequences
            
        Returns:
            List of embeddings as numpy arrays
        """
        embeddings = []
        
        iterator = tqdm(sequences, desc="Generating embeddings")
        
        for sequence in iterator:
            embedding = self.get_embedding(sequence, truncate=truncate)
            embeddings.append(embedding)
            
            # Clear cache after each embedding to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return embeddings
    
    def generate_embeddings_from_dataframe(
        self, 
        df, 
        sequence_column: str = 'sequence',
        truncate: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for sequences in a DataFrame.
        
        Args:
            df: DataFrame containing protein sequences
            sequence_column: Name of the column containing sequences
            truncate: Whether to truncate long sequences
            
        Returns:
            List of embeddings as numpy arrays
        """
        sequences = df[sequence_column].tolist()
        return self.generate_embeddings(sequences, truncate=truncate)
    
def truncate_sequence(sequence: str, max_length: int = 1800) -> str:
    """
    Truncate protein sequence to maximum length.
    
    Args:
        sequence: Protein sequence string
        max_length: Maximum allowed length
        
    Returns:
        Truncated sequence
    """
    return sequence[:max_length] if len(sequence) > max_length else sequence


def embeddings_to_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Convert list of embeddings to a matrix.
    
    Args:
        embeddings: List of embedding arrays
        
    Returns:
        Matrix with shape (n_samples, embedding_dim)
    """
    return np.array(embeddings)

 