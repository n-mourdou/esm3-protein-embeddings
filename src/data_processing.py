"""
Data Processing Module

This module contains functions for processing protein data,
including FASTA file handling, sequence manipulation, and dataset preparation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from ete3 import NCBITaxa
import warnings


def process_fasta_file(
    file_path: Union[str, Path], 
    druggable_value: Optional[int] = None
) -> pd.DataFrame:
    """
    Process a FASTA file to extract protein information.
    
    Args:
        file_path: Path to the FASTA file
        druggable_value: Optional druggable classification (0 or 1)
        
    Returns:
        DataFrame with columns: Accession, Description, Sequence, and optionally Druggable
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
    
    with open(file_path, 'r') as file:
        fasta_content = file.readlines()
    
    # Initialize lists to store data
    accessions = []
    descriptions = []
    sequences = []
    
    # Initialize variables for sequence collection
    current_sequence = []
    current_header = None
    
    # Process the FASTA file content
    for line in fasta_content:
        if line.startswith('>'):  # Header line
            if current_header:
                # Save the previous sequence before starting a new one
                sequences.append(''.join(current_sequence))
                current_sequence = []
                
            # Process the header to extract accession and description
            current_header = line.strip()
            
            # Remove the '>' character and 'sp|' prefix, then split by '|'
            parts = current_header[1:].split('|')
            
            if len(parts) == 3:
                accession = parts[1]
                description = parts[2]
            else:
                accession = None
                description = current_header[1:]  # Use full header as description
            
            accessions.append(accession)
            descriptions.append(description)
        else:
            # Collect sequence lines
            current_sequence.append(line.strip())
    
    # Append the last sequence after exiting the loop
    if current_header and current_sequence:
        sequences.append(''.join(current_sequence))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Accession': accessions,
        'Description': descriptions,
        'Sequence': sequences
    })
    
    # Add druggable column if specified
    if druggable_value is not None:
        df['Druggable'] = druggable_value
    
    return df




def map_taxid_to_kingdom(taxid: int) -> str:
    """
    Map a TaxID to its corresponding kingdom.
    
    Args:
        taxid: NCBI Taxonomy ID
        
    Returns:
        Kingdom name ('bacteria', 'archaea', 'eukaryota', 'virus', or 'unclassified')
    """
    try:
        # Suppress ete3 taxid translation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="taxid .* was translated into .*")
            
            ncbi = NCBITaxa()
            lineage = ncbi.get_lineage(taxid)
            names = ncbi.get_taxid_translator(lineage)
        
        # Check for key taxa in the lineage
        if 'Bacteria' in names.values():
            return 'bacteria'
        elif 'Archaea' in names.values():
            return 'archaea'
        elif 'Eukaryota' in names.values():
            return 'eukaryota'
        elif 'Virus' in names.values():
            return 'virus'
        else:
            return 'unclassified'
    except Exception as e:
        warnings.warn(f"Could not classify TaxID {taxid}: {str(e)}")
        return 'unclassified'




def is_human_protein(description: str) -> bool:
    """
    Determine if a protein is human based on its description.
    
    Args:
        description: Protein description string
        
    Returns:
        True if the protein is human, False otherwise
    """
    if pd.notna(description):
        return 'HUMAN' in description.upper()
    return False


def add_human_classification(df: pd.DataFrame, description_column: str = 'Description') -> pd.DataFrame:
    """
    Add human classification to a DataFrame.
    
    Args:
        df: DataFrame containing protein descriptions
        description_column: Name of the description column
        
    Returns:
        DataFrame with added 'Is_Human' column
    """
    df = df.copy()
    df['Is_Human'] = df[description_column].apply(is_human_protein)
    return df




def check_sequence_location(
    sequence: str, 
    train_df: pd.DataFrame, 
    sequence_column: str = 'Sequence'
) -> bool:
    """
    Check if a protein sequence is present in the training dataset.
    
    This function searches for a specific protein sequence in the training
    DataFrame and returns whether it was found.
    
    Args:
        sequence: The protein sequence to search for
        train_df: Training dataset DataFrame
        sequence_column: Name of the column containing protein sequences
        
    Returns:
        True if the sequence is found in the training set, False otherwise
        
    Examples:
        >>> train_df = pd.DataFrame({'Sequence': ['MKTV...', 'AAAL...']})
        >>> check_sequence_location('MKTV...', train_df)
        True
        
        >>> check_sequence_location('XXXX...', train_df)
        False
    """
    # Find the sequence column (case-insensitive search)
    seq_col = None
    for col in train_df.columns:
        if col.lower() == sequence_column.lower():
            seq_col = col
            break
    
    if seq_col is None:
        print(f"Error: No sequence column '{sequence_column}' found. Available columns: {train_df.columns.tolist()}")
        return False
    
    # Search for the sequence in the training dataset
    return sequence in train_df[seq_col].values



