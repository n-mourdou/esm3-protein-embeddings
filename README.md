# ESM3 Protein Embeddings for Downstream Tasks

A comprehensive analysis demonstrating how to leverage ESM3 (Evolutionary Scale Modeling) embeddings for biological sequence analysis and classification tasks. This project explores the power of protein language models in capturing functional and evolutionary patterns without explicit supervision.

## Project Overview

This project demonstrates ESM3's capabilities through two main case studies:

### 1. **Contextual Embeddings Analysis**
- **Functional Separation**: Visualization showing how proteins with different functions are spatially separated in ESM3's embedding space
- **Taxonomic Clustering**: Analysis of UniRef50 proteins annotated by kingdom (Bacteria, Archaea, Eukaryota)

### 2. **Druggability Prediction**
- Binary classification of proteins as druggable vs non-druggable using the Jamali et al. benchmark dataset
- Zero-shot visualization and supervised learning approaches
- Generalization testing on clinically relevant targets (β-Catenin and β2-Adrenergic Receptor)

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for ESM3 model)

### Installation

```bash
# Clone the repository
git clone https://github.com/n-mourdou/esm3-protein-embeddings.git
cd esm3-protein-embeddings

# Install dependencies with Poetry
poetry env use 3.11
poetry install

# Activate the virtual environment
poetry env activate
```

### Run the Analysis

Open and execute the main notebook:

```bash
jupyter notebook ESM3_protein_analysis.ipynb
```

## Repository Structure

```
esm3-protein-embeddings/
├── ESM3_protein_analysis.ipynb    # Main analysis notebook
├── src/                           # Modular source code
│   ├── __init__.py               # Package initialization
│   ├── esm3_utils.py             # ESM3 model utilities and embedding generation
│   ├── data_processing.py        # Data handling and preprocessing functions
│   └── classification.py         # ML models and training pipelines
├── data/                          # Datasets
│   ├── kingdom_classification/    # UniRef50 taxonomic data
│   │   ├── uniref_50_processed_3k.csv
│   │   └── embeddings_uni_ref_50_kingdom_labelled.csv
│   └── drug_miner/               # Druggability prediction datasets
│       ├── positive_train_sequence.fasta
│       ├── negative_train_sequence.fasta
│       ├── positive_test_sequence.fasta
│       ├── negative_test_sequence.fasta
│       ├── training_set.csv
│       ├── test_set.csv
│       ├── train_embeddings_with_labels.csv
│       ├── test_embeddings_with_labels.csv
        ├── β2_Adrenergic_Receptor.jpeg   # Protein structure visualization
        └── β_Catenin.jpeg                # Protein structure visualization
├── pyproject.toml                # Project dependencies and configuration
└── README.md                     # This file
```


## Key Results

### Functional Separation (Sanity Check)
- **Clear clustering**: Transport/oxygen-related proteins (hemoglobin, myoglobin) separate from glucose/calcium regulation proteins
- **Zero-shot performance**: No fine-tuning required for meaningful biological separation

### Kingdom Classification
- **Test Accuracy**: 88.11%
- **F1 Score**: 87.62%
- **AUC**: 94.53%
- Clear separation of kingdoms in t-SNE visualization despite class imbalance (Archaea underrepresented)

### Druggability Prediction
- **Test Accuracy**: 91.21%
- **F1 Score**: 91.22%
- **AUC**: 96.13%
- Successful generalization to clinically relevant targets:
  - β-Catenin: Correctly predicted as non-druggable (confidence: 0.128)
  - β2-Adrenergic Receptor: Correctly predicted as druggable (confidence: 0.961)


## Core Modules

### `esm3_utils.py`
- `ESM3EmbeddingGenerator`: Main class for generating embeddings
- `embeddings_to_matrix()`: Utility for converting embeddings to matrices
- Memory management and GPU optimization

### `data_processing.py`
- `process_fasta_file()`: FASTA file parsing with label assignment
- `map_taxid_to_kingdom()`: Taxonomic classification using NCBI TaxID
- `add_human_classification()`: Species identification from protein descriptions
- `check_sequence_location()`: Training set membership verification

### `classification.py`
- `KingdomClassifier`: MLP for taxonomic classification
- `DrugMinerClassifier`: MLP for druggability prediction
- Comprehensive training pipelines with early stopping and metrics

## Dataset Information

### Kingdom Classification
- **Source**: UniRef50 subset (3,000 sequences)
- **Classes**: Bacteria (~1,600), Eukaryota (~1,100), Archaea (~150)
- **Features**: Full protein sequences with NCBI TaxID mapping

### Druggability Prediction
- **Source**: Jamali et al. benchmark dataset
- **Training**: 2,064 sequences (balanced druggable/non-druggable)
- **Testing**: 478 sequences
- **Species**: Heavily human-biased (expected for drug discovery)
- **Sequence lengths**: Most <1,000 residues, safe for ESM3 processing

## Key Insights

1. **Zero-shot capabilities**: ESM3 embeddings naturally cluster proteins by function and taxonomy without any task-specific training
2. **Evolutionary encoding**: Kingdom-level evolutionary relationships are captured in the representation space
3. **Druggability signals**: Subtle structural/functional properties relevant to drug targeting are encoded
4. **Generalization**: Models trained on curated datasets can make biologically plausible predictions on new targets
5. **Practical applicability**: High performance with simple classifiers suggests robust feature representations


## Data Sources

- **UniRef50**: Representative protein sequences with taxonomic annotations
- **Jamali et al. Dataset**: Curated druggable vs non-druggable protein benchmark from **DrugBank** (druggable) and **Swiss-Prot/TTD** (non-druggable).


## Acknowledgments

- **[EvolotionaryScale](https://www.evolutionaryscale.ai/)** for the ESM3 protein language model.
- **[Jamali et al.](https://pubmed.ncbi.nlm.nih.gov/26821132/)** for the druggability benchmark dataset
- **[UniProt](https://www.uniprot.org/)** for the UniRef50 database
- **[ETE Toolkit](https://etetoolkit.org/)** for taxonomic tree operations 