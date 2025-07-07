"""
Classification Module

This module contains models and functions for protein classification tasks,
including kingdom classification and drug mining using ESM3 embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for protein classification.
    """
    
    def __init__(
        self, 
        input_size: int, 
        num_classes: int, 
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_size: Size of input features (embedding dimension)
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ProteinClassifier:
    """
    A complete protein classification system using ESM3 embeddings.
    """
    
    def __init__(
        self, 
        input_size: int,
        num_classes: int,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the protein classifier.
        
        Args:
            input_size: Size of input features (embedding dimension)
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            device: Device to use (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size, num_classes, hidden_layers, dropout_rate)
        self.model.to(self.device)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def prepare_data(
        self, 
        embeddings: np.ndarray, 
        labels: Union[List, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training and testing.
        
        Args:
            embeddings: Embedding matrix
            labels: Target labels
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) tensors
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y_encoded, test_size=test_size, random_state=random_state
        )
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to show progress
            
        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
                else:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%")
        
        self.is_fitted = True
        return history
    
    def evaluate(
        self, 
        X_test: torch.Tensor, 
        y_test: torch.Tensor, 
        criterion: Optional[nn.Module] = None
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            criterion: Loss function (optional)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
            
            accuracy = 100 * (predicted == y_test).sum().item() / y_test.size(0)
            
            if criterion:
                loss = criterion(outputs, y_test).item()
            else:
                loss = 0.0
        
        return loss, accuracy
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_detailed_metrics(
        self, 
        X_test: torch.Tensor, 
        y_test: torch.Tensor,
        plot_confusion_matrix: bool = True
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get detailed evaluation metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            plot_confusion_matrix: Whether to plot confusion matrix
            
        Returns:
            Dictionary with detailed metrics
        """
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Convert to numpy for sklearn metrics
        y_true = y_test.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate per-class metrics
        class_accuracies = {}
        if self.label_encoder is not None:
            # Use actual class names from label encoder
            class_names = self.label_encoder.classes_
        else:
            # Use meaningful names for binary/numeric classification
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2 and set(unique_classes) == {0, 1}:
                # Binary drug classification case
                class_names = ["Non-druggable", "Druggable"]
            else:
                # Generic case
                class_names = [f"Class_{i}" for i in unique_classes]
        
        for i, class_label in enumerate(class_names):
            class_indices = (y_true == i)
            if class_indices.sum() > 0:
                class_acc = accuracy_score(y_true[class_indices], y_pred[class_indices])
                class_accuracies[class_label] = class_acc
        
        # Calculate AUC (for multiclass)
        try:
            # Determine number of classes
            if self.label_encoder is not None:
                num_classes = len(self.label_encoder.classes_)
            else:
                num_classes = len(np.unique(y_true))
            
            if num_classes > 2:
                y_true_binarized = label_binarize(y_true, classes=range(num_classes))
                auc = roc_auc_score(y_true_binarized, y_pred_proba, average='weighted', multi_class='ovr')
            else:
                # Binary classification
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            warnings.warn(f"Could not calculate AUC: {str(e)}")
            auc = 0.0
        
        # Plot confusion matrix
        if plot_confusion_matrix:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            # Set axis labels based on whether we have a label encoder
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
            else:
                unique_classes = np.unique(y_true)
                if len(unique_classes) == 2 and set(unique_classes) == {0, 1}:
                    # Binary drug classification case
                    class_names = ["Non-druggable", "Druggable"]
                else:
                    # Generic case
                    class_names = [f"Class_{i}" for i in unique_classes]
            
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }


class KingdomClassifier:
    """
    Kingdom classification using ESM3 embeddings.
    
    This class handles unified datasets and splits them internally.
    Designed for categorical kingdom labels like 'bacteria', 'eukaryota', 'archaea'.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the kingdom classifier.
        
        Args:
            hidden_layers: Hidden layer sizes
            dropout_rate: Dropout rate for regularization
            device: Device to use (cuda/cpu)
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
    
    def train(
        self,
        embeddings: np.ndarray,
        labels: Union[List, np.ndarray],
        test_size: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        random_state: int = 42
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Train the kingdom classifier.
        
        Args:
            embeddings: Protein embeddings
            labels: Kingdom labels (e.g., 'bacteria', 'eukaryota', 'archaea')
            test_size: Test set size
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            random_state: Random state
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Kingdom Classification Model...")
        
        # Encode labels (always needed for kingdom classification)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Initialize classifier
        input_size = embeddings.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        self.classifier = ProteinClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        self.classifier.label_encoder = label_encoder
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.classifier.prepare_data(
            embeddings, labels, test_size=test_size, random_state=random_state
        )
        
        # Train model
        history = self.classifier.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=30
        )
        
        # Evaluate model
        metrics = self.classifier.get_detailed_metrics(X_test, y_test)
        
        print(f"\nKingdom Classification Results:")
        print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final Test F1 Score: {metrics['f1_score']:.4f}")
        print(f"Final Test AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.classifier is None:
            raise ValueError("Model must be trained before making predictions")
        X = torch.tensor(embeddings, dtype=torch.float32)
        return self.classifier.predict(X)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.classifier is None:
            raise ValueError("Model must be trained before making predictions")
        X = torch.tensor(embeddings, dtype=torch.float32)
        return self.classifier.predict_proba(X)


class DrugMinerClassifier:
    """
    Drug miner classification using ESM3 embeddings.
    
    This class handles pre-split training and testing datasets.
    Designed for binary druggability labels (0/1).
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the drug miner classifier.
        
        Args:
            hidden_layers: Hidden layer sizes
            dropout_rate: Dropout rate for regularization
            device: Device to use (cuda/cpu)
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
    
    def train(
        self,
        train_embeddings: np.ndarray,
        train_labels: Union[List, np.ndarray],
        test_embeddings: np.ndarray,
        test_labels: Union[List, np.ndarray],
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        random_state: int = 42
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Train the drug miner classifier with pre-split data.
        
        Args:
            train_embeddings: Training protein embeddings
            train_labels: Training druggability labels (0/1)
            test_embeddings: Test protein embeddings
            test_labels: Test druggability labels (0/1)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            random_state: Random state
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Drug Miner Classification Model...")
        print(f"Training samples: {train_embeddings.shape[0]}, Test samples: {test_embeddings.shape[0]}")
        
        # Convert labels to numpy arrays and ensure they're integers
        train_labels = np.array(train_labels, dtype=int)
        test_labels = np.array(test_labels, dtype=int)
        
        print(f"Unique training labels: {np.unique(train_labels)}")
        print(f"Unique test labels: {np.unique(test_labels)}")
        
        # Initialize classifier
        input_size = train_embeddings.shape[1]
        num_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
        
        self.classifier = ProteinClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        # No label encoder needed for binary classification
        self.classifier.label_encoder = None
        self.classifier.is_fitted = False
        
        # Convert to tensors
        X_train = torch.tensor(train_embeddings, dtype=torch.float32)
        X_test = torch.tensor(test_embeddings, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.long)
        y_test = torch.tensor(test_labels, dtype=torch.long)
        
        # Train model
        history = self.classifier.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=30
        )
        
        # Evaluate model
        metrics = self.classifier.get_detailed_metrics(X_test, y_test)
        
        print(f"\nDrug Miner Classification Results:")
        print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final Test F1 Score: {metrics['f1_score']:.4f}")
        print(f"Final Test AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.classifier is None:
            raise ValueError("Model must be trained before making predictions")
        X = torch.tensor(embeddings, dtype=torch.float32)
        return self.classifier.predict(X)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.classifier is None:
            raise ValueError("Model must be trained before making predictions")
        X = torch.tensor(embeddings, dtype=torch.float32)
        return self.classifier.predict_proba(X) 