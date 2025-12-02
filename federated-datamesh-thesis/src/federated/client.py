"""
Flower client for federated learning.
Each client represents a data owner (e.g., hospital, bank, factory).
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class FederatedClient(fl.client.NumPyClient):
    """
    Flower client for federated learning.
    
    Each client:
    - Holds private local data
    - Trains model locally
    - Sends only model updates to server (not data)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 client_id: int,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration (epochs, lr, etc.)
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Get training config
        local_epochs = config.get("local_epochs", 3)
        learning_rate = config.get("learning_rate", 0.01)
        
        # Create optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Local training
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            total_loss += avg_epoch_loss
            print(f"  [Client {self.client_id}] Epoch {epoch+1}/{local_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Return updated parameters
        updated_params = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)
        
        metrics = {
            "client_id": self.client_id,
            "train_loss": total_loss / local_epochs,
            "num_samples": num_samples
        }
        
        return updated_params, num_samples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = correct / total
        return float(loss / len(self.test_loader)), total, {"accuracy": accuracy}
