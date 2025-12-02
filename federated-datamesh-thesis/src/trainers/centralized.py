"""
Centralized trainer for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

class CentralizedTrainer:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader, test_loader, epochs=5, lr=0.01):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        print(f"🚀 Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})
            
            # Evaluation Phase
            val_acc = self.evaluate(test_loader)
            print(f"   ✅ Epoch {epoch+1} Test Accuracy: {val_acc:.2f}%")
            
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total
