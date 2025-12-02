"""
MNIST data loader with IID and non-IID partitioning support.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple

class MNISTDataLoader:
    def __init__(self, data_dir: str = "./data/mnist"):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Download data once
        self.train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        
    def load_centralized(self, batch_size: int = 32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader
    
    def partition_iid(self, num_clients: int = 3,
                      batch_size: int = 32):
        """IID partition of MNIST among clients."""
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        # Shuffle and split train indices
        train_indices = np.random.permutation(len(self.train_dataset))
        train_splits = np.array_split(train_indices, num_clients)

        # Shuffle and split test indices
        test_indices = np.random.permutation(len(self.test_dataset))
        test_splits = np.array_split(test_indices, num_clients)

        client_loaders = []
        for train_idx, test_idx in zip(train_splits, test_splits):
            train_subset = Subset(self.train_dataset, train_idx)
            test_subset = Subset(self.test_dataset, test_idx)

            train_loader = DataLoader(train_subset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2)
            test_loader = DataLoader(test_subset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=2)
            client_loaders.append((train_loader, test_loader))

        return client_loaders

    def partition_non_iid(self, num_clients: int = 3,
                          alpha: float = 0.5,
                          batch_size: int = 32):
        """Non‑IID partition of MNIST using Dirichlet distribution."""
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        num_classes = 10

        train_labels = np.array(
            [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        )
        test_labels = np.array(
            [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        )

        # Train splits
        train_client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = np.where(train_labels == c)[0]
            np.random.shuffle(idx)
            props = np.random.dirichlet(np.repeat(alpha, num_clients))
            props = (props * len(idx)).astype(int)
            props[-1] = len(idx) - props[:-1].sum()
            start = 0
            for k, cnt in enumerate(props):
                train_client_indices[k].extend(idx[start:start+cnt])
                start += cnt

        # Test splits (same idea)
        test_client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = np.where(test_labels == c)[0]
            np.random.shuffle(idx)
            props = np.random.dirichlet(np.repeat(alpha, num_clients))
            props = (props * len(idx)).astype(int)
            props[-1] = len(idx) - props[:-1].sum()
            start = 0
            for k, cnt in enumerate(props):
                test_client_indices[k].extend(idx[start:start+cnt])
                start += cnt

        client_loaders = []
        for train_idx, test_idx in zip(train_client_indices, test_client_indices):
            train_subset = Subset(self.train_dataset, train_idx)
            test_subset = Subset(self.test_dataset, test_idx)

            train_loader = DataLoader(train_subset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2)
            test_loader = DataLoader(test_subset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=2)
            client_loaders.append((train_loader, test_loader))

        return client_loaders


# Quick test
if __name__ == "__main__":
    loader = MNISTDataLoader()
    tr, te = loader.load_centralized()
    print(f"✅ Data loaded! Train batches: {len(tr)}, Test batches: {len(te)}")
