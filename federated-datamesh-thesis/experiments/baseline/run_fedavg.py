"""
Run FedAvg federated learning experiment.
Compares IID and non-IID performance.
"""

import sys
import os
from pathlib import Path

# FIX: Robustly add project root to path regardless of execution directory
# Get the absolute path of this script file
script_path = Path(__file__).resolve()
# Go up 3 levels: run_fedavg.py -> baseline -> experiments -> PROJECT_ROOT
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

# FIX: Configure Ray BEFORE importing anything else
os.environ["RAY_memory"] = "2000000000"  # 2GB RAM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
import matplotlib.pyplot as plt
import json

from src.models.cnn_mnist import SimpleCNN
from src.data.mnist_loader import MNISTDataLoader
from src.federated.client import FederatedClient
from src.federated.server import FedAvgStrategy


def create_client_fn(client_loaders, device):
    """Factory function to create clients."""
    def client_fn(cid: str):
        client_id = int(cid)
        train_loader, test_loader = client_loaders[client_id]
        
        # Each client gets a fresh model
        model = SimpleCNN(num_classes=10)
        
        # Return the client wrapped in .to_client() (Required for Flower 1.6.0)
        return FederatedClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            client_id=client_id,
            device=device
        ).to_client()
    
    return client_fn


def run_federated_experiment(
    num_clients: int = 3,
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    partition_type: str = "iid",
    alpha: float = 0.5,
    device: str = "cuda"
):
    """Run a federated learning experiment."""
    
    print("=" * 60)
    print(f"FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"Partition: {partition_type.upper()}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load and partition data
    print("\nPartitioning data...")
    data_loader = MNISTDataLoader()
    
    if partition_type == "iid":
        client_loaders = data_loader.partition_iid(
            num_clients=num_clients, 
            batch_size=batch_size
        )
    else:
        client_loaders = data_loader.partition_non_iid(
            num_clients=num_clients, 
            alpha=alpha,
            batch_size=batch_size
        )
    
    # Initialize global model params
    initial_model = SimpleCNN(num_classes=10)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )
    
    # Create strategy
    strategy = FedAvgStrategy(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate
        }
    )
    
    # Create client function
    client_fn = create_client_fn(client_loaders, device)
    
    # Run simulation
    print("\nStarting federated learning simulation...")
    try:
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 4, "num_gpus": 0},
        client_resources={"num_cpus": 1, "num_gpus": 0},

    )
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return [], None
    
    return strategy.round_metrics, history


def main():
    DEVICE = "cpu" #if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {DEVICE}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Experiment 1: IID partition
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: IID PARTITION")
    print("=" * 70)
    
    iid_metrics, iid_history = run_federated_experiment(
        num_clients=3,
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        partition_type="iid",
        device=DEVICE
    )
    
    # Experiment 2: Non-IID partition
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: NON-IID PARTITION (alpha=0.5)")
    print("=" * 70)
    
    non_iid_metrics, non_iid_history = run_federated_experiment(
        num_clients=3,
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        partition_type="non_iid",
        alpha=0.5,
        device=DEVICE
    )
    
    # Save results
    results_dir = project_root / "logs/fedavg_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract accuracy history
    iid_acc = [m.get("eval_accuracy", 0) for m in iid_metrics if "eval_accuracy" in m]
    non_iid_acc = [m.get("eval_accuracy", 0) for m in non_iid_metrics if "eval_accuracy" in m]
    
    # Plot comparison
    if iid_acc or non_iid_acc:
        plt.figure(figsize=(10, 6))
        if iid_acc:
            plt.plot(range(1, len(iid_acc)+1), [a*100 for a in iid_acc], 
                     marker='o', label='IID Partition')
        if non_iid_acc:
            plt.plot(range(1, len(non_iid_acc)+1), [a*100 for a in non_iid_acc], 
                     marker='s', label='Non-IID Partition (α=0.5)')
        plt.xlabel('Communication Round')
        plt.ylabel('Global Test Accuracy (%)')
        plt.title('FedAvg: IID vs Non-IID Partition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / 'iid_vs_noniid.png', dpi=150)
        print(f"\nPlot saved to {results_dir / 'iid_vs_noniid.png'}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"IID Final Accuracy:     {iid_acc[-1]*100:.2f}%" if iid_acc else "IID: No data collected")
    print(f"Non-IID Final Accuracy: {non_iid_acc[-1]*100:.2f}%" if non_iid_acc else "Non-IID: No data collected")
    print("=" * 60)
    print("✅ Federated learning experiments complete!")


if __name__ == "__main__":
    main()
