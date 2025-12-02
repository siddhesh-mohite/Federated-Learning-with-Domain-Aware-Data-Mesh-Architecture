"""
Flower server with FedAvg strategy.
Coordinates federated learning across multiple clients.
"""

import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with logging and metrics tracking.
    
    FedAvg aggregates client updates using weighted averaging:
    θ_global = Σ(n_k / n_total) * θ_k
    
    where n_k is the number of samples at client k.
    """
    
    def __init__(self, 
                 min_fit_clients: int = 3,
                 min_evaluate_clients: int = 3,
                 min_available_clients: int = 3,
                 **kwargs):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        self.round_metrics = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        
        print(f"\n{'='*60}")
        print(f"ROUND {server_round}: Aggregating {len(results)} client updates")
        print(f"{'='*60}")
        
        # Call parent aggregation (weighted average by num_samples)
        parameters_aggregated, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log client metrics
        total_samples = 0
        total_loss = 0.0
        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics
            num_samples = fit_res.num_examples
            total_samples += num_samples
            if "train_loss" in client_metrics:
                total_loss += client_metrics["train_loss"] * num_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        round_info = {
            "round": server_round,
            "num_clients": len(results),
            "total_samples": total_samples,
            "avg_train_loss": avg_loss
        }
        self.round_metrics.append(round_info)
        
        print(f"Aggregated from {len(results)} clients, {total_samples:,} total samples")
        print(f"Average training loss: {avg_loss:.4f}")
        
        return parameters_aggregated, {"avg_train_loss": avg_loss}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        
        if not results:
            return None, {}
        
        # Weighted average of accuracy
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        for client_proxy, eval_res in results:
            num_samples = eval_res.num_examples
            total_samples += num_samples
            total_loss += eval_res.loss * num_samples
            
            if "num_correct" in eval_res.metrics:
                total_correct += eval_res.metrics["num_correct"]
            elif "accuracy" in eval_res.metrics:
                total_correct += eval_res.metrics["accuracy"] * num_samples
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        
        print(f"\nRound {server_round} Evaluation:")
        print(f"  Global Loss: {avg_loss:.4f}")
        print(f"  Global Accuracy: {avg_accuracy*100:.2f}%")
        
        # Update round metrics
        if self.round_metrics and self.round_metrics[-1]["round"] == server_round:
            self.round_metrics[-1]["eval_loss"] = avg_loss
            self.round_metrics[-1]["eval_accuracy"] = avg_accuracy
        
        return avg_loss, {"accuracy": avg_accuracy}
