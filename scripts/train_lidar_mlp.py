"""
Training Script for LIDAR MLP Neural Network.

Trains the obstacle detection MLP using data collected from Webots simulation.

Usage:
    python scripts/train_lidar_mlp.py --data data/lidar/*.json --epochs 100

Output:
    Trained model saved to models/lidar_mlp.pth
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    print("Error: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

from src.perception.lidar_mlp import LidarMLP, train_model


def load_data(data_files: list) -> tuple:
    """Load and combine data from multiple JSON files.

    Args:
        data_files: List of JSON file paths

    Returns:
        (X, y) numpy arrays
    """
    all_readings = []
    all_labels = []

    for filepath in data_files:
        print(f"Loading {filepath}...")

        with open(filepath, 'r') as f:
            data = json.load(f)

        readings = np.array(data['lidar_readings'], dtype=np.float32)
        labels = np.array(data['sector_labels'], dtype=np.float32)

        all_readings.append(readings)
        all_labels.append(labels)

        print(f"  - {len(readings)} samples")

    X = np.vstack(all_readings)
    y = np.vstack(all_labels)

    print(f"\nTotal samples: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    return X, y


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize LIDAR data to [0, 1] range."""
    MIN_RANGE = 0.01
    MAX_RANGE = 5.0

    X = np.clip(X, MIN_RANGE, MAX_RANGE)
    X = (X - MIN_RANGE) / (MAX_RANGE - MIN_RANGE)

    return X


def evaluate_model(model: LidarMLP, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate trained model on test set.

    Args:
        model: Trained LidarMLP
        X_test: Test inputs
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    model.eval()

    X = torch.from_numpy(X_test)
    y = torch.from_numpy(y_test)

    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()

    # Metrics
    accuracy = (predictions == y).float().mean().item()

    # Per-sector accuracy
    sector_accuracy = (predictions == y).float().mean(dim=0).numpy()

    # Confusion matrix values
    tp = ((predictions == 1) & (y == 1)).float().sum().item()
    fp = ((predictions == 1) & (y == 0)).float().sum().item()
    tn = ((predictions == 0) & (y == 0)).float().sum().item()
    fn = ((predictions == 0) & (y == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sector_accuracy': sector_accuracy.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train LIDAR MLP for obstacle detection')
    parser.add_argument('--data', type=str, default='data/lidar/*.json',
                        help='Glob pattern for training data files')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='models/lidar_mlp.pth',
                        help='Output model path')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data for testing')

    args = parser.parse_args()

    # Find data files
    data_files = glob(args.data)
    if not data_files:
        print(f"Error: No data files found matching '{args.data}'")
        print("Run collect_lidar_data.py in Webots first to collect training data.")
        sys.exit(1)

    print(f"Found {len(data_files)} data files")

    # Load and preprocess data
    X, y = load_data(data_files)
    X = normalize_data(X)

    # Split data
    n_test = int(len(X) * args.test_split)
    n_train = len(X) - n_test

    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Check class balance
    obstacle_ratio = y_train.mean()
    print(f"Obstacle ratio in training: {obstacle_ratio:.2%}")

    # Create model
    model = LidarMLP()
    print(f"\nModel architecture:")
    print(model.network)

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    model, loss_history = train_model(
        model,
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1']:.2%}")
    print(f"\n  Per-sector accuracy:")
    for i, acc in enumerate(metrics['sector_accuracy']):
        print(f"    Sector {i}: {acc:.2%}")

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)

    print(f"\nModel saved to {args.output}")

    # Save training history
    history_path = args.output.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss_history': loss_history,
            'final_metrics': metrics,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
            }
        }, f, indent=2)

    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
