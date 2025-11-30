#!/usr/bin/env python3
"""
Train SimpleLIDARMLP for obstacle detection.

Usage:
    python scripts/train_lidar_mlp.py --epochs 100
    python scripts/train_lidar_mlp.py --data data/lidar_training/ --epochs 50

If no training data exists, generates synthetic data for initial training.
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.models.simple_lidar_mlp import SimpleLIDARMLP


def generate_synthetic_data(num_samples: int = 1000, lidar_points: int = 512,
                            num_sectors: int = 9) -> tuple:
    """
    Generate synthetic LIDAR training data with auto-labels.

    Creates diverse scenarios: clear, partial obstacles, full obstacles.
    """
    print(f"Generating {num_samples} synthetic training samples...")

    X = []
    y = []

    for i in range(num_samples):
        scenario = np.random.choice(['clear', 'partial', 'dense'])

        if scenario == 'clear':
            # All clear - distances 3-5m
            scan = np.random.uniform(0.6, 1.0, lidar_points)
        elif scenario == 'partial':
            # Some sectors have obstacles
            scan = np.random.uniform(0.6, 1.0, lidar_points)
            num_obstacles = np.random.randint(1, 4)
            for _ in range(num_obstacles):
                start = np.random.randint(0, lidar_points - 50)
                width = np.random.randint(20, 80)
                dist = np.random.uniform(0.05, 0.4)  # 0.25-2m obstacle
                scan[start:start + width] = dist
        else:  # dense
            # Many obstacles
            scan = np.random.uniform(0.1, 0.5, lidar_points)

        X.append(scan)

        # Auto-label based on threshold
        labels = SimpleLIDARMLP.create_labels_from_scan(scan, threshold=0.5, num_sectors=num_sectors)
        y.append(labels)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_real_data(data_dir: Path, target_size: int = 512) -> tuple:
    """
    Load real LIDAR data from JSON files.

    Expected format: {"scan": [distances...], "timestamp": ...}
    """
    print(f"Loading data from {data_dir}...")

    X = []
    y = []

    json_files = list(data_dir.glob("*.json"))
    print(f"  Found {len(json_files)} JSON files")

    for f in json_files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            if 'scan' not in data:
                continue

            scan = np.array(data['scan'], dtype=np.float32)

            # Normalize and resize
            scan = np.nan_to_num(scan, nan=5.0, posinf=5.0, neginf=0.0)
            scan = np.clip(scan, 0, 5.0) / 5.0

            if len(scan) != target_size:
                indices = np.linspace(0, len(scan) - 1, target_size).astype(int)
                scan = scan[indices]

            X.append(scan)

            # Auto-label
            labels = SimpleLIDARMLP.create_labels_from_scan(scan, threshold=0.5)
            y.append(labels)

        except Exception as e:
            print(f"  Warning: Failed to load {f.name}: {e}")

    if not X:
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 100,
                batch_size: int = 32, lr: float = 0.001,
                validation_split: float = 0.2) -> SimpleLIDARMLP:
    """Train SimpleLIDARMLP model."""

    # Split train/val
    n_val = int(len(X) * validation_split)
    indices = np.random.permutation(len(X))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = torch.tensor(X[train_indices])
    y_train = torch.tensor(y[train_indices])
    X_val = torch.tensor(X[val_indices])
    y_val = torch.tensor(y[val_indices])

    print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleLIDARMLP(input_size=X.shape[1], num_sectors=y.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training for {epochs} epochs...")

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

            # Accuracy (threshold at 0.5)
            val_pred = (val_output > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train SimpleLIDARMLP')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default='models/lidar_mlp.pth',
                        help='Output model path')
    parser.add_argument('--synthetic-samples', type=int, default=2000,
                        help='Number of synthetic samples if no real data')
    args = parser.parse_args()

    # Load or generate data
    X, y = None, None

    if args.data:
        data_dir = Path(args.data)
        if data_dir.exists():
            X, y = load_real_data(data_dir)

    if X is None or len(X) == 0:
        print("No real data found. Using synthetic data.")
        X, y = generate_synthetic_data(num_samples=args.synthetic_samples)

    print(f"Dataset: X={X.shape}, y={y.shape}")

    # Train
    model = train_model(X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nModel saved to {output_path}")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_input = torch.randn(1, X.shape[1])
    with torch.no_grad():
        output = model(test_input)
    print(f"  Test output: {output.squeeze().numpy().round(2)}")
    print("  Done!")


if __name__ == "__main__":
    main()
