#!/usr/bin/env python3
"""
Standalone LIDAR MLP Training Script.

Trains without complex import dependencies.
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LidarMLP(nn.Module):
    """MLP for LIDAR obstacle classification."""

    def __init__(self, input_size: int = 512, hidden_size: int = 128, num_sectors: int = 9, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_sectors),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_data(data_files: list) -> tuple:
    """Load and combine data from multiple JSON files."""
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
    return X, y


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize LIDAR data to [0, 1] range."""
    X = np.clip(X, 0.01, 5.0)
    X = (X - 0.01) / (5.0 - 0.01)
    return X


def train_model(model, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
    """Train the model."""
    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


def evaluate_model(model, X_test, y_test):
    """Evaluate on test set."""
    model.eval()
    X = torch.from_numpy(X_test)
    y = torch.from_numpy(y_test)

    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()

    accuracy = (predictions == y).float().mean().item()

    tp = ((predictions == 1) & (y == 1)).float().sum().item()
    fp = ((predictions == 1) & (y == 0)).float().sum().item()
    fn = ((predictions == 0) & (y == 1)).float().sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/lidar/*.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='models/lidar_mlp.pth')
    args = parser.parse_args()

    # Find data
    data_files = glob(args.data)
    if not data_files:
        print(f"No data files found: {args.data}")
        sys.exit(1)

    print(f"Found {len(data_files)} data files")

    # Load and prep
    X, y = load_data(data_files)
    X = normalize_data(X)

    # Split
    n_test = int(len(X) * 0.2)
    indices = np.random.permutation(len(X))
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model = LidarMLP()
    print(f"\nTraining for {args.epochs} epochs...")

    loss_history = train_model(model, X_train, y_train, args.epochs, args.batch_size, args.lr)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\n=== Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nModel saved to {args.output}")

    # Save history
    history_path = args.output.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({'loss_history': loss_history, 'metrics': metrics}, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
