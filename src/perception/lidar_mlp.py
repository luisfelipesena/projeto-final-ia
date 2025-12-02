"""
LIDAR MLP Neural Network for obstacle classification.

Multi-Layer Perceptron that classifies LIDAR sectors as
containing obstacles or being clear. This is the RNA component
required by MATA64.

Architecture:
- Input: 512 normalized LIDAR range values
- Hidden: 128 neurons with ReLU + Dropout
- Output: 9 sectors (obstacle probability per sector)
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[LIDAR_MLP] Warning: PyTorch not available, using numpy fallback")

from utils.config import LIDAR_MLP


class LidarMLP(nn.Module if TORCH_AVAILABLE else object):
    """MLP for LIDAR obstacle classification.

    Takes raw LIDAR ranges and outputs obstacle probability per sector.
    """

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_sectors: int = None,
        dropout: float = None,
    ):
        """Initialize network.

        Args:
            input_size: Number of LIDAR points (default 512)
            hidden_size: Hidden layer size (default 128)
            num_sectors: Number of output sectors (default 9)
            dropout: Dropout rate (default 0.3)
        """
        if TORCH_AVAILABLE:
            super().__init__()

        self.input_size = input_size or LIDAR_MLP.INPUT_SIZE
        self.hidden_size = hidden_size or LIDAR_MLP.HIDDEN_SIZE
        self.num_sectors = num_sectors or LIDAR_MLP.NUM_SECTORS
        self.dropout_rate = dropout or LIDAR_MLP.DROPOUT

        if TORCH_AVAILABLE:
            self._build_network()
        else:
            self.weights = None  # Placeholder for numpy weights

    def _build_network(self):
        """Build PyTorch network layers."""
        self.network = nn.Sequential(
            # Input layer -> Hidden
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            # Hidden -> Hidden
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            # Hidden -> Output
            nn.Linear(self.hidden_size // 2, self.num_sectors),
            nn.Sigmoid(),  # Output probabilities [0, 1]
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Output tensor of shape (batch, num_sectors)
        """
        return self.network(x)

    def predict(self, lidar_data: np.ndarray) -> np.ndarray:
        """Predict obstacle probabilities for LIDAR data.

        Args:
            lidar_data: Raw LIDAR ranges, shape (512,) or (batch, 512)

        Returns:
            Obstacle probabilities per sector, shape (9,) or (batch, 9)
        """
        # Normalize input
        data = self._normalize(lidar_data)

        if TORCH_AVAILABLE:
            return self._predict_torch(data)
        else:
            return self._predict_numpy(data)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize LIDAR data to [0, 1]."""
        data = np.array(data, dtype=np.float32)

        # Clamp to valid range
        data = np.clip(data, 0.01, 5.0)

        # Normalize
        data = (data - 0.01) / (5.0 - 0.01)

        return data

    def _predict_torch(self, data: np.ndarray) -> np.ndarray:
        """Predict using PyTorch."""
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Convert to tensor
        x = torch.from_numpy(data).float()

        # Inference mode
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

        return output.numpy().squeeze()

    def _predict_numpy(self, data: np.ndarray) -> np.ndarray:
        """Fallback prediction using simple threshold.

        Used when PyTorch is not available.
        """
        # Simple sector-based threshold
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        points_per_sector = self.input_size // self.num_sectors

        output = np.zeros((batch_size, self.num_sectors))
        for i in range(self.num_sectors):
            start = i * points_per_sector
            end = start + points_per_sector
            sector_data = data[:, start:end]

            # If any normalized value < 0.1 (original ~0.5m), consider obstacle
            output[:, i] = (sector_data.min(axis=1) < 0.1).astype(np.float32)

        return output.squeeze()

    def predict_binary(self, lidar_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary obstacle map.

        Args:
            lidar_data: Raw LIDAR ranges
            threshold: Probability threshold for obstacle

        Returns:
            Binary array (1 = obstacle, 0 = clear)
        """
        probs = self.predict(lidar_data)
        return (probs > threshold).astype(np.int32)

    def save(self, path: str = None) -> None:
        """Save model weights.

        Args:
            path: File path (default from config)
        """
        if not TORCH_AVAILABLE:
            print("[LIDAR_MLP] Cannot save: PyTorch not available")
            return

        path = path or LIDAR_MLP.MODEL_PATH
        torch.save(self.state_dict(), path)
        print(f"[LIDAR_MLP] Model saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load model weights.

        Args:
            path: File path (default from config)

        Returns:
            True if successful
        """
        if not TORCH_AVAILABLE:
            print("[LIDAR_MLP] Cannot load: PyTorch not available")
            return False

        path = path or LIDAR_MLP.MODEL_PATH

        try:
            self.load_state_dict(torch.load(path, map_location='cpu'))
            self.eval()
            print(f"[LIDAR_MLP] Model loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"[LIDAR_MLP] Model file not found: {path}")
            return False
        except Exception as e:
            print(f"[LIDAR_MLP] Error loading model: {e}")
            return False


def create_model() -> LidarMLP:
    """Factory function to create LIDAR MLP model."""
    return LidarMLP()


def train_model(
    model: LidarMLP,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Tuple[LidarMLP, list]:
    """Train the LIDAR MLP model.

    Args:
        model: LidarMLP instance
        train_data: Training inputs, shape (N, 512)
        train_labels: Training labels, shape (N, 9)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        (trained_model, loss_history)
    """
    if not TORCH_AVAILABLE:
        print("[LIDAR_MLP] Cannot train: PyTorch not available")
        return model, []

    # Convert to tensors
    X = torch.from_numpy(train_data.astype(np.float32))
    y = torch.from_numpy(train_labels.astype(np.float32))

    # Create data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for probabilities
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            print(f"[LIDAR_MLP] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, loss_history
