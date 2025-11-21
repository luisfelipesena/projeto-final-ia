# Quickstart Guide: Neural Network Perception

**Phase 2 - Implementation Guide**
**Date**: 2025-11-21

This guide covers data collection, training, and deployment of LIDAR and camera neural networks for YouBot perception.

---

## Prerequisites

### Environment Setup

```bash
# Create Python virtual environment (if not exists)
python3 -m venv webots_env
source webots_env/bin/activate  # macOS/Linux

# Install dependencies
pip install torch==2.0.1 torchvision==0.15.2
pip install numpy scipy opencv-python matplotlib
pip install scikit-learn tensorboard pytest
```

### Project Structure

```
projeto-final-ia/
├── src/perception/              # Implementation code
│   ├── lidar_processor.py
│   ├── cube_detector.py
│   ├── perception_system.py
│   └── training/
│       ├── train_lidar.py
│       ├── train_camera.py
│       ├── data_loader.py
│       └── augmentation.py
├── data/                        # Training data
│   ├── lidar/
│   └── camera/
├── models/                      # Trained models
│   ├── lidar_net.pt
│   ├── cube_detector.pt
│   └── metadata.json
├── notebooks/                   # Training experiments
│   ├── 02_lidar_training.ipynb
│   └── 03_camera_training.ipynb
└── logs/                        # Training logs
```

---

## Part 1: LIDAR Neural Network

### Step 1.1: Data Collection (Day 1-2)

**Goal:** Collect 1000+ LIDAR scans from diverse arena positions

**Script:** `scripts/collect_lidar_data.py`

```python
#!/usr/bin/env python3
"""Collect LIDAR training data from Webots simulation"""

from controller import Robot
import numpy as np
import json
from pathlib import Path

def collect_lidar_data(num_samples=1000):
    """
    Collect LIDAR scans with obstacle labels

    Strategy:
        - Spawn robot at random positions (20+ locations)
        - Record LIDAR scan (667 points)
        - Use GPS to label obstacles (known arena positions)
        - Save to data/lidar/
    """
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    lidar = robot.getDevice("lidar")
    lidar.enable(timestep)

    gps = robot.getDevice("gps")  # GPS allowed for training
    gps.enable(timestep)

    data_dir = Path("data/lidar")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "scans").mkdir(exist_ok=True)
    (data_dir / "labels").mkdir(exist_ok=True)

    for i in range(num_samples):
        robot.step(timestep)

        # Get LIDAR scan
        ranges = np.array(lidar.getRangeImage())

        # Get robot position
        position = gps.getValues()

        # Label obstacles (9 sectors)
        labels = label_obstacles(ranges, position, arena_map)

        # Save scan
        np.save(data_dir / "scans" / f"scan_{i:04d}.npy", ranges)

        # Save label
        label_data = {
            "sectors": labels.tolist(),
            "position": position,
            "timestamp": robot.getTime()
        }
        with open(data_dir / "labels" / f"scan_{i:04d}.json", "w") as f:
            json.dump(label_data, f)

        # Move to new position every 50 scans
        if i % 50 == 0:
            teleport_robot_random()

        print(f"Collected {i+1}/{num_samples} scans")

    print(f"✅ Data collection complete: {num_samples} scans")

if __name__ == "__main__":
    collect_lidar_data(num_samples=1200)  # 1200 → split 840/180/180
```

**Output:**
- `data/lidar/scans/scan_XXXX.npy`: 1200 LIDAR scans
- `data/lidar/labels/scan_XXXX.json`: Obstacle labels (9 binary flags)

---

### Step 1.2: Training (Day 3-4)

**Goal:** Train hybrid MLP + 1D-CNN model (>90% accuracy)

**Notebook:** `notebooks/02_lidar_training.ipynb`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.perception.training.data_loader import LIDARDataset
from src.perception.training.augmentation import LIDARAugmentation

# 1. Load dataset
dataset = LIDARDataset(data_dir="data/lidar", augment=True)

# 2. Train/Val/Test split (70/15/15)
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# 3. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 4. Define model
class HybridLIDARNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D-CNN branch (667 → 64 features)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # MLP classifier (70 → 9)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 6, 128),  # 64 CNN + 6 hand-crafted
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 9)  # 9 sectors
        )

    def forward(self, cnn_features, hand_features):
        cnn_out = self.cnn(cnn_features.unsqueeze(1))  # [B, 1, 667]
        combined = torch.cat([cnn_out, hand_features], dim=1)  # [B, 70]
        return self.mlp(combined)  # [B, 9]

model = HybridLIDARNet()

# 5. Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# 6. Train (100 epochs, ~15 min)
best_val_acc = 0.0
for epoch in range(100):
    # Train epoch
    model.train()
    train_loss = 0.0
    for scans, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(scans, extract_hand_features(scans))
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_acc = evaluate(model, val_loader)
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/lidar_net_best.pth")

    print(f"Epoch {epoch+1}/100: Loss={train_loss/len(train_loader):.4f}, Val_Acc={val_acc:.2%}")

# 7. Test set evaluation
model.load_state_dict(torch.load("models/lidar_net_best.pth"))
test_acc = evaluate(model, test_loader)
print(f"✅ Test Accuracy: {test_acc:.2%} (target: >90%)")

# 8. Export TorchScript
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("models/lidar_net.pt")
print("✅ Model saved: models/lidar_net.pt")
```

**Expected Results:**
- Training time: 10-15 minutes (100 epochs)
- Validation accuracy: 92-95%
- Test accuracy: >90% (SC-001)
- Model size: ~1MB

---

### Step 1.3: Integration (Day 5)

**Goal:** Integrate LIDAR processor into Webots controller

**File:** `IA_20252/controllers/youbot/youbot.py`

```python
from controller import Robot
from src.perception.lidar_processor import LIDARProcessor

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)

        # Load LIDAR neural network
        self.lidar_processor = LIDARProcessor(
            model_path="models/lidar_net.pt",
            device="cpu"
        )
        print("✅ LIDAR processor initialized")

    def run(self):
        while self.robot.step(self.timestep) != -1:
            # Get LIDAR data
            ranges = self.lidar.getRangeImage()

            # Process with neural network
            obstacle_map = self.lidar_processor.process(ranges)

            # Query results
            nearest_sector, nearest_dist = (
                obstacle_map.min_sector,
                obstacle_map.min_distance
            )

            print(f"Nearest obstacle: sector {nearest_sector}, {nearest_dist:.2f}m")

            # TODO Phase 3: Feed to fuzzy controller

if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
```

**Validation:**
- Run simulation, verify inference <100ms (SC-003)
- Check obstacle detection accuracy visually
- Ensure no crashes during 5-min run (SC-008)

---

## Part 2: Camera CNN

### Step 2.1: Data Collection (Day 6-7)

**Goal:** Collect 500+ cube images with color labels

**Script:** `scripts/collect_camera_data.py`

```python
#!/usr/bin/env python3
"""Collect camera training data from Webots simulation"""

from controller import Robot, Supervisor
import numpy as np
from PIL import Image
from pathlib import Path
import json

def collect_camera_data(num_samples=600):
    """
    Collect camera images with cube color labels

    Strategy:
        - Use supervisor to spawn random cubes
        - Capture camera image
        - Use recognitionColors to label cube color
        - Save image + label
    """
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    camera = supervisor.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    data_dir = Path("data/camera")
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    color_counts = {"green": 0, "blue": 0, "red": 0}

    for i in range(num_samples):
        supervisor.step(timestep)

        # Get camera image
        image_data = camera.getImageArray()
        image = Image.fromarray(np.uint8(image_data))

        # Get recognition objects
        objects = camera.getRecognitionObjects()

        if objects:
            # Get cube color from recognition
            obj = objects[0]
            color = get_color_from_recognition(obj.get_colors())

            # Balance dataset
            if color_counts[color] >= num_samples // 3:
                continue  # Skip if color has enough samples

            # Save image
            filename = f"{color}_{color_counts[color]:04d}"
            image.save(data_dir / "images" / f"{filename}.png")

            # Save label
            label_data = {
                "color": color,
                "bbox": obj.get_position_on_image(),
                "timestamp": supervisor.getTime()
            }
            with open(data_dir / "labels" / f"{filename}.json", "w") as f:
                json.dump(label_data, f)

            color_counts[color] += 1
            print(f"Collected: {sum(color_counts.values())}/{num_samples} "
                  f"(G:{color_counts['green']}, B:{color_counts['blue']}, R:{color_counts['red']})")

        # Respawn cubes every 20 captures
        if i % 20 == 0:
            respawn_cubes()

    print(f"✅ Data collection complete: {sum(color_counts.values())} images")

if __name__ == "__main__":
    collect_camera_data(num_samples=600)  # 600 → split 420/90/90
```

**Output:**
- `data/camera/images/{color}_XXXX.png`: 600 RGB images (balanced across colors)
- `data/camera/labels/{color}_XXXX.json`: Color + bounding box

---

### Step 2.2: Training (Day 8-9)

**Goal:** Train custom CNN (>95% accuracy)

**Notebook:** `notebooks/03_camera_training.ipynb`

```python
import torch
import torch.nn as nn
from torchvision import transforms
from src.perception.training.data_loader import CameraDataset

# 1. Data augmentation
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),
    transforms.ToTensor(),
])

# 2. Load dataset
train_dataset = CameraDataset("data/camera", split="train", transform=train_transform)
val_dataset = CameraDataset("data/camera", split="val", transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 3. Define model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CustomCNN(num_classes=3)

# 4. Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(50):
    # Train + validate
    train_epoch(model, train_loader, criterion, optimizer)
    val_acc = validate(model, val_loader)
    scheduler.step()
    print(f"Epoch {epoch+1}/50: Val_Acc={val_acc:.2%}")

# 5. Test
test_acc = test_per_class(model, test_loader)
print(f"✅ Test Accuracy: Green={test_acc['green']:.2%}, Blue={test_acc['blue']:.2%}, Red={test_acc['red']:.2%}")

# 6. Export
scripted_model = torch.jit.script(model)
scripted_model.save("models/cube_detector.pt")
```

**Expected Results:**
- Training time: 10-15 minutes (50 epochs)
- Validation accuracy: 93-96%
- Test accuracy per color: >95% (SC-002)

---

### Step 2.3: Integration (Day 10)

**File:** `IA_20252/controllers/youbot/youbot.py`

```python
from src.perception.cube_detector import CubeDetector

class YouBotController:
    def __init__(self):
        # ... (previous LIDAR setup)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # Load camera CNN
        self.cube_detector = CubeDetector(
            model_path="models/cube_detector.pt",
            device="cpu"
        )
        print("✅ Cube detector initialized")

    def run(self):
        while self.robot.step(self.timestep) != -1:
            # LIDAR processing
            # ...

            # Camera processing
            image = self.camera.getImageArray()
            cubes = self.cube_detector.detect(image)

            for cube in cubes:
                if cube.is_valid(min_confidence=0.7):
                    print(f"Detected {cube.color} cube at {cube.distance:.2f}m")
```

---

## Part 3: Unified Perception

**File:** `IA_20252/controllers/youbot/youbot.py` (final version)

```python
from src.perception.perception_system import PerceptionSystem

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Sensors
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # Unified perception system
        self.perception = PerceptionSystem(
            lidar_model_path="models/lidar_net.pt",
            camera_model_path="models/cube_detector.pt"
        )
        self.perception.enable_logging("logs/perception.log")
        print("✅ Perception system ready")

    def run(self):
        while self.robot.step(self.timestep) != -1:
            # Get sensor data
            lidar_ranges = self.lidar.getRangeImage()
            camera_image = self.camera.getImageArray()

            # Unified perception
            world_state = self.perception.update(lidar_ranges, camera_image)

            # Query world state
            sector, dist = self.perception.get_nearest_obstacle()
            green_cube = self.perception.get_nearest_cube(color="green")

            print(f"Obstacles: sector {sector} @ {dist:.2f}m | "
                  f"Green cube: {green_cube.distance if green_cube else 'N/A'}m")

            # TODO Phase 3: Fuzzy controller
```

---

## Testing & Validation

### Success Criteria Checklist

**LIDAR (SC-001, SC-003, SC-006):**
- [ ] Test accuracy >90% on held-out set
- [ ] Inference time <100ms (average over 100 scans)
- [ ] Detect all obstacles in 360° rotation

**Camera (SC-002, SC-004, SC-005, SC-007):**
- [ ] Test accuracy >95% per color
- [ ] Inference FPS >10
- [ ] False positive rate <5%
- [ ] Correctly identify all 3 colors at 1m

**Integration (SC-008, SC-010, SC-011):**
- [ ] Run 5 minutes without crashes
- [ ] Models load successfully (<50MB each)
- [ ] Logs saved with timestamps

### Run Tests

```bash
# Unit tests
pytest tests/perception/ -v

# Accuracy validation
python scripts/validate_lidar_accuracy.py    # SC-001
python scripts/validate_camera_accuracy.py   # SC-002

# Performance benchmarks
python scripts/benchmark_lidar_speed.py      # SC-003
python scripts/benchmark_camera_speed.py     # SC-004

# Integration test
python scripts/test_perception_integration.py  # SC-008
```

---

## Troubleshooting

**LIDAR accuracy <90%:**
- Collect more data (aim for 1500+ scans)
- Increase augmentation strength (noise σ=0.02)
- Try longer training (150-200 epochs)
- Fallback: Implement PointNet architecture

**Camera accuracy <95%:**
- Check class balance (equal samples per color)
- Reduce hue augmentation (±5° instead of ±10°)
- Try ResNet18 transfer learning fallback
- Increase training epochs (70-100)

**Inference too slow (>100ms):**
- Profile with `torch.profiler`
- Optimize with TorchScript: `torch.jit.optimize_for_inference()`
- Consider ONNX Runtime (faster CPU inference)
- Reduce batch size to 1 (no batching in real-time)

---

## Next Steps (Phase 3)

Phase 2 delivers perception system. Phase 3 will use it:

```python
# Phase 3: Fuzzy Controller Integration
from fuzzy_controller import FuzzyController

fuzzy = FuzzyController()

while robot.step(timestep) != -1:
    world_state = perception.update(lidar, camera)

    # Fuzzy inputs
    obstacle_dist = perception.get_nearest_obstacle()[1]
    cube = perception.get_nearest_cube(color="green")

    # Fuzzy decision
    vx, vy, omega = fuzzy.compute(obstacle_dist, cube)

    # Actuate
    base.move(vx, vy, omega)
```

**Status:** Quickstart complete, ready for implementation (Phase 2.1-2.3).
