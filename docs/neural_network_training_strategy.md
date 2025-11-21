# Neural Network Training Strategy - YouBot Perception System

**Project:** Sistema Autônomo de Coleta e Organização de Objetos com YouBot
**Author:** Luis Felipe Cordeiro Sena
**Date:** 2025-11-21
**Phase:** 2 - Percepção com Redes Neurais

---

## Overview

This document defines comprehensive training strategies for two neural network systems:
1. **LIDAR Obstacle Detector**: MLP/CNN for processing 667-point LIDAR scans
2. **Camera Cube Detector**: CNN for detecting and classifying colored cubes (green, blue, red)

Both systems must handle limited training data (~1000 LIDAR scans, ~500 camera images) while achieving robust performance under variations (lighting, noise, occlusion).

---

## 1. Data Collection Strategy

### 1.1 LIDAR Training Dataset

**Target Size:** 1000+ labeled scans (FR-010)

**Collection Protocol:**
- **Diversity Requirements (SC-009):**
  - Sample from >20 distinct robot positions in arena
  - Include positions: near walls, near obstacles, open space, corners
  - Vary robot orientation: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- **Labeling Strategy:**
  - Use arena_map.md for ground truth obstacle positions
  - Label each LIDAR point: {obstacle, free_space, wall, unknown}
  - Distance bins: [0-0.5m, 0.5-1.0m, 1.0-2.0m, 2.0-3.5m]
- **Data Format:**
  - Input: 667 float32 ranges (normalized to [0,1])
  - Output: Binary occupancy grid (64×64 cells, 0.1m resolution)
- **Storage:** `data/lidar/train_scans.npz` (NumPy compressed format)

**Collection Script:** Use GPS during data collection phase (allowed per DECISÃO 009) to track robot position for accurate labeling.

### 1.2 Camera Training Dataset

**Target Size:** 500+ labeled images (FR-020)

**Collection Protocol:**
- **Diversity Requirements (SC-009):**
  - Collect from >10 lighting conditions (Webots directional light intensity: 0.5-1.5)
  - Distance variation: 0.5m, 1.0m, 1.5m, 2.0m from cubes
  - Angle variation: 0°, ±15°, ±30° from cube center
  - Include partial occlusion: 50%, 70%, 100% cube visibility
  - Balance classes: ~167 images per color (green, blue, red)
- **Labeling Strategy:**
  - Use Webots supervisor.getFromDef() to get ground truth cube positions
  - Compute bounding boxes from 3D position → 2D camera projection
  - Format: YOLO-style (class_id, x_center, y_center, width, height) normalized to [0,1]
- **Data Format:**
  - Input: 512×512×3 RGB images (uint8)
  - Output: Class label (0=green, 1=blue, 2=red) + bounding box coordinates
- **Storage:** `data/camera/images/*.jpg` + `data/camera/labels/*.txt`

**Annotation Tool:** Custom script using OpenCV for manual verification of automated labels.

### 1.3 Train/Validation/Test Split

**Standard Split (FR-023):** 70% train / 15% validation / 15% test

**Stratification:**
- **LIDAR:** Stratify by distance to nearest obstacle (bins: 0-0.5m, 0.5-1.0m, 1.0-2.0m, 2.0+m)
- **Camera:** Stratify by cube color (ensure equal representation per split)

**Implementation:**
```python
from sklearn.model_selection import train_test_split

# LIDAR
X_train, X_temp, y_train, y_temp = train_test_split(
    lidar_data, labels, test_size=0.3, stratify=distance_bins, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=distance_bins_temp, random_state=42
)

# Camera
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    camera_imgs, cube_labels, test_size=0.3, stratify=cube_colors, random_state=42
)
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, stratify=cube_colors_temp, random_state=42
)
```

---

## 2. Data Augmentation Strategies

### 2.1 LIDAR Augmentation (FR-024)

**Scientific Basis:**
- **Goodfellow et al. (2016), Chapter 7.4**: "Data augmentation is essential for regularization when training data is limited"
- **Qi et al. (2017)**: PointNet uses rotation and jittering for 3D point cloud robustness

**Augmentation Pipeline:**

1. **Gaussian Noise Injection** (Simulate sensor noise)
   - Add N(0, σ²) noise to range measurements
   - σ = 0.01m (1cm standard deviation, realistic for LIDAR)
   - Probability: 100% of training batches
   - Implementation: `ranges += np.random.normal(0, 0.01, size=ranges.shape)`

2. **Random Dropout** (Simulate occlusion/missed detections)
   - Randomly set 5-10% of LIDAR points to max range (inf)
   - Simulates beam blockage by gripper arm or transient occlusion
   - Probability: 50% of training batches
   - Implementation: `mask = np.random.rand(667) > 0.95; ranges[mask] = 3.5`

3. **Rotation Augmentation** (Orientation invariance)
   - Rotate LIDAR scan by random angle θ ∈ [-15°, +15°]
   - Circular shift of range array: `np.roll(ranges, shift_points)`
   - Probability: 70% of training batches
   - Corresponds to small robot orientation errors

4. **Range Scaling** (Distance variation simulation)
   - Multiply all ranges by factor ∈ [0.95, 1.05]
   - Simulates minor calibration errors
   - Probability: 30% of training batches

**Total Effective Dataset Size:** ~1000 real scans × 3-5 augmentations = **~3000-5000 training examples**

**PyTorch Implementation:**
```python
class LIDARAugmentation:
    def __init__(self, noise_std=0.01, dropout_rate=0.1, rotation_range=15):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.rotation_range = rotation_range

    def __call__(self, lidar_ranges):
        # Gaussian noise
        ranges = lidar_ranges + np.random.normal(0, self.noise_std, size=lidar_ranges.shape)

        # Random dropout
        if np.random.rand() > 0.5:
            dropout_mask = np.random.rand(len(ranges)) < self.dropout_rate
            ranges[dropout_mask] = 3.5  # Max range

        # Rotation
        if np.random.rand() > 0.3:
            angle_deg = np.random.uniform(-self.rotation_range, self.rotation_range)
            shift_points = int((angle_deg / 270) * len(ranges))
            ranges = np.roll(ranges, shift_points)

        # Range scaling
        if np.random.rand() > 0.7:
            scale = np.random.uniform(0.95, 1.05)
            ranges *= scale

        return ranges
```

### 2.2 Camera Augmentation (FR-024)

**Scientific Basis:**
- **Krizhevsky et al. (2012)**: AlexNet pioneered data augmentation for image classification (horizontal flips, crops, color jitter)
- **Redmon et al. (2016)**: YOLO uses extensive augmentation (hue, saturation, exposure) for robust object detection
- **Goodfellow et al. (2016), Chapter 7.4**: Geometric and photometric transformations improve generalization

**Augmentation Pipeline:**

1. **Brightness/Contrast Adjustment** (Lighting robustness)
   - Brightness: multiply pixel values by factor ∈ [0.7, 1.3]
   - Contrast: apply gamma correction with γ ∈ [0.8, 1.2]
   - Probability: 80% of training images
   - Addresses lighting variations (SC-009: >10 lighting conditions)
   - Implementation: `image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)`

2. **Color Jitter** (Hue/Saturation variation)
   - Hue shift: ±10° (critical for color classification!)
   - Saturation: ×[0.8, 1.2]
   - Probability: 60% of training images
   - **Caution:** Limited hue shift to avoid label contamination (green → blue)
   - Implementation: Convert RGB → HSV, modify channels, convert back

3. **Horizontal/Vertical Flip** (Spatial augmentation)
   - Horizontal flip: 50% probability (mirrors left/right)
   - Vertical flip: 20% probability (rare, but valid)
   - Update bounding box coordinates accordingly
   - Implementation: `image = cv2.flip(image, flipCode=1)`

4. **Random Rotation** (Orientation robustness)
   - Rotate image by angle ∈ [-10°, +10°]
   - Use OpenCV affine transformation
   - Probability: 40% of training images
   - Update bounding box corners and recompute axis-aligned box

5. **Gaussian Blur** (Motion blur simulation)
   - Apply Gaussian kernel (σ = 1-2 pixels)
   - Simulates motion blur when robot moves during capture
   - Probability: 30% of training images
   - Implementation: `image = cv2.GaussianBlur(image, (5,5), sigma)`

6. **Random Crop and Resize** (Scale variation)
   - Crop to 90-100% of original image size
   - Resize back to 512×512
   - Simulates distance variation
   - Probability: 50% of training images

**Total Effective Dataset Size:** ~500 real images × 5-8 augmentations = **~2500-4000 training examples**

**PyTorch/Albumentations Implementation:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

camera_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(1, 2), p=0.3),
    A.RandomResizedCrop(height=512, width=512, scale=(0.9, 1.0), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 3. Training Hyperparameters

### 3.1 LIDAR Neural Network Training

**Architecture Decision:** MLP (Baseline) vs PointNet (Advanced) → Document in DECISIONS.md

**Recommended Hyperparameters (MLP Baseline):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 32 | Small dataset → smaller batches reduce overfitting (Goodfellow 2016, Ch 8.1) |
| **Learning Rate** | 0.001 (initial) | Adam default, works well for MLPs (Kingma & Ba, 2014) |
| **Optimizer** | Adam | Adaptive learning rate, faster convergence than SGD (Kingma & Ba, 2014) |
| **Epochs** | 100-200 | Monitor validation loss, use early stopping (patience=20) |
| **Loss Function** | Binary Cross-Entropy | For binary occupancy classification (obstacle vs free) |
| **Weight Decay** | 1e-4 | L2 regularization to prevent overfitting |
| **Dropout** | 0.2-0.3 | Applied to hidden layers (Srivastava et al., 2014) |
| **Learning Rate Schedule** | ReduceLROnPlateau | Reduce LR by factor 0.5 when validation loss plateaus (patience=10) |
| **Gradient Clipping** | Max norm = 1.0 | Prevents exploding gradients |

**Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = LIDAR_MLP(input_size=667, hidden_sizes=[256, 128, 64], output_size=4096)  # 64×64 occupancy grid
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training loop
for epoch in range(200):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    val_loss = validate(model, val_loader, criterion)
    scheduler.step(val_loss)

    # Early stopping
    if early_stopping_check(val_loss, patience=20):
        break
```

### 3.2 Camera CNN Training

**Architecture Decision:** YOLOv5-nano vs Custom CNN vs SSD → Document in DECISIONS.md

**Recommended Hyperparameters (Transfer Learning with YOLOv5):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 16 | Larger than LIDAR (images are memory-intensive), balances speed and stability |
| **Learning Rate** | 0.001 (backbone: 0.0001) | Lower LR for pre-trained layers (Yosinski et al., 2014) |
| **Optimizer** | Adam | Good for fine-tuning (faster than SGD with momentum for small datasets) |
| **Epochs** | 50-100 | Fewer epochs needed with transfer learning |
| **Loss Function** | Combined: BBox (IoU) + Classification (CrossEntropy) | YOLO standard loss |
| **Weight Decay** | 5e-4 | Standard for YOLO training |
| **Augmentation** | See Section 2.2 | Critical for robustness |
| **Learning Rate Schedule** | Cosine Annealing | Smooth LR decay from 0.001 → 0.00001 over training |
| **Warmup Epochs** | 3 | Gradual LR increase 0 → 0.001 in first 3 epochs (stabilizes training) |
| **Label Smoothing** | 0.0 | Disabled (only 3 classes, high confidence needed) |

**Implementation (PyTorch + YOLOv5):**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.model[-1] = nn.Conv2d(in_channels, 18, kernel_size=1)  # Modify head for 3 classes

# Separate learning rates for backbone vs head
optimizer = optim.Adam([
    {'params': model.model[:-1].parameters(), 'lr': 0.0001},  # Backbone (frozen initially)
    {'params': model.model[-1].parameters(), 'lr': 0.001}     # Head (train from scratch)
], weight_decay=5e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# Warmup
def warmup_lr(epoch, warmup_epochs=3, base_lr=0.001):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# Training loop
for epoch in range(100):
    if epoch < 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr(epoch)

    for batch_idx, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(images)
        loss = compute_yolo_loss(predictions, targets)  # BBox IoU + Class CE
        loss.backward()
        optimizer.step()

    scheduler.step()
```

**Recommended Hyperparameters (Custom CNN from Scratch):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 8-16 | Smaller batch if training deeper custom network |
| **Learning Rate** | 0.01 (initial) | Higher initial LR for training from scratch (decay to 0.0001) |
| **Optimizer** | SGD with momentum (0.9) | Often outperforms Adam for CNNs trained from scratch (Goodfellow 2016) |
| **Epochs** | 150-300 | More epochs needed without transfer learning |
| **Loss Function** | CrossEntropyLoss (multi-class) | For 3-class color classification |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Learning Rate Schedule** | StepLR (decay by 0.1 every 50 epochs) | Standard for CNNs |
| **Dropout** | 0.5 in FC layers | Heavy dropout to combat overfitting (small dataset) |

---

## 4. Handling Class Imbalance

### 4.1 Camera Cube Colors

**Problem:** Supervisor spawns 15 cubes randomly → unequal color distribution per simulation run

**Strategies:**

1. **Balanced Data Collection** (Preferred)
   - Run multiple simulations, manually ensure ~167 images per color
   - Stratified sampling during train/val/test split
   - **Implementation:** Track color counts during collection, stop when balanced

2. **Class Weighting in Loss Function**
   - Compute class weights: `w_c = N_total / (N_classes × N_c)`
   - Apply to CrossEntropyLoss: `nn.CrossEntropyLoss(weight=class_weights)`
   - **When to use:** If imbalance persists after collection (e.g., 200 green, 150 blue, 150 red)

3. **Oversampling Minority Classes**
   - Use `WeightedRandomSampler` in PyTorch DataLoader
   - Sample minority classes more frequently during training
   - **Implementation:**
   ```python
   from torch.utils.data import WeightedRandomSampler

   class_counts = [200, 150, 150]  # green, blue, red
   class_weights = [1.0 / count for count in class_counts]
   sample_weights = [class_weights[label] for label in all_labels]
   sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
   train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
   ```

4. **Focal Loss** (Advanced)
   - **Reference:** Lin et al. (2017) "Focal Loss for Dense Object Detection" (RetinaNet paper)
   - Down-weights easy examples, focuses on hard-to-classify samples
   - **When to use:** If accuracy on minority class is significantly lower (>10% gap)
   - **Implementation:**
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma

       def forward(self, inputs, targets):
           ce_loss = F.cross_entropy(inputs, targets, reduction='none')
           pt = torch.exp(-ce_loss)
           focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
           return focal_loss.mean()
   ```

**Recommendation:** Start with balanced data collection. If imbalance occurs (>20% difference between classes), use class weighting or oversampling.

### 4.2 LIDAR Obstacle Distance Classes

**Problem:** Most scans will have more "free space" than "obstacle" labels (imbalanced binary classification)

**Strategies:**

1. **Weighted Binary Cross-Entropy**
   - Compute pos_weight = (N_free / N_obstacle)
   - Apply to loss: `nn.BCEWithLogitsLoss(pos_weight=pos_weight)`
   - Emphasizes obstacle detection (more critical than free space)

2. **Hard Negative Mining**
   - During training, select hard negatives (free space cells near obstacles)
   - Increases focus on decision boundaries
   - **Implementation:** Sample 80% near-obstacle free cells + 20% random free cells per batch

**Recommendation:** Use weighted BCE loss with pos_weight tuned on validation set.

---

## 5. Validation Strategy

### 5.1 Validation Monitoring

**Primary Metrics (Track every epoch):**

1. **LIDAR:**
   - Validation loss (BCE)
   - Obstacle detection accuracy (%)
   - Precision/Recall/F1-score (obstacle class)
   - False positive rate (<10% target)
   - Inference time per scan (ms)

2. **Camera:**
   - Validation loss (CE + BBox IoU)
   - Color classification accuracy per class (%)
   - Mean Average Precision (mAP@0.5) for bounding boxes
   - False positive rate (<5% target per FR-019)
   - Inference time per frame (ms)

**Implementation (TensorBoard Logging):**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/lidar_experiment1')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss, val_metrics = validate(...)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
    writer.add_scalar('Metrics/f1_score', val_metrics['f1'], epoch)
    writer.add_scalar('Timing/inference_ms', val_metrics['inference_time'], epoch)
```

### 5.2 Early Stopping

**Implementation:**
```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

**Stopping Criteria:**
- Validation loss plateaus for 20 consecutive epochs (no improvement >0.0001)
- Validation accuracy reaches 95% (for camera) or 90% (for LIDAR) → Success criteria met
- Training loss << validation loss (overfitting detected) → Increase regularization

### 5.3 Cross-Validation (Optional)

**When to use:** If dataset is very small (<500 samples) or results are unstable

**Implementation:** 5-fold cross-validation
- Split training+validation data into 5 folds
- Train 5 models, each using 4 folds for training and 1 for validation
- Report mean ± std accuracy across folds
- Use for hyperparameter tuning, then train final model on all training data

**Trade-off:** 5× training time, but more robust performance estimates

---

## 6. Test Set Evaluation

### 6.1 Final Testing Protocol

**LIDAR (SC-001, SC-003, SC-006):**
1. Freeze model weights (no training on test set!)
2. Run inference on 100+ test scans (held-out)
3. Compute metrics:
   - Obstacle detection accuracy: >90% (SC-001)
   - Inference time: <100ms per scan (SC-003)
   - Zero missed obstacles in 360° rotation test (SC-006)
4. Qualitative analysis: Visualize predictions on test scans (matplotlib polar plots)
5. Document results in `docs/lidar_test_results.md`

**Camera (SC-002, SC-004, SC-005, SC-007):**
1. Freeze model weights
2. Run inference on 50+ test images per color (150 total minimum)
3. Compute metrics:
   - Color classification accuracy: >95% per class (SC-002)
   - Inference FPS: >10 FPS (SC-004)
   - False positive rate: <5% (SC-005)
   - Correctly identify all 3 colors at 1m distance (SC-007)
4. Qualitative analysis: Visualize bounding boxes and color predictions
5. Document results in `docs/camera_test_results.md`

### 6.2 Failure Analysis

**For both models:**
- Identify failure modes: Which test cases fail most often?
- Error categories:
  - False positives (detect when nothing there)
  - False negatives (miss real obstacles/cubes)
  - Misclassification (wrong color)
  - Localization errors (bounding box off)
- Root cause analysis: Lighting? Occlusion? Distance? Angle?
- Document in test results files with example images/scans

---

## 7. Training Best Practices Summary

### 7.1 Checklist Before Training

- [ ] Data collected and verified (>1000 LIDAR, >500 camera with diversity)
- [ ] Train/val/test split implemented (70/15/15) with stratification
- [ ] Data augmentation pipeline tested (verify augmentations look reasonable)
- [ ] Class imbalance assessed and mitigation strategy chosen
- [ ] Hyperparameters documented in DECISIONS.md with justifications
- [ ] Training logging configured (TensorBoard or Weights & Biases)
- [ ] Early stopping and learning rate scheduling implemented
- [ ] Model saving strategy defined (save best validation checkpoint)

### 7.2 During Training

- [ ] Monitor training/validation loss curves (should converge, not diverge)
- [ ] Check for overfitting (train acc >> val acc → add regularization)
- [ ] Check for underfitting (both train and val acc low → increase model capacity)
- [ ] Validate inference time periodically (must meet <100ms requirement)
- [ ] Save checkpoints every 10 epochs (disk space permitting)
- [ ] Visualize predictions on validation set (sanity check)

### 7.3 After Training

- [ ] Evaluate on held-out test set (report all success criteria metrics)
- [ ] Serialize model for Webots deployment (torch.jit.save or ONNX export)
- [ ] Document final architecture and hyperparameters in DECISIONS.md
- [ ] Commit model checkpoints to Git LFS or separate storage
- [ ] Update REFERENCIAS.md if using new techniques not yet documented

---

## 8. References from REFERENCIAS.md

### Deep Learning Fundamentals

1. **Goodfellow, I.; Bengio, Y.; Courville, A. (2016)** *Deep Learning*. MIT Press.
   - **Chapter 6:** Deep Feedforward Networks (MLP architecture)
   - **Chapter 7.4:** Dataset Augmentation (noise injection, geometric transforms)
   - **Chapter 8:** Optimization for Training Deep Models (SGD, Adam, learning rates)
   - **Chapter 11:** Practical Methodology (hyperparameter tuning, debugging)

### Neural Network Training

2. **Krizhevsky, A.; Sutskever, I.; Hinton, G. E. (2012)** "ImageNet Classification with Deep CNNs" (AlexNet). *NeurIPS*.
   - Pioneered data augmentation: horizontal flips, color jitter, random crops
   - Dropout regularization (0.5 in FC layers)
   - SGD with momentum (0.9) and weight decay (5e-4)

3. **He, K.; Zhang, X.; Ren, S.; Sun, J. (2016)** "Deep Residual Learning for Image Recognition" (ResNet). *IEEE CVPR*.
   - Batch normalization for faster convergence
   - Learning rate schedule: divide by 10 when validation loss plateaus
   - Transfer learning: pre-trained backbones for small datasets

### Object Detection

4. **Redmon, J.; et al. (2016)** "You Only Look Once: Unified, Real-Time Object Detection" (YOLO). *IEEE CVPR*.
   - Real-time detection: >45 FPS on GPU
   - Data augmentation: hue (±20°), saturation (×[0.25, 2]), exposure (×[0.25, 4])
   - Mosaic augmentation (combine 4 images) for robust small object detection

5. **Liu, W.; et al. (2016)** "SSD: Single Shot MultiBox Detector". *ECCV*.
   - Multi-scale feature maps for detecting small objects (cubes!)
   - Hard negative mining: 3:1 ratio of negatives to positives per batch

6. **Lin, T. Y.; et al. (2017)** "Focal Loss for Dense Object Detection" (RetinaNet). *IEEE ICCV*.
   - Focal loss addresses class imbalance (α=0.25, γ=2)
   - Down-weights easy examples, focuses on hard cases

### LIDAR/Point Cloud Processing

7. **Qi, C. R.; et al. (2017)** "PointNet: Deep Learning on Point Sets for 3D Classification". *IEEE CVPR*.
   - Point cloud augmentation: random rotation, jittering (σ=0.02)
   - Dropout on point features (0.3-0.5)
   - Trained with Adam (lr=0.001, decay=0.7 every 20 epochs)

### Optimization Algorithms

8. **Kingma, D. P.; Ba, J. (2014)** "Adam: A Method for Stochastic Optimization". *ICLR*.
   - Adaptive learning rates per parameter
   - Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8
   - Works well for most deep learning tasks

### Regularization

9. **Srivastava, N.; et al. (2014)** "Dropout: A Simple Way to Prevent Overfitting". *JMLR*.
   - Dropout rate 0.2-0.5 depending on layer depth
   - Applied during training, disabled during inference
   - Equivalent to ensemble of 2^n subnetworks

### Transfer Learning

10. **Yosinski, J.; et al. (2014)** "How transferable are features in deep neural networks?". *NeurIPS*.
    - Lower layers learn general features (edges, textures)
    - Higher layers learn task-specific features
    - Fine-tuning strategy: freeze early layers, train later layers with lower LR

---

## 9. Implementation Timeline (Phase 2)

### Week 1: Data Collection & Preprocessing
- **Days 1-2:** LIDAR data collection script (1000 scans from 20+ positions)
- **Days 3-4:** Camera data collection script (500 images, 10+ lighting conditions)
- **Days 5-7:** Data labeling, augmentation pipeline, train/val/test split

### Week 2: LIDAR Neural Network
- **Days 8-9:** MLP architecture implementation and baseline training
- **Days 10-11:** Hyperparameter tuning, early stopping, learning rate scheduling
- **Days 12-14:** Evaluation on test set, inference time optimization, documentation

### Week 3: Camera CNN
- **Days 15-16:** YOLOv5 or custom CNN implementation
- **Days 17-18:** Transfer learning setup, fine-tuning, data augmentation validation
- **Days 19-21:** Evaluation on test set, multi-class accuracy analysis, documentation

### Week 4: Integration & Final Testing
- **Days 22-23:** Serialize models (torch.jit), integrate into Webots controller
- **Days 24-25:** End-to-end testing in simulation, perception API design
- **Days 26-28:** Performance profiling, documentation, DECISIONS.md updates

**Buffer:** 2 days for unexpected issues (model convergence problems, data quality issues)

---

## 10. Success Criteria Mapping

| Requirement | Training Strategy | Validation Method |
|-------------|-------------------|-------------------|
| FR-003: >90% LIDAR accuracy | Augmentation + regularization | Test set evaluation (SC-001) |
| FR-004: <100ms LIDAR inference | Model pruning, batch inference | Timing tests (SC-003) |
| FR-013: >95% camera accuracy | Transfer learning, class balancing | Per-class accuracy (SC-002) |
| FR-015: >10 FPS camera | Lightweight model (YOLOv5-nano) | FPS benchmarking (SC-004) |
| FR-019: <5% false positives | Hard negative mining, confidence threshold | False positive test (SC-005) |
| FR-024: Data augmentation | LIDAR (noise, dropout, rotation), Camera (brightness, color, flip) | Ablation study (with vs without) |

---

## 11. Troubleshooting Guide

### Training Loss Not Decreasing
- **Cause:** Learning rate too high or too low, bad initialization
- **Solution:** Try LR in range [1e-4, 1e-2], use Xavier/He initialization

### Overfitting (Train Acc >> Val Acc)
- **Cause:** Model too complex, insufficient regularization
- **Solution:** Increase dropout (0.5), add weight decay (1e-3), reduce model size

### Underfitting (Both Train and Val Acc Low)
- **Cause:** Model too simple, insufficient training
- **Solution:** Increase model capacity (more layers/units), train longer (more epochs)

### Class Imbalance Issues
- **Cause:** Unequal class distribution in training data
- **Solution:** Use class weighting, oversampling, or focal loss (see Section 4)

### Inference Too Slow
- **Cause:** Model too large, inefficient operations
- **Solution:** Use model quantization (FP16), batch processing, or lighter architecture

### Augmentation Causing Label Errors
- **Cause:** Excessive hue shift changes cube colors, rotation breaks bounding boxes
- **Solution:** Limit hue shift to ±10°, verify augmented labels visually

---

**Document Status:** Complete
**Next Steps:** Begin data collection (Week 1), document architecture decision in DECISIONS.md, create training notebooks in `notebooks/02_lidar_training.ipynb` and `notebooks/03_cube_detection_training.ipynb`
