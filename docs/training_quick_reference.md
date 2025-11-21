# Neural Network Training - Quick Reference

**Quick access guide for training hyperparameters and strategies**

---

## Training Data Requirements

| Sensor | Minimum Samples | Diversity Requirements |
|--------|----------------|------------------------|
| LIDAR | 1000 scans | >20 arena positions, 8 orientations |
| Camera | 500 images | >10 lighting conditions, 3 colors balanced (~167 each) |

**Split:** 70% train / 15% validation / 15% test (stratified by distance/color)

---

## Data Augmentation Cheat Sheet

### LIDAR Augmentation
```python
# Apply all during training
1. Gaussian noise: N(0, 0.01m) → 100% of batches
2. Random dropout: 5-10% points → 50% of batches
3. Rotation: ±15° → 70% of batches
4. Range scaling: ×[0.95, 1.05] → 30% of batches

# Effective dataset: ~1000 → ~3000-5000 examples
```

### Camera Augmentation
```python
# Apply during training
1. Brightness/Contrast: ×[0.7, 1.3] / γ[0.8, 1.2] → 80% of images
2. Color jitter: Hue ±10°, Sat ×[0.8, 1.2] → 60% of images
3. Horizontal flip: 50% | Vertical flip: 20%
4. Rotation: ±10° → 40% of images
5. Gaussian blur: σ=1-2px → 30% of images
6. Random crop: 90-100% size → 50% of images

# Effective dataset: ~500 → ~2500-4000 examples
```

---

## Hyperparameters Table

### LIDAR MLP

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | Small dataset |
| Learning rate | 0.001 | Adam default |
| Optimizer | Adam | lr_decay via ReduceLROnPlateau |
| Epochs | 100-200 | Early stop patience=20 |
| Loss | BCEWithLogitsLoss | Binary occupancy |
| Weight decay | 1e-4 | L2 regularization |
| Dropout | 0.2-0.3 | Hidden layers |
| LR schedule | ReduceLROnPlateau | factor=0.5, patience=10 |
| Grad clipping | max_norm=1.0 | Stability |

### Camera CNN (YOLOv5 Transfer Learning)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 16 | Memory intensive |
| Learning rate | 0.001 (head), 0.0001 (backbone) | Separate LR |
| Optimizer | Adam | Fine-tuning |
| Epochs | 50-100 | Transfer learning |
| Loss | BBox IoU + CrossEntropy | YOLO standard |
| Weight decay | 5e-4 | YOLO standard |
| LR schedule | CosineAnnealing | T_max=100, eta_min=1e-5 |
| Warmup | 3 epochs | 0 → 0.001 gradual |
| Label smoothing | 0.0 | Disabled (3 classes) |

### Camera CNN (Custom from Scratch)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 8-16 | Smaller for deep networks |
| Learning rate | 0.01 | Higher for scratch training |
| Optimizer | SGD (momentum=0.9) | Often better for CNNs |
| Epochs | 150-300 | More epochs needed |
| Loss | CrossEntropyLoss | 3-class classification |
| Weight decay | 1e-4 | L2 regularization |
| Dropout | 0.5 | FC layers only |
| LR schedule | StepLR | decay=0.1 every 50 epochs |

---

## Class Imbalance Solutions

### Camera (Cube Colors)
1. **Balanced collection** (preferred): ~167 images per color
2. **Class weighting**: `w_c = N_total / (N_classes × N_c)`
3. **Oversampling**: `WeightedRandomSampler` in DataLoader
4. **Focal loss** (advanced): α=0.25, γ=2.0 (if accuracy gap >10%)

### LIDAR (Obstacle vs Free Space)
1. **Weighted BCE**: `pos_weight = N_free / N_obstacle`
2. **Hard negative mining**: Sample 80% near-obstacle + 20% random free cells

---

## Validation Metrics Checklist

### LIDAR (Every Epoch)
- [ ] Validation loss (BCE)
- [ ] Obstacle detection accuracy (%) → Target: >90%
- [ ] Precision/Recall/F1-score (obstacle class)
- [ ] False positive rate → Target: <10%
- [ ] Inference time per scan (ms) → Target: <100ms

### Camera (Every Epoch)
- [ ] Validation loss (CE + BBox IoU)
- [ ] Color classification accuracy per class (%) → Target: >95%
- [ ] mAP@0.5 for bounding boxes
- [ ] False positive rate → Target: <5%
- [ ] Inference time per frame (ms) → Target: <33ms (>10 FPS)

---

## Early Stopping

```python
patience = 20  # epochs without improvement
min_delta = 1e-4  # minimum change to qualify as improvement

# Stop if:
# 1. Val loss plateaus for 20 epochs
# 2. Val accuracy reaches success criteria (90% LIDAR, 95% camera)
# 3. Overfitting detected (train_loss << val_loss)
```

---

## Test Set Evaluation Protocol

### LIDAR (SC-001, SC-003, SC-006)
1. Freeze model weights
2. Run inference on 100+ held-out test scans
3. Metrics:
   - Obstacle detection accuracy: >90%
   - Inference time: <100ms per scan
   - Zero missed obstacles in 360° rotation
4. Visualize predictions (polar plots)
5. Document in `docs/lidar_test_results.md`

### Camera (SC-002, SC-004, SC-005, SC-007)
1. Freeze model weights
2. Run inference on 50+ test images per color (150 total)
3. Metrics:
   - Color classification accuracy: >95% per class
   - Inference FPS: >10 FPS
   - False positive rate: <5%
   - Correctly identify all 3 colors at 1m
4. Visualize bounding boxes and colors
5. Document in `docs/camera_test_results.md`

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Loss not decreasing | Try LR in [1e-4, 1e-2], check initialization |
| Overfitting | Increase dropout to 0.5, add weight decay 1e-3 |
| Underfitting | Increase model size, train longer |
| Class imbalance | Use class weighting or oversampling |
| Inference too slow | Model quantization (FP16), lighter architecture |
| Augmentation errors | Limit hue shift to ±10°, verify labels visually |

---

## Implementation Checklist

### Before Training
- [ ] Data collected with diversity requirements met
- [ ] Train/val/test split (70/15/15) with stratification
- [ ] Augmentation pipeline tested
- [ ] Class imbalance assessed and mitigation chosen
- [ ] Hyperparameters documented in DECISIONS.md
- [ ] TensorBoard logging configured
- [ ] Early stopping + LR scheduling implemented
- [ ] Model saving strategy defined

### During Training
- [ ] Monitor train/val loss curves
- [ ] Check for over/underfitting
- [ ] Validate inference time periodically
- [ ] Save checkpoints every 10 epochs
- [ ] Visualize predictions on val set

### After Training
- [ ] Evaluate on held-out test set
- [ ] Serialize model (torch.jit or ONNX)
- [ ] Document architecture in DECISIONS.md
- [ ] Commit model checkpoints
- [ ] Update REFERENCIAS.md if needed

---

## PyTorch Code Templates

### LIDAR Training Loop
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = LIDAR_MLP(input_size=667, hidden_sizes=[256, 128, 64], output_size=4096)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
early_stopping = EarlyStopping(patience=20)

for epoch in range(200):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_loss = validate(model, val_loader, criterion)
    scheduler.step(val_loss)

    if early_stopping(val_loss):
        break
```

### Camera Training Loop (Transfer Learning)
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.model[-1] = nn.Conv2d(in_channels, 18, kernel_size=1)  # 3 classes

optimizer = optim.Adam([
    {'params': model.model[:-1].parameters(), 'lr': 0.0001},
    {'params': model.model[-1].parameters(), 'lr': 0.001}
], weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

for epoch in range(100):
    if epoch < 3:  # Warmup
        warmup_factor = (epoch + 1) / 3
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * warmup_factor

    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = compute_yolo_loss(predictions, targets)
        loss.backward()
        optimizer.step()

    scheduler.step()
```

### Albumentations Augmentation
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## Key References

1. **Goodfellow et al. (2016)** - Deep Learning textbook (Ch 7.4 augmentation, Ch 8 optimization)
2. **Krizhevsky et al. (2012)** - AlexNet (dropout, data augmentation pioneers)
3. **Redmon et al. (2016)** - YOLO (real-time detection, extensive augmentation)
4. **Qi et al. (2017)** - PointNet (point cloud augmentation: rotation, jittering)
5. **Kingma & Ba (2014)** - Adam optimizer (adaptive learning rates)
6. **Lin et al. (2017)** - Focal loss (class imbalance solution)

See `REFERENCIAS.md` for full citations and `docs/neural_network_training_strategy.md` for detailed explanations.

---

**Last Updated:** 2025-11-21
**Status:** Ready for Phase 2 implementation
