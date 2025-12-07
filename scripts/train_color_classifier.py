#!/usr/bin/env python3
"""
Training script for YouBot cube color classifier.
Uses MobileNetV3-Small backbone with fine-tuning.

Usage:
    python train_color_classifier.py --data_dir dataset/ --epochs 15
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path


class CubeColorDataset(Dataset):
    """Dataset para cubos coloridos do Webots."""

    CLASSES = ["red", "green", "blue"]

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), class_idx))
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx))

        print(f"Dataset carregado: {len(self.samples)} amostras")
        for i, name in enumerate(self.CLASSES):
            count = sum(1 for _, c in self.samples if c == i)
            print(f"  {name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class ColorClassifierModel(nn.Module):
    """MobileNetV3-Small com cabeça customizada para 3 classes."""

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # Backbone MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Substituir classificador
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def get_transforms(train=True):
    """Retorna transformações para treino/validação."""
    if train:
        return transforms.Compose([
            transforms.Resize((72, 72)),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    """Treina por uma época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Avalia o modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confusion = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[t, p] += 1

    return total_loss / len(loader), 100.0 * correct / total, confusion


def export_onnx(model, output_path, device):
    """Exporta modelo para ONNX."""
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
    )
    print(f"Modelo exportado para: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Treina classificador de cores")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Diretório com dataset (subpastas: red, green, blue)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="color_model.onnx",
                        help="Caminho do modelo ONNX de saída")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fração para validação")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Dataset
    full_dataset = CubeColorDataset(args.data_dir, transform=None)

    if len(full_dataset) == 0:
        print("Erro: Dataset vazio!")
        return

    # Split treino/validação
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [n_train, n_val]
    )

    # Datasets com transforms
    train_dataset = CubeColorDataset(args.data_dir, transform=get_transforms(train=True))
    val_dataset = CubeColorDataset(args.data_dir, transform=get_transforms(train=False))

    # Subset para treino/val
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    print(f"Treino: {len(train_subset)}, Validação: {len(val_subset)}")

    # Modelo
    model = ColorClassifierModel(num_classes=3, pretrained=True).to(device)

    # Congelar backbone inicialmente (apenas treinar cabeça)
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    best_model_state = None

    # Treino
    print("\n=== Fase 1: Treino da cabeça (backbone congelado) ===")
    for epoch in range(args.epochs // 2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs//2}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Descongelar backbone para fine-tuning
    print("\n=== Fase 2: Fine-tuning (backbone descongelado) ===")
    for param in model.backbone.features.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=args.epochs - args.epochs // 2)

    for epoch in range(args.epochs // 2, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Restaurar melhor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Avaliação final
    _, final_acc, confusion = evaluate(model, val_loader, criterion, device)
    print(f"\n=== Resultado Final ===")
    print(f"Melhor Acurácia: {best_acc:.1f}%")
    print(f"\nMatriz de Confusão:")
    print("         Pred: Red  Green  Blue")
    for i, name in enumerate(["Red", "Green", "Blue"]):
        print(f"Real {name:5}: {confusion[i, 0]:4d}  {confusion[i, 1]:5d}  {confusion[i, 2]:4d}")

    # Exportar ONNX
    export_onnx(model, args.output, device)

    # Copiar para diretório do controller
    controller_model_dir = Path(__file__).parent.parent / "IA_20252" / "controllers" / "youbot" / "model"
    controller_model_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.output, controller_model_dir / "color_model.onnx")
    print(f"Modelo copiado para: {controller_model_dir / 'color_model.onnx'}")


if __name__ == "__main__":
    main()
