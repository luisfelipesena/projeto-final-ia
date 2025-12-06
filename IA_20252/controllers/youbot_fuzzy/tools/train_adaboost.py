"""Train AdaBoost color classifier from cube patches."""

from __future__ import annotations

import os
import argparse
from glob import glob
from pathlib import Path

try:
    import cv2
    import numpy as np
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    import joblib
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Install with: pip install opencv-python scikit-learn joblib")


def extract_features(patch: np.ndarray, hog_descriptor=None) -> np.ndarray:
    """Extract HSV histogram + HOG features (must match adaboost_classifier.py)."""
    resized = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # HSV histogram with 8x8x4 bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    # HOG features (same config as adaboost_classifier.py)
    if hog_descriptor is not None:
        hog_vec = hog_descriptor.compute(resized)
    else:
        hog_vec = resized.flatten().astype("float32") / 255.0
    features = np.concatenate([hist.flatten(), hog_vec.flatten()])
    return features


def load_dataset(dataset_path: Path) -> tuple[list, list]:
    """Load images from dataset directory organized by color folders."""
    X, y = [], []
    colors = ["red", "green", "blue"]
    # HOG descriptor matching adaboost_classifier.py
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)

    for color in colors:
        color_dir = dataset_path / color
        if not color_dir.exists():
            print(f"Warning: Directory {color_dir} not found, skipping {color}")
            continue

        patterns = [str(color_dir / "*.png"), str(color_dir / "*.jpg")]
        files = []
        for pattern in patterns:
            files.extend(glob(pattern))

        print(f"Found {len(files)} images for {color}")

        for img_path in files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_features(img, hog)
            X.append(features)
            y.append(color.upper())

    return X, y


def train_classifier(X: list, y: list) -> AdaBoostClassifier:
    """Train AdaBoost classifier on extracted features."""
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    clf.fit(X, y)
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train AdaBoost color classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/cubes/train",
        help="Path to dataset directory with red/green/blue subdirs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/adaboost_color.pkl",
        help="Output path for trained model"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} not found")
        print("Run dataset capture first: python tools/run_dataset_capture.py")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    X, y = load_dataset(dataset_path)

    if len(X) == 0:
        print("Error: No images found in dataset")
        return 1

    print(f"Training on {len(X)} samples...")
    clf = train_classifier(X, y)

    # Evaluate on training data
    accuracy = clf.score(X, y)
    print(f"Training accuracy: {accuracy:.2%}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Model saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
