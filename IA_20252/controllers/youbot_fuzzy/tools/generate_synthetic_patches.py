"""Generate synthetic color patches for AdaBoost training (no Webots needed)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Install: pip install opencv-python numpy")


# Webots cube colors (pure RGB)
CUBE_COLORS = {
    "red": (0, 0, 255),     # BGR format for OpenCV
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def generate_patches(
    output_dir: Path,
    samples_per_class: int = 200,
    patch_size: int = 32,
    add_noise: bool = True,
    add_lighting_variation: bool = True,
):
    """Generate synthetic color patches with variations."""

    for color_name, base_bgr in CUBE_COLORS.items():
        color_dir = output_dir / color_name
        color_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            # Start with base color
            patch = np.full((patch_size, patch_size, 3), base_bgr, dtype=np.uint8)

            if add_lighting_variation:
                # Simulate lighting variation (brightness)
                brightness = np.random.uniform(0.6, 1.4)
                patch = np.clip(patch * brightness, 0, 255).astype(np.uint8)

            if add_noise:
                # Add Gaussian noise
                noise = np.random.normal(0, 15, patch.shape).astype(np.int16)
                patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random small color shift (simulate camera white balance)
            if np.random.random() < 0.3:
                shift = np.random.randint(-20, 20, 3)
                patch = np.clip(patch.astype(np.int16) + shift, 0, 255).astype(np.uint8)

            # Save patch
            filename = color_dir / f"{color_name}_{i:04d}.png"
            cv2.imwrite(str(filename), patch)

        print(f"Generated {samples_per_class} patches for {color_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic color patches")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/cubes/train",
        help="Output directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Samples per color class"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable noise augmentation"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"Generating dataset in {output_dir}...")

    generate_patches(
        output_dir,
        samples_per_class=args.samples,
        add_noise=not args.no_noise,
    )

    print(f"\nDataset ready! Now train with:")
    print(f"  python tools/train_adaboost.py --dataset {output_dir} --output models/adaboost_color.pkl")


if __name__ == "__main__":
    main()
