#!/usr/bin/env python3
"""
Generate synthetic training data for color classifier.
Creates images that simulate cube appearances in Webots.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

# Colors as they appear in Webots (RGB normalized)
BASE_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

def add_noise(img_array, intensity=0.1):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy = img_array.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def vary_brightness(img_array, factor):
    """Adjust brightness."""
    return np.clip(img_array.astype(float) * factor, 0, 255).astype(np.uint8)

def vary_color(base_color, variation=30):
    """Add slight variation to base color."""
    r, g, b = base_color
    return (
        max(0, min(255, r + random.randint(-variation, variation))),
        max(0, min(255, g + random.randint(-variation, variation))),
        max(0, min(255, b + random.randint(-variation, variation))),
    )

def create_cube_image(color, size=64):
    """Create a synthetic cube image."""
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)

    # Random cube size and position (simulating different distances)
    cube_size = random.randint(size // 3, size - 4)
    offset_x = random.randint(0, size - cube_size)
    offset_y = random.randint(0, size - cube_size)

    # Draw cube with slight color variation
    varied_color = vary_color(color, variation=20)
    draw.rectangle(
        [offset_x, offset_y, offset_x + cube_size, offset_y + cube_size],
        fill=varied_color
    )

    # Add some edge shading for 3D effect
    edge_color = tuple(max(0, c - 40) for c in varied_color)
    draw.line(
        [(offset_x + cube_size, offset_y), (offset_x + cube_size, offset_y + cube_size)],
        fill=edge_color, width=2
    )
    draw.line(
        [(offset_x, offset_y + cube_size), (offset_x + cube_size, offset_y + cube_size)],
        fill=edge_color, width=2
    )

    return img

def generate_dataset(output_dir, samples_per_class=200):
    """Generate synthetic dataset."""
    output_path = Path(output_dir)

    for color_name, base_color in BASE_COLORS.items():
        class_dir = output_path / color_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {samples_per_class} samples for {color_name}...")

        for i in range(samples_per_class):
            # Create base image
            img = create_cube_image(base_color, size=64)
            img_array = np.array(img)

            # Apply random augmentations
            # 1. Brightness variation
            brightness = random.uniform(0.6, 1.4)
            img_array = vary_brightness(img_array, brightness)

            # 2. Add noise
            if random.random() > 0.3:
                noise_level = random.uniform(0.02, 0.1)
                img_array = add_noise(img_array, noise_level)

            # Convert back to PIL
            img = Image.fromarray(img_array)

            # 3. Apply blur sometimes
            if random.random() > 0.7:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            # Save
            img.save(class_dir / f"{color_name}_{i:04d}.png")

    print(f"\nDataset generated at: {output_path}")
    print(f"Total samples: {samples_per_class * 3}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic color dataset")
    parser.add_argument("--output", type=str, default="dataset",
                        help="Output directory")
    parser.add_argument("--samples", type=int, default=200,
                        help="Samples per class")
    args = parser.parse_args()

    generate_dataset(args.output, args.samples)
