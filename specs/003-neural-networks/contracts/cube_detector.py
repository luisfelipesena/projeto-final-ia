"""
API Contract: CubeDetector

Purpose: CNN interface for cube color detection and localization
Phase: 2 - Perception (Neural Networks)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox:
    """Rectangular region in image"""

    x_min: int  # Left edge (pixels)
    y_min: int  # Top edge
    x_max: int  # Right edge
    y_max: int  # Bottom edge

    @property
    def width(self) -> int:
        """Box width in pixels"""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Box height in pixels"""
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[int, int]:
        """Center coordinates (x, y)"""
        return ((self.x_min + self.x_max) // 2,
                (self.y_min + self.y_max) // 2)

    @property
    def area(self) -> int:
        """Box area in pixels²"""
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """
        Intersection over Union with another box

        Args:
            other: Another BoundingBox

        Returns:
            IoU score [0,1]
        """
        ...


@dataclass
class CubeObservation:
    """Single detected cube with color and location"""

    color: str          # "green" | "blue" | "red"
    bbox: BoundingBox   # Pixel coordinates in image
    confidence: float   # Classification confidence [0,1]
    distance: float     # Estimated distance (meters)
    angle: float        # Bearing from robot center (degrees)
    timestamp: float    # Unix timestamp

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """
        Check if detection confidence is sufficient

        Args:
            min_confidence: Minimum acceptable confidence

        Returns:
            True if confidence >= min_confidence
        """
        ...

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging"""
        ...


class CubeDetector:
    """CNN wrapper for cube color detection"""

    # Class constants
    CLASS_NAMES = ["green", "blue", "red"]
    NUM_CLASSES = 3
    IMAGE_SIZE = (512, 512)

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize cube detector with trained CNN

        Args:
            model_path: Path to TorchScript model (.pt file)
            device: "cpu" or "cuda" (default: "cpu")

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails
        """
        ...

    def detect(self, image: np.ndarray) -> List[CubeObservation]:
        """
        Detect cubes in camera image

        Args:
            image: [512, 512, 3] RGB array (uint8)

        Returns:
            List of CubeObservation (one per detected cube)
            Empty list if no cubes detected

        Raises:
            ValueError: If image.shape != (512, 512, 3)
            RuntimeError: If inference fails

        Performance:
            - Inference time: <100ms (target)
            - Throughput: >10 FPS (target)
            - Accuracy: >95% per color (target)

        Notes:
            - Max 15 detections per frame (cube count limit)
            - Confidence threshold: 0.5 (configurable)
            - NMS applied to remove duplicate detections
        """
        ...

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for neural network

        Args:
            image: [512, 512, 3] uint8 RGB

        Returns:
            [3, 512, 512] float32 tensor, normalized [0,1]

        Processing:
            - Convert uint8 [0,255] → float32 [0,1]
            - Transpose HWC → CHW
            - Add batch dimension [1, 3, 512, 512]
        """
        ...

    def segment_candidates(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Use color-based segmentation (HSV) to find candidate regions

        Args:
            image: [512, 512, 3] RGB array

        Returns:
            List of bounding boxes for potential cubes

        Segmentation Strategy:
            - Convert RGB → HSV
            - Apply color thresholds:
                Green: H=[40°, 80°], S=[50, 255], V=[50, 255]
                Blue:  H=[100°, 130°], S=[50, 255], V=[50, 255]
                Red:   H=[0°, 10°] ∪ [170°, 180°], S=[50, 255], V=[50, 255]
            - Find contours
            - Filter by area (>100 pixels, <10000 pixels)
            - Return bounding boxes

        Note: HSV segmentation provides region proposals,
              CNN classifies color with higher accuracy
        """
        ...

    def classify_region(self, image: np.ndarray, bbox: BoundingBox) -> CubeObservation:
        """
        Classify cube color in bounding box region

        Args:
            image: [512, 512, 3] RGB array
            bbox: Region to classify

        Returns:
            CubeObservation with predicted color + confidence

        Processing:
            - Crop image to bbox
            - Resize to network input size (if needed)
            - Run CNN inference
            - Apply softmax to get class probabilities
            - Return color with highest probability
        """
        ...

    def estimate_distance(self, bbox: BoundingBox, cube_size: float = 0.05) -> float:
        """
        Estimate cube distance from camera using known size

        Args:
            bbox: Bounding box of detected cube
            cube_size: Known cube side length (meters, default 5cm)

        Returns:
            Estimated distance in meters

        Formula:
            distance = (focal_length × real_size) / pixel_size
            where:
                focal_length = 462 pixels (from camera.fov = 0.68 rad)
                real_size = 0.05 meters (cube side)
                pixel_size = sqrt(bbox.area)

        Note: Assumes cube is roughly square in image (frontal view)
        """
        ...

    def estimate_angle(self, bbox: BoundingBox) -> float:
        """
        Estimate cube bearing angle from robot center

        Args:
            bbox: Bounding box of detected cube

        Returns:
            Angle in degrees [-90, 90]
                0° = directly ahead
                Positive = right
                Negative = left

        Formula:
            angle = arctan((center_x - image_center) / focal_length)
            where:
                center_x = bbox.center[0]
                image_center = 256 (for 512×512 image)
                focal_length = 462 pixels
        """
        ...

    def non_max_suppression(self, observations: List[CubeObservation],
                           iou_threshold: float = 0.5) -> List[CubeObservation]:
        """
        Remove duplicate detections using NMS

        Args:
            observations: List of cube detections
            iou_threshold: IoU threshold for considering overlap

        Returns:
            Filtered list with duplicates removed

        Algorithm:
            1. Sort by confidence (descending)
            2. Keep highest confidence detection
            3. Remove all overlapping detections (IoU > threshold)
            4. Repeat for remaining detections
        """
        ...

    def get_fps(self) -> float:
        """
        Get average processing FPS over last 100 frames

        Returns:
            Frames per second

        Target: >10 FPS
        """
        ...


# Example Usage
if __name__ == "__main__":
    # Initialize detector
    detector = CubeDetector(model_path="models/cube_detector.pt")

    # Simulate camera image (512×512 RGB)
    image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

    # Detect cubes
    cubes = detector.detect(image)

    # Process results
    for cube in cubes:
        if cube.is_valid(min_confidence=0.7):
            print(f"Detected {cube.color} cube at {cube.distance:.2f}m, "
                  f"angle {cube.angle:.1f}°, confidence {cube.confidence:.2%}")

    # Find nearest green cube
    green_cubes = [c for c in cubes if c.color == "green"]
    if green_cubes:
        nearest = min(green_cubes, key=lambda c: c.distance)
        print(f"Nearest green cube: {nearest.distance:.2f}m")
