"""
CubeDetector Module

Camera-based cube detection and color classification using CNN.
Based on: Redmon et al. (2016) - YOLO, Custom Lightweight CNN (DECISAO 017)

Contract: specs/003-neural-networks/contracts/cube_detector.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import cv2


@dataclass
class CubeDetection:
    """
    Single cube detection result

    Attributes:
        color: Detected color ('green', 'blue', 'red')
        confidence: Classification confidence [0, 1]
        bbox: Bounding box (x_center, y_center, width, height) normalized [0, 1]
        distance: Estimated distance to cube (meters)
        angle: Angle to cube center from camera axis (degrees, negative=left)
    """
    color: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h) normalized
    distance: float
    angle: float

    @property
    def is_valid(self) -> bool:
        """Check if detection meets minimum confidence threshold"""
        return self.confidence >= 0.5


class ColorSegmenter:
    """
    HSV-based color segmentation for cube detection

    Robust fallback when CNN is unavailable or for initial detection.
    Uses calibrated HSV ranges for green, blue, red cubes.
    """

    # HSV ranges calibrated for Webots simulation lighting
    COLOR_RANGES = {
        'green': {
            'lower': np.array([35, 100, 100]),
            'upper': np.array([85, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 100, 100]),
            'upper': np.array([130, 255, 255])
        },
        'red': {
            # Red wraps around in HSV, need two ranges
            'lower1': np.array([0, 100, 100]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([160, 100, 100]),
            'upper2': np.array([180, 255, 255])
        }
    }

    MIN_CONTOUR_AREA = 1500  # pixels - increased to filter noise
    MAX_CONTOUR_AREA = 15000  # pixels - filter out large objects (deposit boxes)

    def segment(self, image: np.ndarray) -> List[CubeDetection]:
        """
        Segment cubes by color in image

        Args:
            image: RGB image [H, W, 3] uint8

        Returns:
            List of CubeDetection for each detected cube
        """
        detections = []

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h, w = image.shape[:2]

        for color_name, ranges in self.COLOR_RANGES.items():
            if color_name == 'red':
                # Red needs two masks
                mask1 = cv2.inRange(hsv, ranges['lower1'], ranges['upper1'])
                mask2 = cv2.inRange(hsv, ranges['lower2'], ranges['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])

            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_CONTOUR_AREA:
                    continue
                # Filter out large objects (deposit boxes are much bigger than cubes)
                if area > self.MAX_CONTOUR_AREA:
                    continue

                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(contour)

                # Aspect ratio filter - cubes should be roughly square
                bbox_aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
                if bbox_aspect < 0.4:  # Too elongated, not a cube
                    continue

                # Calculate normalized bbox center
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bbox_w = bw / w
                bbox_h = bh / h

                # Estimate distance based on bbox size (larger = closer)
                # Calibrated for 5cm cube: at 1m apparent_size ~0.05, at 0.5m ~0.10
                apparent_size = max(bbox_w, bbox_h)
                CUBE_REAL_SIZE = 0.05  # 5cm cube
                MIN_APPARENT_SIZE = 0.02  # Minimum valid detection size

                if apparent_size > MIN_APPARENT_SIZE:
                    # distance = real_size / (apparent_size * tan(FOV/2))
                    # tan(30°) = 0.577 for 60° horizontal FOV
                    distance = CUBE_REAL_SIZE / (apparent_size * 0.577)
                    distance = np.clip(distance, 0.15, 3.0)  # Min 15cm, max 3m
                else:
                    distance = 3.0  # Far away or noise

                # Calculate angle from center of image
                # Camera FOV is approximately 60 degrees
                angle = (cx - 0.5) * 60.0  # degrees from center

                # Confidence based on contour shape (square-ness)
                # Base confidence 0.6 + up to 0.3 bonus for squareness
                rect = cv2.minAreaRect(contour)
                aspect_ratio = min(rect[1]) / max(rect[1]) if max(rect[1]) > 0 else 0
                confidence = 0.6 + (aspect_ratio * 0.3)  # More stable detection

                detections.append(CubeDetection(
                    color=color_name,
                    confidence=confidence,
                    bbox=(cx, cy, bbox_w, bbox_h),
                    distance=distance,
                    angle=angle
                ))

        return detections


class CubeDetector:
    """
    Neural network-based cube detection and color classification

    Combines CNN color classification with HSV segmentation for
    robust cube detection.

    Contract Requirements:
    - MUST achieve >95% color accuracy (SC-002)
    - MUST process at >10 FPS (SC-004)
    - MUST handle 512x512 input images
    - MUST return CubeDetection with distance and angle estimates

    Usage:
        detector = CubeDetector(model_path="models/camera_net.pt")
        image = camera.getImage()  # RGB [512, 512, 3]
        detections = detector.detect(image)

        for det in detections:
            if det.is_valid:
                print(f"Found {det.color} cube at {det.distance:.2f}m, angle {det.angle:.1f}°")
    """

    COLOR_CLASSES = ['green', 'blue', 'red']
    INPUT_SIZE = (512, 512)

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize cube detector

        Args:
            model_path: Path to trained TorchScript model (optional, uses HSV fallback if None)
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = None
        self.color_segmenter = ColorSegmenter()

        # Try to load CNN model
        if model_path:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    self.model = torch.jit.load(str(model_file), map_location=self.device)
                    self.model.eval()
                    print(f"CubeDetector: CNN model loaded from {model_path}")
                except Exception as e:
                    print(f"CubeDetector: Failed to load model: {e}, using HSV fallback")
            else:
                print(f"CubeDetector: Model not found at {model_path}, using HSV fallback")
        else:
            print("CubeDetector: No model path provided, using HSV segmentation")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CNN inference

        Args:
            image: RGB image [H, W, 3] uint8

        Returns:
            tensor: [1, 3, 512, 512] normalized float tensor
        """
        # Resize if needed
        if image.shape[:2] != self.INPUT_SIZE:
            image = cv2.resize(image, self.INPUT_SIZE)

        # Convert to float and normalize
        tensor = torch.from_numpy(image).float() / 255.0

        # [H, W, C] -> [C, H, W]
        tensor = tensor.permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.to(self.device)

    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify color of a cropped cube region using CNN

        Args:
            crop: RGB crop of cube region [H, W, 3]

        Returns:
            (color_name, confidence)
        """
        if self.model is None:
            # Fallback to dominant color analysis
            return self._classify_by_dominant_color(crop)

        # Resize crop to model input size
        crop_resized = cv2.resize(crop, self.INPUT_SIZE)
        tensor = self.preprocess(crop_resized)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        confidence, pred_idx = probs.max(dim=1)
        color = self.COLOR_CLASSES[pred_idx.item()]

        return color, confidence.item()

    def _classify_by_dominant_color(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify by analyzing dominant color in HSV space

        Args:
            crop: RGB crop [H, W, 3]

        Returns:
            (color_name, confidence)
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

        # Count pixels in each color range
        counts = {}
        for color_name, ranges in ColorSegmenter.COLOR_RANGES.items():
            if color_name == 'red':
                mask1 = cv2.inRange(hsv, ranges['lower1'], ranges['upper1'])
                mask2 = cv2.inRange(hsv, ranges['lower2'], ranges['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            counts[color_name] = np.sum(mask > 0)

        total = sum(counts.values())
        if total == 0:
            return 'green', 0.0  # Default fallback

        best_color = max(counts, key=counts.get)
        confidence = counts[best_color] / total

        return best_color, confidence

    def detect(self, image: np.ndarray) -> List[CubeDetection]:
        """
        Detect cubes in image

        Args:
            image: RGB image [H, W, 3] uint8

        Returns:
            List of CubeDetection sorted by distance (closest first)
        """
        # First, use HSV segmentation to find cube candidates
        detections = self.color_segmenter.segment(image)

        # If CNN model available, refine color classification
        if self.model is not None:
            h, w = image.shape[:2]
            refined_detections = []

            for det in detections:
                # Extract crop
                cx, cy, bw, bh = det.bbox
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                # Add padding
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                crop = image[y1:y2, x1:x2]

                if crop.size > 0:
                    color, confidence = self.classify_crop(crop)
                    refined_detections.append(CubeDetection(
                        color=color,
                        confidence=confidence,
                        bbox=det.bbox,
                        distance=det.distance,
                        angle=det.angle
                    ))
                else:
                    refined_detections.append(det)

            detections = refined_detections

        # Sort by distance (closest first)
        detections.sort(key=lambda d: d.distance)

        return detections

    def get_closest_cube(self, image: np.ndarray) -> Optional[CubeDetection]:
        """
        Get closest detected cube

        Args:
            image: RGB image [H, W, 3]

        Returns:
            Closest CubeDetection or None if no cubes found
        """
        detections = self.detect(image)
        valid = [d for d in detections if d.is_valid]
        return valid[0] if valid else None


def test_cube_detector():
    """Test cube detection"""
    print("Testing CubeDetector...")

    detector = CubeDetector()

    # Create synthetic test image with colored regions
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Add green cube
    image[100:200, 100:200] = [0, 255, 0]  # Green

    # Add blue cube
    image[100:200, 300:400] = [0, 0, 255]  # Blue

    # Add red cube
    image[300:400, 200:300] = [255, 0, 0]  # Red

    detections = detector.detect(image)

    print(f"  Found {len(detections)} cubes")
    for det in detections:
        print(f"    {det.color}: conf={det.confidence:.2f}, dist={det.distance:.2f}m, angle={det.angle:.1f}°")

    assert len(detections) >= 3, "Should detect at least 3 cubes"
    print("  CubeDetector test passed")


if __name__ == "__main__":
    test_cube_detector()
