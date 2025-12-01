"""
CubeDetector Module

Camera-based cube detection and color classification using CNN.
Based on: Redmon et al. (2016) - YOLO, Custom Lightweight CNN (DECISAO 017)

Contract: specs/003-neural-networks/contracts/cube_detector.py
"""

import math
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
    # Saturation lowered to 70 (was 80) - provides tolerance to desaturation
    # Value lowered to 40 (was 50) - handle mild shadows
    COLOR_RANGES = {
        'green': {
            'lower': np.array([35, 70, 40]),    # H:35-85, relaxed S/V for Webots
            'upper': np.array([85, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 70, 40]),   # H:100-130, relaxed S/V
            'upper': np.array([130, 255, 255])
        },
        'red': {
            # Red wraps around in HSV, need two ranges
            'lower1': np.array([0, 70, 40]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([160, 70, 40]),
            'upper2': np.array([180, 255, 255])
        }
    }

    # Cube detection thresholds
    # 5cm cube at 1m ≈ 300-600 pixels area, at 0.3m ≈ 15000 pixels, at 0.20m ≈ 25000 pixels
    MIN_CONTOUR_AREA = 120    # pixels - detect cubes at distance
    MAX_CONTOUR_AREA = 30000  # pixels - reject deposit boxes (they're huge)

    # Maximum bbox size as fraction of image
    # 5cm cube at 0.30m ≈ 15% of image, at 0.20m ≈ 25%, at 0.15m ≈ 35%
    # Deposit boxes are 40%+ of image - reject them
    MAX_BBOX_FRACTION = 0.35

    # Minimum aspect ratio (width/height or height/width)
    # Cubes are SQUARE (aspect ~1.0), deposit boxes are rectangular (aspect ~0.3-0.5)
    MIN_BBOX_ASPECT = 0.70

    # Cubes are on the floor - center_y should be in lower portion of image
    # But distant cubes appear higher (closer to horizon)
    MIN_CENTER_Y = 0.30  # Cube center must be below 30% from top (allow distant cubes)
    MAX_CENTER_Y = 0.95  # Ignore reflections at the extreme bottom edge

    MIN_SOLIDITY = 0.80   # Reject elongated/highly concave blobs (e.g., rims of bins)
    MIN_EXTENT = 0.65     # Ensures blob fills bounding box similar to a square

    # Reject extremely tall/wide blobs (deposit bins)
    MAX_PROJECTED_WIDTH = 0.35
    MAX_PROJECTED_HEIGHT = 0.35

    CAMERA_VERTICAL_FOV = math.radians(60.0)  # Webots default for youBot camera
    CUBE_REAL_SIZE = 0.05  # 5 cm
    CAMERA_OFFSET = 0.15   # Camera to arm-base offset

    # Debug: save images to diagnose detection issues
    DEBUG_SAVE_IMAGES = False  # Set True to save debug images
    _debug_frame_count = 0

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

        ColorSegmenter._debug_frame_count += 1

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
                if bbox_aspect < self.MIN_BBOX_ASPECT:  # Too elongated, not a cube
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < self.MIN_SOLIDITY:
                    continue

                extent = area / (bw * bh) if bw * bh > 0 else 0
                if extent < self.MIN_EXTENT:
                    continue

                # Maximum bbox size filter - deposit boxes are much larger than cubes
                # A cube taking up >35% of image is either too close or a false detection
                bbox_fraction = max(bw / w, bh / h)
                if bbox_fraction > self.MAX_BBOX_FRACTION:
                    continue

                if bw / w > self.MAX_PROJECTED_WIDTH or bh / h > self.MAX_PROJECTED_HEIGHT:
                    continue

                # Calculate normalized bbox center
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bbox_w = bw / w
                bbox_h = bh / h

                # Position filter: cubes are on the floor (lower part of image)
                # Deposit boxes are elevated and appear higher in the image
                if cy < self.MIN_CENTER_Y or cy > self.MAX_CENTER_Y:
                    continue

                apparent_pixels = max(bbox_w * w, bbox_h * h)
                distance = self._estimate_distance(apparent_pixels, h)

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

    @classmethod
    def _estimate_distance(cls, apparent_pixels: float, image_height: int) -> float:
        """
        Estimate distance using pinhole camera model.

        Args:
            apparent_pixels: Measured cube size in pixels (max dimension of bbox)
            image_height: Image height in pixels

        Returns:
            Distance in meters from arm base to cube
        """
        if apparent_pixels <= 1e-3:
            return 3.0

        focal_px = (image_height / 2.0) / math.tan(cls.CAMERA_VERTICAL_FOV / 2.0)
        camera_distance = (cls.CUBE_REAL_SIZE * focal_px) / apparent_pixels
        distance = camera_distance + cls.CAMERA_OFFSET
        return float(np.clip(distance, 0.10, 3.0))


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

        # Debug logging for color classification
        print(f"[ColorDebug] G={counts.get('green',0)} B={counts.get('blue',0)} R={counts.get('red',0)} → {best_color} ({confidence:.2f})")

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
