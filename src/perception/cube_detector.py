"""
Cube detection using HSV color segmentation.

Detects colored cubes (red, green, blue) in camera images
and estimates their distance and angle relative to the robot.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("[PERCEPTION] Warning: OpenCV not available, cube detection disabled")

from utils.config import (
    HSV_RANGES,
    MIN_CUBE_AREA,
    CUBE_REAL_SIZE,
    FOCAL_LENGTH_PIXELS,
    CAMERA,
)


@dataclass
class CubeDetection:
    """Result of cube detection."""
    color: str                    # 'red', 'green', or 'blue'
    center_x: int                 # Center X in image pixels
    center_y: int                 # Center Y in image pixels
    width: int                    # Bounding box width
    height: int                   # Bounding box height
    area: int                     # Contour area in pixels
    distance: float               # Estimated distance in meters
    angle: float                  # Angle from center (degrees, + = right)
    confidence: float             # Detection confidence [0-1]


class CubeDetector:
    """Detects colored cubes using HSV color segmentation."""

    # Build marker to verify code is loaded
    _BUILD_ID = "2024-12-02-v12-aspect1.6-ts1211"

    def __init__(self, camera_width: int = None, camera_height: int = None):
        """Initialize detector.

        Args:
            camera_width: Image width (default from config)
            camera_height: Image height (default from config)
        """
        print(f"[CubeDetector] Initialized (BUILD: {self._BUILD_ID}, MAX_CUBE_PIXELS=35, aspect<1.6)")
        self.width = camera_width or CAMERA.WIDTH
        self.height = camera_height or CAMERA.HEIGHT
        self.center_x = self.width // 2

        # Focal length for distance estimation
        self.focal_length = FOCAL_LENGTH_PIXELS

    def _create_color_mask(self, hsv_image: np.ndarray, color: str) -> np.ndarray:
        """Create binary mask for specified color.

        Args:
            hsv_image: Image in HSV color space
            color: Color name ('red', 'green', 'blue')

        Returns:
            Binary mask where color is present
        """
        if color not in HSV_RANGES:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        ranges = HSV_RANGES[color]
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            partial_mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.bitwise_or(mask, partial_mask)

        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def _estimate_distance(self, apparent_size: float) -> float:
        """Estimate distance using apparent size.

        Uses pinhole camera model: distance = (real_size * focal_length) / apparent_size

        Args:
            apparent_size: Size in pixels (larger of width/height)

        Returns:
            Estimated distance in meters
        """
        if apparent_size <= 0:
            return float('inf')

        distance = (CUBE_REAL_SIZE * self.focal_length) / apparent_size

        # Sanity check: robot can't grasp closer than ~7cm (arm reach limit)
        # Cubes appearing at <5cm are likely detection errors
        if distance < 0.05:
            return float('inf')

        return distance

    def _estimate_angle(self, center_x: int) -> float:
        """Estimate horizontal angle from image center.

        Args:
            center_x: X coordinate of detection center

        Returns:
            Angle in degrees (positive = right of center)
        """
        # Pixel offset from center
        offset = center_x - self.center_x

        # Convert to angle using FOV
        # For 57° FOV across 128 pixels: ~0.445°/pixel
        degrees_per_pixel = CAMERA.FOV_DEGREES / self.width
        angle = offset * degrees_per_pixel

        return angle

    def _find_cubes_of_color(
        self, image: np.ndarray, hsv_image: np.ndarray, color: str
    ) -> List[CubeDetection]:
        """Find all cubes of specified color.

        Args:
            image: Original BGR image
            hsv_image: Image in HSV color space
            color: Color to detect

        Returns:
            List of CubeDetection objects
        """
        mask = self._create_color_mask(hsv_image, color)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CUBE_AREA:
                continue

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio (cubes should be SQUARE, aspect ratio ~1.0)
            aspect_ratio = max(w, h) / (min(w, h) + 0.001)
            if aspect_ratio > 1.6:  # Stricter: cubes are square, boxes are not
                print(f"[CubeDetector] REJECT {color}: aspect={aspect_ratio:.2f} > 1.6, size={max(w,h)}px")
                continue

            # Filter out detections that are too large (likely deposit boxes)
            MAX_CUBE_PIXELS = 35  # Tighter: cube at 0.3m is ~35px max
            if max(w, h) > MAX_CUBE_PIXELS:
                continue  # Too large

            # Filter: area should be close to bounding box (regular shape)
            expected_area = w * h
            if area > expected_area * 1.3:  # Irregular shape = not a cube
                continue

            # Center point
            center_x = x + w // 2
            center_y = y + h // 2

            # Filter: cubes on ground appear in lower portion of image (y > 35%)
            # Relaxed to allow detection during scan
            if center_y < self.height * 0.35:
                continue  # Too high in image

            # Distance estimation using larger dimension
            apparent_size = max(w, h)
            distance = self._estimate_distance(apparent_size)

            # Angle from center
            angle = self._estimate_angle(center_x)

            # Confidence based on area and shape
            # Larger, more square objects get higher confidence
            solidity = area / (w * h + 0.001)
            size_factor = min(area / 500, 1.0)  # Larger = more confident
            confidence = solidity * size_factor * (1.0 / aspect_ratio)

            detection = CubeDetection(
                color=color,
                center_x=center_x,
                center_y=center_y,
                width=w,
                height=h,
                area=area,
                distance=distance,
                angle=angle,
                confidence=min(confidence, 1.0),
            )
            detections.append(detection)

        return detections

    def detect(self, image: np.ndarray) -> List[CubeDetection]:
        """Detect all colored cubes in image.

        Args:
            image: BGR image from camera (numpy array)

        Returns:
            List of CubeDetection sorted by distance (nearest first)
        """
        if cv2 is None:
            return []

        if image is None or image.size == 0:
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Find cubes of each color
        all_detections = []
        for color in HSV_RANGES.keys():
            detections = self._find_cubes_of_color(image, hsv, color)
            all_detections.extend(detections)

        # Sort by distance (nearest first)
        all_detections.sort(key=lambda d: d.distance)

        return all_detections

    def detect_nearest(self, image: np.ndarray) -> Optional[CubeDetection]:
        """Find the nearest cube of any color.

        Args:
            image: BGR image from camera

        Returns:
            Nearest CubeDetection or None if no cubes found
        """
        detections = self.detect(image)
        return detections[0] if detections else None

    def detect_color(self, image: np.ndarray, color: str) -> Optional[CubeDetection]:
        """Find the nearest cube of specific color.

        Args:
            image: BGR image from camera
            color: Color to find ('red', 'green', 'blue')

        Returns:
            Nearest CubeDetection of that color or None
        """
        if cv2 is None:
            return None

        if image is None or image.size == 0:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = self._find_cubes_of_color(image, hsv, color)

        if not detections:
            return None

        # Return nearest
        detections.sort(key=lambda d: d.distance)
        return detections[0]

    @staticmethod
    def webots_image_to_numpy(image_data, width: int, height: int) -> np.ndarray:
        """Convert Webots camera image to numpy array.

        Args:
            image_data: Raw bytes from camera.getImage()
            width: Image width
            height: Image height

        Returns:
            BGR numpy array suitable for OpenCV
        """
        # Webots returns BGRA format
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        # Convert BGRA to BGR
        return image[:, :, :3]
