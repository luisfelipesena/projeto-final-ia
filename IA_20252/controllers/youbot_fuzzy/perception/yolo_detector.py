"""YOLO-based detector for cubes and boxes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:  # pragma: no cover
    UltralyticsYOLO = None  # type: ignore

from .. import config
from ..types import Detection


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path: Optional[Path] = None, confidence: Optional[float] = None):
        self._confidence = confidence or config.YOLO_CONFIDENCE_THRESHOLD
        self._model_path = Path(model_path or config.YOLO_MODEL_PATH)
        self._model = None
        self._names = {}
        self._available = config.ENABLE_YOLO and UltralyticsYOLO is not None and self._model_path.is_file()
        if self._available:
            try:
                self._model = UltralyticsYOLO(str(self._model_path))  # type: ignore[call-arg]
                self._names = getattr(self._model, "names", {}) or {}
            except Exception:
                self._model = None
                self._available = False

    def available(self) -> bool:
        return self._model is not None

    def detect(self, image) -> List[Detection]:
        if not self.available() or image is None or np is None:
            return []
        try:  # pragma: no cover - heavy external dependency
            results = self._model.predict(source=image, imgsz=image.shape[0], conf=self._confidence, verbose=False)  # type: ignore[operator]
        except Exception:
            return []
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf_tensor = getattr(box, "conf", None)
                if conf_tensor is None:
                    continue
                conf = float(conf_tensor.item()) if hasattr(conf_tensor, "item") else float(conf_tensor)
                if conf < self._confidence:
                    continue
                cls_attr = getattr(box, "cls", None)
                label = None
                if cls_attr is not None:
                    cls_id = int(cls_attr.item()) if hasattr(cls_attr, "item") else int(cls_attr)
                    label = str(self._names.get(cls_id, "")) if self._names else str(cls_id)
                xyxy = getattr(box, "xyxy", None)
                if xyxy is None:
                    continue
                coords = xyxy[0].cpu().numpy() if hasattr(xyxy[0], "cpu") else xyxy[0]
                x1, y1, x2, y2 = [int(v) for v in coords]
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections
