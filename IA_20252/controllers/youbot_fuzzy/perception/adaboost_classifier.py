"""AdaBoost-based color classifier using OpenCV features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

import config


class AdaBoostColorClassifier:
    """Wrapper autour d'un modèle AdaBoost entraîné pour cores."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = Path(model_path or config.ADABOOST_MODEL_PATH)
        self._model = None
        self._hog = None
        self._enabled = config.ENABLE_ADABOOST and cv2 is not None and joblib is not None and np is not None
        if self._enabled and self._model_path.is_file():
            try:
                self._model = joblib.load(self._model_path)  # type: ignore[arg-type]
                self._hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)  # type: ignore[operator]
            except Exception:
                self._model = None
                self._hog = None
                if config.ENABLE_LOGGING:
                    print(f"[WARN] Failed to load AdaBoost model at {self._model_path}, disabling AdaBoost.")
                self._enabled = False
        elif self._enabled:
            if config.ENABLE_LOGGING:
                print(f"[WARN] AdaBoost model not found at {self._model_path}, using HSV fallback.")
            self._enabled = False

    def available(self) -> bool:
        return self._model is not None

    def predict(self, patch) -> Optional[str]:
        if not self.available() or patch is None or np is None or cv2 is None:
            return None
        features = self._extract_features(patch)
        if features is None:
            return None
        try:
            prediction = self._model.predict([features])  # type: ignore[union-attr]
        except Exception:
            return None
        return str(prediction[0])

    def _extract_features(self, patch):
        if np is None or cv2 is None:
            return None
        resized = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)  # type: ignore[operator]
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)  # type: ignore[operator]
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])  # type: ignore[operator]
        cv2.normalize(hist, hist)  # type: ignore[operator]
        hog_vec = None
        if self._hog is not None:
            hog_vec = self._hog.compute(resized)
        if hog_vec is None:
            hog_vec = resized.flatten().astype("float32") / 255.0
        features = np.concatenate([hist.flatten(), hog_vec.flatten()])
        return features.tolist()
