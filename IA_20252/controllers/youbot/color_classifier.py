"""
Neural Network Color Classifier for YouBot cube detection.
Uses ONNX model for inference with fallback to HSV heuristics.
"""

import os
import math

# Tentar importar dependências opcionais
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ColorClassifier:
    """Classificador de cores usando rede neural (ONNX) ou fallback HSV."""

    CLASSES = ["red", "green", "blue"]

    # Normalização ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, model_path=None):
        """Inicializa o classificador.

        Args:
            model_path: Caminho para o modelo ONNX. Se None ou não existe,
                       usa fallback HSV.
        """
        self.session = None
        self.input_name = None
        self.use_nn = False

        if model_path and HAS_ONNX and HAS_NUMPY:
            full_path = model_path
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(base_dir, model_path)

            if os.path.exists(full_path):
                try:
                    self.session = ort.InferenceSession(
                        full_path,
                        providers=['CPUExecutionProvider']
                    )
                    self.input_name = self.session.get_inputs()[0].name
                    self.use_nn = True
                    print(f"[RNA] Modelo carregado: {full_path}")
                except Exception as e:
                    print(f"[RNA] Erro ao carregar modelo: {e}")
                    self.use_nn = False
            else:
                print(f"[RNA] Modelo não encontrado: {full_path}, usando fallback HSV")
        else:
            if not HAS_NUMPY:
                print("[RNA] numpy não disponível, usando fallback HSV")
            if not HAS_ONNX:
                print("[RNA] onnxruntime não disponível, usando fallback HSV")

    def preprocess(self, image):
        """Preprocessa imagem para inferência.

        Args:
            image: Array numpy (H, W, 3) RGB com valores 0-255

        Returns:
            Array numpy (1, 3, 64, 64) normalizado
        """
        if not HAS_NUMPY:
            return None

        # Resize para 64x64
        if image.shape[0] != 64 or image.shape[1] != 64:
            try:
                import cv2
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # Fallback: nearest neighbor manual
                h, w = image.shape[:2]
                new_img = np.zeros((64, 64, 3), dtype=image.dtype)
                for i in range(64):
                    for j in range(64):
                        src_i = min(int(i * h / 64), h - 1)
                        src_j = min(int(j * w / 64), w - 1)
                        new_img[i, j] = image[src_i, src_j]
                image = new_img

        # Converter para float e normalizar
        img = image.astype(np.float32) / 255.0

        # Aplicar normalização ImageNet
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - self.MEAN[c]) / self.STD[c]

        # Transpor para (C, H, W) e adicionar batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def predict(self, image):
        """Classifica a cor de um cubo na imagem.

        Args:
            image: Array numpy (H, W, 3) RGB com valores 0-255,
                   ou tupla (r, g, b) normalizada 0-1 para fallback

        Returns:
            tuple: (class_name, confidence) onde class_name é "red"/"green"/"blue"/None
        """
        # Se NN disponível e imagem é numpy array
        if self.use_nn and HAS_NUMPY and isinstance(image, np.ndarray):
            try:
                preprocessed = self.preprocess(image)
                if preprocessed is None:
                    return self._fallback_hsv(image)

                outputs = self.session.run(None, {self.input_name: preprocessed})
                logits = outputs[0][0]

                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                class_idx = np.argmax(probs)
                confidence = probs[class_idx]

                return self.CLASSES[class_idx], float(confidence)

            except Exception as e:
                print(f"[RNA] Erro na inferência: {e}")
                return self._fallback_hsv(image)

        # Fallback para HSV/RGB simples
        return self._fallback_hsv(image)

    def _fallback_hsv(self, image):
        """Classificação por heurística HSV/RGB.

        Args:
            image: Array numpy ou tupla (r, g, b)

        Returns:
            tuple: (class_name, confidence)
        """
        # Se é tupla RGB normalizada (0-1)
        if isinstance(image, (tuple, list)) and len(image) >= 3:
            r, g, b = image[0], image[1], image[2]
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            # Calcular média das cores
            if image.ndim == 3:
                r = np.mean(image[:, :, 0]) / 255.0
                g = np.mean(image[:, :, 1]) / 255.0
                b = np.mean(image[:, :, 2]) / 255.0
            else:
                return None, 0.0
        else:
            return None, 0.0

        # Heurística simples
        if r > 0.5 and g < 0.4 and b < 0.4:
            conf = min(1.0, r - max(g, b) + 0.5)
            return "red", conf
        elif g > 0.5 and r < 0.4 and b < 0.4:
            conf = min(1.0, g - max(r, b) + 0.5)
            return "green", conf
        elif b > 0.5 and r < 0.4 and g < 0.4:
            conf = min(1.0, b - max(r, g) + 0.5)
            return "blue", conf

        return None, 0.0

    def predict_from_rgb(self, r, g, b):
        """Classificação direta de valores RGB normalizados.

        Args:
            r, g, b: Valores RGB normalizados (0-1)

        Returns:
            tuple: (class_name, confidence)
        """
        return self._fallback_hsv((r, g, b))


def color_from_rgb_nn(classifier, r, g, b):
    """Wrapper para compatibilidade com código existente.

    Args:
        classifier: Instância de ColorClassifier
        r, g, b: Valores RGB normalizados (0-1)

    Returns:
        str ou None: Nome da cor detectada
    """
    color, confidence = classifier.predict_from_rgb(r, g, b)
    if confidence > 0.5:
        return color
    return None
