# Plano de Rede Neural para Classificação de Cor dos Cubos (Webots YouBot)

## Objetivo
Treinar e integrar um classificador de cores (red/green/blue) para cubos de 3 cm no cenário Webots, usando modelo pré-treinado leve e exportado para ONNX, inferindo no controller Python (`youbot.py`). Depósitos têm posições fixas, então apenas a cor é necessária.

## Escolha de Framework e Pipeline
- **Treino:** PyTorch (mais simples para fine-tuning e exporta fácil para ONNX).
- **Inferência no Webots:** `onnxruntime` + `opencv-python-headless` (rápido, sem GPU obrigatória).
- **Modelo base:** CNN leve (ex.: MobileNetV3-Small ou uma CNN custom pequena) pré-treinada em ImageNet, re-treinada só na cabeça de classificação (3 classes).

## Coleta de Dados
1. Habilitar salvamento de frames da câmera frontal (128x128) durante a simulação.
2. Automatizar captura:
   - Log de bounding boxes dos cubos via máscara HSV (ranges já usados no código) para gerar labels rápidas.
   - Salvar ROI 64×64 centrada no cubo, com pasta por classe (`dataset/{red,green,blue}`).
   - Incluir negativos (`dataset/none`) para reduzir falsos positivos.
3. Diversidade:
   - Distâncias: 0.2 m a 1.2 m.
   - Ângulos: -25° a +25°.
   - Iluminação: variar brilho/contraste no Webots e via augmentation.

## Pré-processamento e Augmentation
- Redimensionar ROI para 64×64.
- Normalizar (mean/std de ImageNet se usar backbone pré-treinado).
- Augmentations: rotação ±15°, brilho/contraste, leve blur, flip horizontal opcional.

## Treino (PyTorch)
1. Carregar backbone (MobileNetV3-Small) com pesos ImageNet.
2. Congelar camadas iniciais; treinar apenas a cabeça (últimas camadas) para 3 classes.
3. Hiperparâmetros sugeridos:
   - Epochs: 10–20
   - Batch: 64
   - LR: 1e-3 com Cosine/Step decay
   - Loss: CrossEntropy
4. Métricas: acurácia por classe, matriz de confusão, F1 macro.

## Exportação
1. Converter modelo para ONNX (`model.onnx`) com entrada `[1,3,64,64]`.
2. Salvar também mean/std usados.
3. Guardar em `IA_20252/controllers/youbot/model/`.

## Integração no Controller
1. Dependências no Python do Webots: `pip install onnxruntime opencv-python-headless numpy`.
2. No `ColorClassifier`:
   - Carregar `model.onnx` uma vez no `__init__`.
   - Ao detectar um contorno válido (já feito via HSV/bbox), recortar ROI, redimensionar 64×64, normalizar com mean/std.
   - Rodar `session.run` do onnxruntime e obter softmax; escolher classe com maior probabilidade.
   - Fallback: se confiança < threshold (ex. 0.5), usar HSV ranges atuais.
3. Output final (red/green/blue) seleciona o depósito (posições já fixas no código).

## Validação em Simulação
1. Rodar supervisor para 15 cubos aleatórios.
2. Log: predição da cor, confiança, bbox, decisão de depósito.
3. Avaliar taxa de acerto e tempo por frame (alvo: < 5 ms em CPU).
4. Ajustar threshold de confiança e augmentation se houver confusão entre vermelho/verde ou azul em baixa luz.

## Manutenção
- Se mudar iluminação/câmera/FOV, recapturar dataset e re-treinar só a cabeça (few epochs).
- Versão do modelo e hash do arquivo no README curto para rastrear.

## Alternativa Mais Leve (se onnxruntime indisponível)
- Usar `cv2.ml.Boost` ou SVM com histograma HSV (34 dims) treinado offline, salvar em `model.yml`, carregar com OpenCV ML. Menos robusto, porém sem onnxruntime.
