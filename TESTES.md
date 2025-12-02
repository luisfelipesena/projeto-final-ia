# TESTES — Passo a passo para validar o YouBot Fuzzy no Webots

Este roteiro garante que cada teste esteja alinhado aos requisitos do professor, ao guia CLAUDE.md e à documentação oficial do Webots e DataCamp.

## 1. Preparação do Ambiente
1. **Abrir o Webots R2025a** (mesma versão referenciada nos arquivos `IA_20252`).
2. **Carregar o mundo do professor**: `File > Open World…` e selecionar `IA_20252/worlds/IA_20252.wbt`.
3. **Conferir Supervisor**: na árvore de cena, verificar o nó `Robot (Supervisor)` com controle `supervisor` ativo (não alterar conforme requisito).

## 2. Apontar o Novo Controlador
1. Localizar o nó `Youbot` na árvore.
2. Em *Controller*, escolher `youbot_fuzzy` (o arquivo `IA_20252/controllers/youbot_fuzzy/youbot_fuzzy.py`).
3. Garantir que `controllerArgs` esteja vazio (spawn aleatório deve permanecer inalterado).

## 3. Checks Iniciais (sensores)
1. Na aba `Devices` do Youbot, confirmar que `lidar` e `camera` estão presentes e habilitados (conforme doc Lidar/Camera Webots).
2. Ajustar velocidade da simulação para `1.00x` para observação realista.
3. Iniciar a simulação (`Run`). Verificar no console se aparece log `youbot_fuzzy` (se `ENABLE_LOGGING` estiver `True`).

## 4. Validação Etapa a Etapa
### 4.1 Spawn e Ambiente
- Observar se 15 cubos aparecem distribuídos dentro de `x ∈ [-3, 1.75]` e `y ∈ [-1, 1]`. Se não, revisar logs do supervisor (requisito do professor).

### 4.2 LIDAR
1. Abrir o gráfico de `lidar` (`Robot Window > Lidar View`).
2. Confirmar que o campo de visão cobre ~180° e que distâncias frontais variam ao aproximar de obstáculos.
3. No console, procurar linhas que mostrem `front=`, `left=`, `right=`. Verificar se os valores diminuem ao dirigir em direção às caixas de madeira.

### 4.3 Câmera/Percepção
1. Ativar a visualização da câmera (`Display Camera`).
2. Posicionar o robô de frente para um cubo; observar se o log imprime `cube=GREEN`/`BLUE`/`RED` com `conf > 0.005` (DataCamp: pertinências dependem da cobertura).
3. Se não detectar, ajustar iluminação no Webots (Tutorial 3) ou revisar `HSV_RANGES` no `config.py`.

### 4.4 Planner Fuzzy
1. A cada log (intervalo de 10 steps), verificar se `vx`, `vy`, `omega` estão sendo calculados; ao detectar obstáculo próximo, `front_distance` deve ficar < 0.35 e `omega` alterar sinal, conforme regras.
2. Testar manualmente: posicionar o robô entre dois obstáculos e observar se ele recua (regra “avoidance”).

### 4.5 Manipulação
1. Quando um cubo alinhado for detectado, observar se `ArmService` executa sequência (`FLOOR` → `GRIP`).
2. Confirmar no console se, após `GRIP`, `MissionState.load_state` passa a `True`.
3. Aproximar o robô da caixa da cor correspondente e confirmar `lift_request="PLATE"` seguido de `gripper_request="RELEASE"` antes de `load_state` voltar a `False`.

## 5. Cenários de Teste Sugeridos
1. **Percurso completo**: deixar o robô coletar ao menos um cubo de cada cor, validando fluxo “buscar → coletar → depositar → avançar meta”.
2. **Stress Test**: alterar temporariamente `n_objects` (argumento do supervisor) para 20 e checar se não ocorrem colisões (distâncias nunca < 0.1 m).
3. **Iluminação**: alterar luz do mundo e confirmar se a classificação HSV ainda funciona; se falhar, considerar ANN (já previsto em `CubeDetector`).
4. **Log Inspection**: após rodar o teste, salvar o console para comprovar métricas (será necessário no vídeo de entrega).

## 6. Troubleshooting Com Base nas Referências
- **Lidar valores inconsistentes**: revisar doc `Lidar` (Cyberbotics) para garantir que `minRange` > 0 e `sampling_period` seja múltiplo do `basicTimeStep` (16 ms).
- **Garra não alcança**: usar presets validados em `draft.md` (FRONT_FLOOR, FRONT_PLATE) e confirmar `ARM_WAIT_SECONDS`.
- **Regras Fuzzy não respondem**: reabrir `control/fuzzy_planner.py` e ajustar membership functions triangulares/trapezoidais (conforme DataCamp) com base nos logs coletados.

## 7. Checklist Final para o Vídeo
1. Mostrar visão de cima da arena, confirmando spawn aleatório + caixas.
2. Exibir overlay da câmera e console com logs `front=…, cube=…`.
3. Demonstrar um ciclo completo de coleta e depósito para cada cor.
4. Explicar brevemente (sem mostrar código) a arquitetura Domain Driven + Lidar/Câmera + Fuzzy (seguindo CLAUDE.md).
