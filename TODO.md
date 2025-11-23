# Plano de Execução - YouBot Autônomo

**Projeto:** Sistema Autônomo de Coleta e Organização de Objetos
**Aluno:** Luis Felipe Cordeiro Sena
**Data Limite:** 06/01/2026, 23:59
**Status:** Planejamento Concluído

---

## 🎯 Objetivo Final

Desenvolver sistema autônomo para YouBot que:
- Coleta 15 cubos coloridos (verde, azul, vermelho) distribuídos aleatoriamente
- Identifica cor via câmera RGB
- Deposita em caixa correspondente
- Evita obstáculos usando LIDAR
- **SEM GPS** - navegação baseada apenas em sensores

**Requisitos Técnicos Obrigatórios:**
1. ✅ RNA (MLP ou CNN) para LIDAR/mapeamento
2. ✅ Lógica Fuzzy para controle de ações

---

## 📋 Fases do Projeto

### Fase 0: Setup e Documentação ✅
**Prazo:** Completado
**Status:** ✅ CONCLUÍDO

- [x] Estrutura de projeto criada
- [x] CLAUDE.md com contexto do projeto
- [x] REFERENCIAS.md com base científica
- [x] TODO.md (este arquivo) para planejamento
- [x] DECISIONS.md para rastreamento de decisões
- [x] .gitignore configurado

---

### Fase 1: Ambiente e Exploração Inicial
**Prazo:** 3 dias
**Objetivo:** Familiarização com Webots e sensores do YouBot

#### 1.1 Setup do Webots ✅ CONCLUÍDO
- [x] Instalar/atualizar Webots (manual) - R2023b instalado
- [x] Verificar versão do Python (compatibilidade com Webots) - Python 3.14.0 com venv
- [x] Testar abertura do mundo IA_20252.wbt - Carrega em ~5s, funcional
- [x] Verificar spawn de cubos pelo supervisor - 15/15 cubos spawnados com sucesso
- [x] Documentar setup em DECISIONS.md (DECISÃO 005-010 adicionadas)
- [x] Criar documentação de setup (specs/001-webots-setup/* completo)
- [x] Criar testes de validação automatizados (tests/test_webots_setup.py - 6/8 passed)

**Deliverable:** ✅ Simulação rodando sem erros | DECISÃO 010: World file R2025a funciona em R2023b com warnings não-críticos

#### 1.2 Exploração dos Controles Base ✅ CONCLUÍDO
- [x] Testar movimentos da base (forward, backward, strafe, rotate)
- [x] Testar comandos do braço (set_height, set_orientation)
- [x] Testar garra (grip, release)
- [x] Documentar limites de movimento
- [x] Criar script de teste básico: `tests/test_basic_controls.py`

**Deliverable:** ✅ Script de teste validando todos os controles (13/13 testes implementados)

#### 1.3 Análise dos Sensores ✅ CONCLUÍDO
- [x] **LIDAR:**
  - [x] Ler dados brutos (range_image)
  - [x] Entender formato (número de pontos, range, FOV)
  - [x] Visualizar varredura (matplotlib/plot)
  - [x] Identificar obstáculos na visualização
- [x] **Câmera RGB:**
  - [x] Capturar frames
  - [x] Verificar resolução e FPS
  - [x] Testar detecção de cores (threshold simples)
  - [x] Salvar imagens de exemplo
- [x] Criar notebook: `notebooks/01_sensor_exploration.ipynb`

**Deliverable:** ✅ Notebook com visualizações e análises dos sensores

#### 1.4 Mapeamento da Arena ✅ CONCLUÍDO
- [x] Medir dimensões da arena manualmente
- [x] Identificar posições aproximadas das caixas de depósito
- [x] Mapear distribuição típica de obstáculos
- [x] Documentar coordenadas em `docs/arena_map.md`

**Deliverable:** ✅ Mapa esquemático da arena (7.0×4.0m documentado)

**Referências Fase 1:**
- Bischoff et al. (2011): YouBot specifications
- Michel (2004): Webots documentation

---

### Fase 2: Percepção com Redes Neurais
**Prazo:** 10 dias
**Status:** 🟡 INFRAESTRUTURA COMPLETA - Falta treinamento e integração final
**Objetivo:** Implementar detecção de obstáculos (LIDAR) e classificação de cores (RGB)

**📦 COMPLETADO (PR #3):**
- [x] Infraestrutura de dados (coleta, anotação, augmentation, splits)
- [x] Arquiteturas de redes neurais implementadas
- [x] Data loaders PyTorch com augmentation
- [x] Scripts de validação e testes
- [x] Documentação completa (DECISÃO 016-017)

**📦 COMPLETADO (PR #4 - specs/002-script-updates):**
- [x] Scripts de coleta atualizados com suporte a mock e metadata externa
- [x] Scripts de anotação com auto-labeling (LIDAR threshold, Camera HSV)
- [x] Script de geração de manifests (`generate_dataset_manifest.py`)
- [x] Script de split atualizado para usar manifests (`split_dataset.py`)
- [x] Todos os scripts verificados com dados mock

**⚠️ PENDENTE (Retornar após Fase 3):**

#### 2.1 Processamento LIDAR com RNA

**2.1.1 Abordagem Implementada: Hybrid MLP + 1D-CNN** ✅
- [x] Arquitetura Hybrid: `src/perception/models/lidar_net.py`
  - [x] CNN branch: Conv1D(1→32→64→64) + GlobalAvgPool
  - [x] Hand-crafted features: min, mean, std, occupancy, symmetry, variance
  - [x] MLP classifier: Fusion(70→128→64→9) + Sigmoid
  - [x] ~250K parâmetros
- [x] LIDARProcessor: `src/perception/lidar_processor.py`
- [x] ObstacleMap: Estrutura 9-sector
- [x] Data augmentation: noise, dropout, rotation

**⚠️ FALTA EXECUTAR:**
- [ ] **T018:** Coletar 1000+ scans LIDAR no Webots
  ```bash
  python scripts/collect_lidar_data.py
  ```
- [ ] **T019:** Revisar/corrigir labels se necessário
  ```bash
  python scripts/annotate_lidar.py
  ```
- [ ] **T024-T025:** Criar notebook de treinamento LIDAR
  - [ ] `notebooks/lidar_training.ipynb`
  - [ ] Adam optimizer, BCE loss, 100-200 epochs
  - [ ] Early stopping (patience=20)
- [ ] **T026:** Treinar modelo e validar: >90% accuracy, <100ms latency
- [ ] **T027-T028:** Exportar modelo treinado
  - [ ] Salvar como TorchScript: `models/lidar_net.pt`
  - [ ] Salvar metadata: `models/lidar_net_metadata.json`

#### 2.1.3 Detecção de Obstáculos
- [x] LIDARProcessor com inference implementado
- [x] ObstacleMap com métodos de consulta
- [ ] **T029-T033:** Integrar no controller Webots (após treinamento)

**Deliverable:** ⏳ Módulo LIDAR funcionando com >90% precisão

#### 2.2 Detecção de Cubos com CNN

**2.2.1 Arquitetura Implementada: Lightweight CNN** ✅
- [x] **DECISÃO 017:** Custom Lightweight CNN escolhida
- [x] Arquitetura: `src/perception/models/camera_net.py`
  - [x] Conv2D(3→32→64→128) + BatchNorm + ReLU + MaxPool
  - [x] GlobalAvgPool + FC(128→64→3) + Dropout(0.5)
  - [x] ~250K parâmetros
- [x] Fallback: ResNet18 transfer learning (se accuracy <93%)
- [x] CameraDataset com augmentation
- [x] Data augmentation: brightness, hue, flip, rotation

**⚠️ FALTA EXECUTAR:**
- [ ] **T034:** Coletar 500+ imagens no Webots
  ```bash
  python scripts/collect_camera_data.py
  ```
- [ ] **T035:** Revisar/corrigir labels e bboxes
  ```bash
  python scripts/annotate_camera.py
  ```
- [ ] **T038-T039:** Criar notebook de treinamento Camera
  - [ ] `notebooks/camera_training.ipynb`
  - [ ] SGD+momentum, CrossEntropy loss, 30-50 epochs
  - [ ] StepLR scheduler, class weighting
- [ ] **T040:** Treinar e validar: >95% per-color, >10 FPS
- [ ] **T041:** Se accuracy <93%, usar ResNet18 fallback
- [ ] **T042-T043:** Exportar modelo treinado
  - [ ] Salvar como TorchScript: `models/camera_net.pt`
  - [ ] Salvar metadata: `models/camera_net_metadata.json`

**2.2.3 Detecção e Localização de Cubos**
- [ ] **T044-T048:** Implementar CubeDetector completo
  - [ ] HSV color segmentation para region proposals
  - [ ] Bbox + color classification
  - [ ] Distance estimation (focal_length=462, cube_size=0.05m)
  - [ ] Angle estimation (bearing)
  - [ ] Non-Max Suppression (IoU=0.5)
- [ ] **T049-T051:** Integrar no controller Webots (após treinamento)

**Deliverable:** ⏳ Detector de cubos com >95% precisão em cores

#### 2.3 Integração Percepção
- [ ] **T052-T067:** Implementar PerceptionSystem completo (Fase 5)
  - [ ] Classe que unifica LIDAR + Camera
  - [ ] Output estruturado: obstacles + cubes
  - [ ] WorldState tracking
  - [ ] Sensor fusion e filtros
- [ ] Implementar em: `src/perception/perception_system.py`
- [ ] Testes de integração: `tests/test_perception_integration.py`

**Deliverable:** ⏳ Sistema de percepção integrado e testado

**📝 NOTA:** Infraestrutura completa permite desenvolvimento paralelo de Fase 3 (Controle Fuzzy).
Retornar aqui após Fase 3 para executar coleta de dados e treinamento.

**Referências Fase 2:**
- Goodfellow et al. (2016): Deep Learning fundamentals
- Qi et al. (2017): PointNet architecture
- Redmon et al. (2016): YOLO detection
- Liu et al. (2016): SSD for small objects

---

### Fase 3: Controle com Lógica Fuzzy
**Prazo:** 7 dias
**Objetivo:** Implementar controlador fuzzy para navegação e ações

#### 3.1 Design do Controlador Fuzzy

**3.1.1 Definir Variáveis Linguísticas**

**Inputs:**
- [ ] `distance_to_obstacle`: {muito_perto, perto, medio, longe}
- [ ] `angle_to_obstacle`: {esquerda, centro, direita}
- [ ] `distance_to_cube`: {muito_perto, perto, medio, longe}
- [ ] `angle_to_cube`: {esquerda_forte, esquerda, centro, direita, direita_forte}
- [ ] `cube_detected`: {sim, nao} (crisp)
- [ ] `holding_cube`: {sim, nao} (crisp)

**Outputs:**
- [ ] `linear_velocity`: {parar, devagar, medio, rapido}
- [ ] `angular_velocity`: {esquerda_forte, esquerda, reto, direita, direita_forte}
- [ ] `action`: {buscar, aproximar, pegar, levar_caixa, soltar}

**Funções de Pertinência:**
- [ ] Definir funções (triangular, trapezoidal, gaussiana)
- [ ] Plotar e validar visualmente
- [ ] Documentar ranges em `docs/fuzzy_membership.md`

#### 3.1.2 Definir Regras Fuzzy**

Categorias de regras:
- [ ] **Evitação de obstáculos** (prioridade máxima):
  ```
  SE distance_to_obstacle É muito_perto ENTÃO linear_velocity É parar E angular_velocity É esquerda_forte
  SE distance_to_obstacle É perto E angle_to_obstacle É centro ENTÃO linear_velocity É devagar E angular_velocity É direita
  ```
- [ ] **Busca de cubos**:
  ```
  SE cube_detected É nao E obstacle_free ENTÃO action É buscar E linear_velocity É medio E angular_velocity É esquerda
  ```
- [ ] **Aproximação de cubos**:
  ```
  SE cube_detected É sim E distance_to_cube É longe ENTÃO action É aproximar E linear_velocity É medio
  SE distance_to_cube É perto ENTÃO linear_velocity É devagar
  SE distance_to_cube É muito_perto ENTÃO action É pegar
  ```
- [ ] **Navegação para caixa**:
  ```
  SE holding_cube É sim ENTÃO action É levar_caixa
  ```
- [ ] Criar arquivo: `src/control/fuzzy_rules.txt` com todas as regras

**Total de regras:** ~20-30 regras bem definidas

#### 3.1.3 Implementação
- [ ] Usar biblioteca `scikit-fuzzy`
- [ ] Implementar controlador Mamdani
- [ ] Métodos de defuzzificação: centroid
- [ ] Classe `FuzzyController` em: `src/control/fuzzy_controller.py`
- [ ] Testes unitários: `tests/test_fuzzy.py`

#### 3.2 Máquina de Estados
- [ ] Definir estados do robô:
  - [ ] `SEARCHING`: Procurando cubos
  - [ ] `APPROACHING`: Aproximando de cubo detectado
  - [ ] `GRASPING`: Pegando cubo
  - [ ] `NAVIGATING_TO_BOX`: Indo para caixa correspondente
  - [ ] `DEPOSITING`: Depositando cubo
  - [ ] `AVOIDING`: Evitando obstáculo (override)
- [ ] Transições entre estados
- [ ] Implementar em: `src/control/state_machine.py`

#### 3.3 Integração Controle
- [ ] Conectar fuzzy controller com state machine
- [ ] Input: dados de percepção
- [ ] Output: comandos para base, arm, gripper
- [ ] Implementar em: `src/control/robot_controller.py`

**Deliverable:** Controlador fuzzy funcional com máquina de estados

**Referências Fase 3:**
- Zadeh (1965): Fuzzy Sets theory
- Mamdani & Assilian (1975): Fuzzy controller
- Saffiotti (1997): Fuzzy navigation
- Antonelli et al. (2007): Path tracking

---

### Fase 4: Navegação e Path Planning
**Prazo:** 5 dias
**Objetivo:** Implementar estratégias de navegação eficientes

#### 4.1 Mapeamento Local
- [ ] Criar occupancy grid simplificado
  - [ ] Baseado em leituras LIDAR recentes
  - [ ] Atualização incremental
  - [ ] Resolução: 10cm x 10cm
- [ ] Marcar células: livre, ocupado, desconhecido
- [ ] Implementar em: `src/navigation/local_map.py`

#### 4.2 Planejamento de Trajetória
- [ ] **Abordagem Simples (Recomendada):**
  - [ ] Navegação reativa pura (fuzzy)
  - [ ] Sem path planning explícito
  - [ ] Evitação local de obstáculos
- [ ] **Abordagem Avançada (Opcional):**
  - [ ] A* ou RRT para path planning
  - [ ] Planejar trajeto para cubo/caixa
  - [ ] Implementar em: `src/navigation/path_planner.py`

**Escolha:** Documentar em DECISIONS.md

#### 4.3 Localização Relativa
- [ ] Odometria baseada em comandos de velocidade
- [ ] Estimativa de posição relativa (sem GPS!)
- [ ] Reset ao depositar cubo
- [ ] Implementar em: `src/navigation/odometry.py`

**Deliverable:** Sistema de navegação funcional

**Referências Fase 4:**
- Thrun et al. (2005): Probabilistic Robotics
- Siegwart et al. (2011): Mobile Robots

---

### Fase 5: Manipulação e Grasping
**Prazo:** 4 dias
**Objetivo:** Sequências confiáveis de pegar e soltar cubos

#### 5.1 Sequência de Grasping
- [ ] Definir posições do braço para pegar cubo:
  - [ ] Reset → posição preparatória
  - [ ] Preparatória → posição de pegada (FRONT_FLOOR)
  - [ ] Abrir garra
  - [ ] Descer braço até cubo
  - [ ] Fechar garra
  - [ ] Levantar braço
- [ ] Timing entre comandos (espera estabilização)
- [ ] Verificação de sucesso (sensor de força ou timeout)
- [ ] Implementar em: `src/manipulation/grasping.py`

#### 5.2 Sequência de Deposição
- [ ] Posicionar robô perto da caixa
- [ ] Mover braço para posição sobre caixa
- [ ] Abrir garra
- [ ] Retrair braço
- [ ] Reset para posição inicial
- [ ] Implementar em: `src/manipulation/depositing.py`

#### 5.3 Identificação das Caixas
- [ ] Mapear posições fixas das caixas (verde, azul, vermelha)
- [ ] Navegação para caixa baseada na cor do cubo segurado
- [ ] Hardcode inicial de posições (simplificação)
- [ ] Opcional: Detecção visual das caixas

**Deliverable:** Sequências de manipulação confiáveis (>80% sucesso)

**Referências Fase 5:**
- Craig (2005): Robot kinematics
- Bohg et al. (2014): Grasp synthesis

---

### Fase 6: Integração do Sistema Completo
**Prazo:** 5 dias
**Objetivo:** Loop principal funcionando end-to-end

#### 6.1 Arquitetura do Main Controller
- [ ] Implementar loop principal em: `src/main_controller.py`
```python
while cubos_coletados < 15:
    # 1. Percepção
    obstacles, cubes = perception_system.update()

    # 2. Decisão (State Machine + Fuzzy)
    state, action = controller.decide(obstacles, cubes, robot_state)

    # 3. Atuação
    if state == SEARCHING:
        base.move(vx, vy, omega)
    elif state == GRASPING:
        grasping.execute()
    elif state == DEPOSITING:
        depositing.execute()

    # 4. Update estado
    robot_state.update()

    step()
```

#### 6.2 Fluxo Completo
- [ ] Estado inicial: Busca
- [ ] Detecção de cubo → Aproximação
- [ ] Chegou perto → Pegar
- [ ] Pegou → Navegar para caixa
- [ ] Chegou na caixa → Depositar
- [ ] Depositou → Voltar para busca
- [ ] Repetir até 15 cubos

#### 6.3 Tratamento de Erros
- [ ] Timeout em estados (se travar)
- [ ] Retentar grasp se falhar
- [ ] Evitar ficar preso em cantos
- [ ] Log de eventos: `logs/execution.log`

#### 6.4 Testes de Integração
- [ ] Teste com 3 cubos primeiro
- [ ] Depois 5, 10, 15
- [ ] Diferentes configurações de obstáculos
- [ ] Medir taxa de sucesso

**Deliverable:** Sistema completo funcional

---

### Fase 7: Otimização e Refinamento
**Prazo:** 5 dias
**Objetivo:** Melhorar performance e confiabilidade

#### 7.1 Ajuste de Parâmetros
- [ ] Fuzzy:
  - [ ] Funções de pertinência
  - [ ] Pesos das regras
  - [ ] Thresholds de decisão
- [ ] Percepção:
  - [ ] Thresholds de confiança
  - [ ] Filtros de ruído
- [ ] Navegação:
  - [ ] Velocidades máximas
  - [ ] Distâncias seguras
- [ ] Manipulação:
  - [ ] Timings
  - [ ] Posições do braço

#### 7.2 Métricas de Performance
- [ ] Taxa de sucesso na coleta (%)
- [ ] Tempo médio por cubo (s)
- [ ] Número de colisões
- [ ] Precisão na deposição por cor (%)
- [ ] Documentar em: `docs/performance_metrics.md`

#### 7.3 Debugging
- [ ] Adicionar logs detalhados
- [ ] Visualizações em tempo real:
  - [ ] Mapa LIDAR
  - [ ] Cubos detectados
  - [ ] Estado atual
- [ ] Modo de replay para análise

**Deliverable:** Sistema otimizado com métricas documentadas

---

### Fase 8: Documentação e Apresentação
**Prazo:** 7 dias
**Objetivo:** Material para vídeo de 15 minutos

#### 8.1 Documentação Técnica
- [ ] `README.md` com:
  - [ ] Descrição do projeto
  - [ ] Como executar
  - [ ] Estrutura de código
  - [ ] Dependências
- [ ] `docs/architecture.md`:
  - [ ] Diagramas de arquitetura
  - [ ] Fluxo de dados
  - [ ] Decisões de design
- [ ] `docs/results.md`:
  - [ ] Métricas finais
  - [ ] Análise de resultados

#### 8.2 Material Visual (SEM CÓDIGO!)

**REGRA DE OURO:** Slides = IMAGENS E FIGURAS. Texto excessivo perde pontos!

- [ ] Adaptar template LaTeX: `slides-template/main.tex`
  - [ ] Atualizar título: "YouBot Autônomo - Sistema de Coleta com RNA + Fuzzy"
  - [ ] Autor: Luis Felipe Cordeiro Sena
  - [ ] Estrutura: 7 seções (Intro, Teoria, Arquitetura, Percepção, Controle, Demo, Resultados)
  - [ ] Integrar bibliografia (Top 10 de REFERENCIAS.md)
  - [ ] **Máximo 3-4 bullet points por slide, NUNCA parágrafos**
- [ ] Roteiro de fala: `slides-template/falas.txt`
  - [ ] Ajustar para apresentação individual de 15 min
  - [ ] Sincronizar com estrutura de slides
  - [ ] Foco: Você explica verbalmente, slides só apoiam visualmente
- [ ] Diagramas:
  - [ ] Arquitetura do sistema
  - [ ] Pipeline de percepção
  - [ ] Funções de pertinência fuzzy
  - [ ] Regras fuzzy (visual)
  - [ ] Máquina de estados
  - [ ] Modelo cinemático do YouBot
- [ ] Gráficos:
  - [ ] Curvas de aprendizado (RNA)
  - [ ] Métricas de performance
  - [ ] Comparação de abordagens
- [ ] Vídeos/GIFs:
  - [ ] Robô coletando cubos (diferentes ângulos)
  - [ ] Evitação de obstáculos
  - [ ] Sequência de grasp
  - [ ] Visualização LIDAR em tempo real
  - [ ] Detecção de cubos com bounding boxes
- [ ] Ferramentas:
  - [ ] Draw.io para diagramas
  - [ ] Matplotlib/seaborn para gráficos
  - [ ] OBS Studio para gravação

#### 8.3 Roteiro do Vídeo (15 min)
- [ ] **Intro (1 min):**
  - [ ] Apresentação do problema
  - [ ] Objetivos do projeto
- [ ] **Fundamentação Teórica (3 min):**
  - [ ] Redes Neurais (LIDAR + Câmera)
  - [ ] Lógica Fuzzy (Controle)
  - [ ] Citações: Top 10 referências
- [ ] **Arquitetura do Sistema (2 min):**
  - [ ] Diagrama completo
  - [ ] Módulos e integração
- [ ] **Percepção (2 min):**
  - [ ] Processamento LIDAR
  - [ ] Detecção de cubos
  - [ ] Demonstração visual
- [ ] **Controle Fuzzy (2 min):**
  - [ ] Variáveis e regras
  - [ ] Máquina de estados
  - [ ] Exemplos de decisão
- [ ] **Demonstração (4 min):**
  - [ ] Vídeo do robô em ação
  - [ ] Coleta completa de 15 cubos
  - [ ] Diferentes cenários
- [ ] **Resultados (1 min):**
  - [ ] Métricas de performance
  - [ ] Taxa de sucesso
  - [ ] Gráficos

#### 8.4 Gravação e Edição
- [ ] Gravar áudio (microfone de qualidade)
- [ ] Gravar tela com apresentação
- [ ] Gravar simulação no Webots
- [ ] Editar no DaVinci Resolve / Premiere
- [ ] Adicionar legendas (opcional)
- [ ] Música de fundo discreta (opcional)
- [ ] Exportar em 1080p

#### 8.5 Submissão
- [ ] Upload no Youtube (não listado)
- [ ] Código em .zip
- [ ] Preencher formulário de entrega
- [ ] Verificar prazo: **06/01/2026, 23:59**

**Deliverable:** Vídeo de 15min + código-fonte

**Referências:** Todas as Top 10 de REFERENCIAS.md

---

## 🏗️ Estrutura de Código Final

```
projeto-final-ia/
├── src/
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── lidar_processor.py       # RNA para LIDAR
│   │   ├── cube_detector.py         # CNN para cubos
│   │   └── perception_system.py     # Integração
│   ├── control/
│   │   ├── __init__.py
│   │   ├── fuzzy_controller.py      # Lógica Fuzzy
│   │   ├── fuzzy_rules.txt          # Regras
│   │   ├── state_machine.py         # Estados
│   │   └── robot_controller.py      # Integração
│   ├── navigation/
│   │   ├── __init__.py
│   │   ├── local_map.py             # Mapa local
│   │   ├── odometry.py              # Odometria
│   │   └── path_planner.py          # Opcional
│   ├── manipulation/
│   │   ├── __init__.py
│   │   ├── grasping.py              # Sequência grasp
│   │   └── depositing.py            # Sequência deposição
│   └── main_controller.py           # Loop principal
├── models/
│   ├── lidar_model.pth              # Modelo LIDAR
│   └── cube_detector.pth            # Modelo CNN
├── tests/
│   ├── test_basic_controls.py
│   ├── test_perception.py
│   ├── test_fuzzy.py
│   └── test_integration.py
├── notebooks/
│   ├── 01_sensor_exploration.ipynb
│   ├── 02_lidar_training.ipynb
│   └── 03_cube_detection_training.ipynb
├── docs/
│   ├── arena_map.md
│   ├── fuzzy_membership.md
│   ├── architecture.md
│   ├── performance_metrics.md
│   └── results.md
├── logs/
│   └── execution.log
├── media/
│   ├── diagrams/
│   ├── graphs/
│   └── videos/
├── IA_20252/                        # Código base (existente)
├── CLAUDE.md
├── REFERENCIAS.md
├── TODO.md                          # Este arquivo
├── DECISIONS.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Cronograma Visual

```
Semanas 1-2:  [████████] Fase 1-2: Setup + Percepção (RNA)
Semanas 3:    [████]     Fase 3: Controle Fuzzy
Semana 4:     [███]      Fase 4-5: Navegação + Manipulação
Semana 5:     [████]     Fase 6: Integração
Semana 6:     [███]      Fase 7: Otimização
Semanas 7-8:  [█████]    Fase 8: Documentação + Vídeo
                         [BUFFER: 1 semana antes da entrega]
```

**Total:** ~8 semanas + 1 buffer = 9 semanas até 06/01/2026

---

## ✅ Critérios de Sucesso

### Mínimo Viável (Aprovação)
- [ ] Sistema coleta pelo menos 10/15 cubos
- [ ] Identificação de cores >80% precisa
- [ ] Evitação de obstáculos funcional
- [ ] RNA para LIDAR implementada e funcional
- [ ] Lógica Fuzzy implementada e funcional
- [ ] Vídeo de 15min explicando tudo (SEM CÓDIGO!)

### Excelência (Nota Máxima)
- [ ] Sistema coleta 15/15 cubos consistentemente
- [ ] Identificação de cores >95% precisa
- [ ] Navegação eficiente (tempo otimizado)
- [ ] Zero colisões com obstáculos
- [ ] Apresentação visual impecável
- [ ] Fundamentação teórica sólida
- [ ] Código bem documentado e organizado

---

## 🔧 Dependências Técnicas

### Python Packages
```txt
# requirements.txt
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scikit-fuzzy>=0.4.2
opencv-python>=4.8.0
torch>=2.0.0                # PyTorch para CNNs
torchvision>=0.15.0
pillow>=10.0.0
jupyter>=1.0.0
pytest>=7.4.0
```

### Instalação
```bash
pip install -r requirements.txt
```

---

## 📝 Notas Importantes

1. **Sem GPS:** Toda navegação baseada em sensores (LIDAR + câmera)
2. **Sem modificar supervisor.py:** Sob pena de perda de pontos
3. **Sem mostrar código no vídeo:** Perda de 3-10 pontos
4. **Foco visual:** Figuras, gráficos, vídeos > texto
5. **Prazo fatal:** 06/01/2026, 23:59
6. **Documentar tudo:** DECISIONS.md a cada escolha técnica

---

## 🎓 Referências por Fase

**Setup:** Michel (2004)
**Percepção:** Goodfellow (2016), Qi (2017), Redmon (2016), Liu (2016)
**Controle:** Zadeh (1965), Mamdani (1975), Saffiotti (1997)
**Navegação:** Thrun (2005), Siegwart (2011)
**Manipulação:** Craig (2005), Bohg (2014)
**Integração:** Todas as Top 10

Ver REFERENCIAS.md para lista completa.

---

**Última atualização:** 2025-11-18
**Próxima revisão:** Após conclusão de cada fase
