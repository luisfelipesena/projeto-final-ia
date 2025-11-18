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

#### 1.1 Setup do Webots
- [ ] Instalar/atualizar Webots (manual - seguir quickstart.md)
- [x] Verificar versão do Python (compatibilidade com Webots) - Script criado em tests/
- [ ] Testar abertura do mundo IA_20252.wbt (manual - após instalação Webots)
- [ ] Verificar spawn de cubos pelo supervisor (manual - após instalação Webots)
- [x] Documentar setup em DECISIONS.md (DECISÃO 005-008 adicionadas)
- [x] Criar documentação de setup (specs/001-webots-setup/* completo)
- [x] Criar testes de validação automatizados (tests/test_webots_setup.py)

**Deliverable:** Simulação rodando sem erros

#### 1.2 Exploração dos Controles Base
- [ ] Testar movimentos da base (forward, backward, strafe, rotate)
- [ ] Testar comandos do braço (set_height, set_orientation)
- [ ] Testar garra (grip, release)
- [ ] Documentar limites de movimento
- [ ] Criar script de teste básico: `tests/test_basic_controls.py`

**Deliverable:** Script de teste validando todos os controles

#### 1.3 Análise dos Sensores
- [ ] **LIDAR:**
  - [ ] Ler dados brutos (range_image)
  - [ ] Entender formato (número de pontos, range, FOV)
  - [ ] Visualizar varredura (matplotlib/plot)
  - [ ] Identificar obstáculos na visualização
- [ ] **Câmera RGB:**
  - [ ] Capturar frames
  - [ ] Verificar resolução e FPS
  - [ ] Testar detecção de cores (threshold simples)
  - [ ] Salvar imagens de exemplo
- [ ] Criar notebook: `notebooks/01_sensor_exploration.ipynb`

**Deliverable:** Notebook com visualizações e análises dos sensores

#### 1.4 Mapeamento da Arena
- [ ] Medir dimensões da arena manualmente
- [ ] Identificar posições aproximadas das caixas de depósito
- [ ] Mapear distribuição típica de obstáculos
- [ ] Documentar coordenadas em `docs/arena_map.md`

**Deliverable:** Mapa esquemático da arena

**Referências Fase 1:**
- Bischoff et al. (2011): YouBot specifications
- Michel (2004): Webots documentation

---

### Fase 2: Percepção com Redes Neurais
**Prazo:** 10 dias
**Objetivo:** Implementar detecção de obstáculos (LIDAR) e classificação de cores (RGB)

#### 2.1 Processamento LIDAR com RNA

**2.1.1 Abordagem Simplificada (Recomendada)**
- [ ] Converter LIDAR 2D para representação processável
  - [ ] Grid ocupancy map (2D array)
  - [ ] Polar representation (distância, ângulo)
- [ ] Arquitetura MLP:
  - [ ] Input: LIDAR ranges (normalized)
  - [ ] Hidden layers: 2-3 camadas
  - [ ] Output: Classificação de setores (livre/ocupado)
- [ ] Treinar com dados sintéticos:
  - [ ] Gerar cenários variados no Webots
  - [ ] Coletar 1000+ exemplos de varreduras LIDAR
  - [ ] Labels: obstáculo detectado em cada setor
- [ ] Validar precisão (>90% em test set)
- [ ] Implementar em: `src/perception/lidar_processor.py`

**2.1.2 Abordagem Avançada (Opcional - se tempo permitir)**
- [ ] Adaptar PointNet para LIDAR 2D
  - [ ] Converter ranges para point cloud
  - [ ] Simplificar arquitetura (menos layers)
- [ ] Usar modelo pré-treinado e fine-tuning
- [ ] Implementar em: `src/perception/lidar_pointnet.py`

**Escolha:** Documentar abordagem escolhida em DECISIONS.md

#### 2.1.3 Detecção de Obstáculos
- [ ] Processar output da RNA para identificar obstáculos
- [ ] Calcular distância e ângulo de cada obstáculo
- [ ] Implementar filtro de ruído (média móvel)
- [ ] Criar visualização em tempo real

**Deliverable:** Módulo LIDAR funcionando com >90% precisão

**2.2 Detecção de Cubos com CNN**

**2.2.1 Escolha de Arquitetura**
Opções (escolher UMA e documentar em DECISIONS.md):
- [ ] **Opção A:** YOLO pré-treinado + transfer learning
  - Rápido, tempo real
  - Bom para detecção + classificação simultânea
- [ ] **Opção B:** SSD (melhor para objetos pequenos)
- [ ] **Opção C:** CNN customizada simples
  - Sliding window + classificação de cores
  - Menos overhead, mais controle

**2.2.2 Implementação**
- [ ] Preparar dataset:
  - [ ] Coletar 500+ imagens da câmera no Webots
  - [ ] Anotar bounding boxes de cubos
  - [ ] Labels: cor (verde/azul/vermelho)
  - [ ] Split: 70% treino, 15% validação, 15% teste
- [ ] Treinar modelo:
  - [ ] Se YOLO: fine-tune últimas camadas
  - [ ] Se custom: treinar do zero com data augmentation
  - [ ] Early stopping com validation loss
  - [ ] Salvar melhor modelo em `models/cube_detector.pth`
- [ ] Validar:
  - [ ] Precisão por cor (>95%)
  - [ ] FPS (target: >10 fps)
  - [ ] Falsos positivos/negativos
- [ ] Implementar em: `src/perception/cube_detector.py`

**2.2.3 Classificação de Cores (Alternativa Simples)**
Se detecção for muito complexa:
- [ ] Usar threshold RGB simples
- [ ] Definir ranges para verde, azul, vermelho
- [ ] Aplicar em região detectada
- [ ] Validar com imagens de teste

**Deliverable:** Detector de cubos com >95% precisão em cores

**2.3 Integração Percepção**
- [ ] Classe `PerceptionSystem` que unifica:
  - [ ] LIDAR → obstáculos
  - [ ] Câmera → cubos coloridos
- [ ] Output estruturado:
  ```python
  {
    'obstacles': [(dist, angle), ...],
    'cubes': [{'color': 'green', 'position': (x,y), 'distance': d}, ...]
  }
  ```
- [ ] Implementar em: `src/perception/perception_system.py`
- [ ] Testes unitários: `tests/test_perception.py`

**Deliverable:** Sistema de percepção integrado e testado

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
