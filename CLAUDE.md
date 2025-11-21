# MATA64 - Projeto Final de IA: YouBot Autônomo

**Aluno:** Luis Felipe Cordeiro Sena
**Professor:** Luciano Oliveira (lrebouca@ufba.br)
**Data de Entrega:** 06/01/2026, 23:59
**Semestre:** 2025.2

## Objetivo

Desenvolver um sistema autônomo para robô YouBot no simulador Webots que executa tarefa de coleta e organização de cubos coloridos em arena com obstáculos.

## Especificações do Sistema

### Tarefa Principal
- Coletar **15 cubos coloridos** (verde, azul, vermelho) distribuídos aleatoriamente na arena
- Para cada cubo: pegar com garra → identificar cor → depositar na caixa correspondente
- Navegar evitando obstáculos fixos (caixotes de madeira)

### Restrições Técnicas
- **Sensores permitidos:** LIDAR (detecção de obstáculos/mapeamento) + Câmera RGB (identificação de cores)
- **GPS:** Pode usar para treinar modelos, mas PROIBIDO na demonstração/apresentação final
- **Critério:** Sistema final deve funcionar com GPS desabilitado
- **Podem adicionar:** sensores complementares se necessário

### Requisitos Obrigatórios de IA

#### 1. Redes Neurais Artificiais (RNA)
- **Tipo:** MLP ou CNN (podem usar modelos pré-treinados)
- **Função:** Detecção de obstáculos e mapeamento do ambiente
- **Alternativa:** Pode usar arquiteturas/redes já existentes para detecção/classificação

#### 2. Lógica Fuzzy
- **Função:** Controle das ações do robô (navegação, decisões de movimento)

## Estrutura do Projeto

```
IA_20252/
├── controllers/
│   ├── youbot/
│   │   ├── youbot.py      # Controle principal (base, arm, gripper)
│   │   ├── base.py         # Controle da base móvel
│   │   ├── arm.py          # Controle do braço robótico
│   │   └── gripper.py      # Controle da garra
│   └── supervisor/
│       └── supervisor.py   # Spawn aleatório de cubos (NÃO MODIFICAR)
├── libraries/              # Versão C dos controles (alternativa)
├── textures/
└── worlds/
    └── IA_20252.wbt       # Arena de simulação
```

## Ambiente de Simulação

- **Arena:** Grid com paredes delimitando área de operação
- **Obstáculos:** Caixotes de madeira (WoodenBox) fixos
- **Cubos:** Spawned aleatoriamente pelo supervisor a cada execução (15 unidades)
- **Caixas de depósito:** Verde, azul e vermelha (PlasticFruitBox)

### Coordenadas de Spawn
- X: [-3, 1.75]
- Y: [-1, 1]
- Z: size/2 (nível do chão)

## Código Base

### YouBotController (youbot.py)
```python
- Robot instance com time_step
- Base, Arm, Gripper modules initialized
- Camera e LIDAR já habilitados
- Método run() para implementar
```

### Supervisor (supervisor.py)
- **NÃO MODIFICAR** sob pena de perda de pontos
- Deleta cubos anteriores e spawna 15 novos aleatoriamente
- Evita colisões entre cubos e obstáculos existentes
- Usa recognitionColors para identificação

## Arquitetura Proposta

### Pipeline de Processamento
1. **Percepção:** LIDAR + Câmera RGB → dados brutos
2. **Detecção:** RNA processa LIDAR para mapeamento/obstáculos
3. **Identificação:** Câmera RGB identifica cor dos cubos
4. **Decisão:** Lógica Fuzzy determina ação (aproximar, desviar, pegar, depositar)
5. **Atuação:** Base (movimento) + Arm (posicionamento) + Gripper (pegada)

### Componentes IA a Implementar

#### RNA para Navegação
- **Input:** Dados do LIDAR (distâncias)
- **Output:** Mapa de obstáculos ou decisões de navegação
- **Opções:** CNN para processar varredura LIDAR ou MLP para decisões de movimento

#### RNA para Visão
- **Input:** Imagem RGB da câmera
- **Output:** Classificação de cor (verde/azul/vermelho) e localização de cubos
- **Opções:** CNN pré-treinada (transfer learning) ou modelo custom

#### Lógica Fuzzy para Controle
- **Inputs:** Distância a obstáculos, distância a cubos, estado do robô
- **Outputs:** Velocidade linear, velocidade angular, ações do braço/garra
- **Regras:** Definir comportamentos (aproximação cautelosa, evasão de obstáculos, etc)

## Regras de Entrega

### Vídeo de Apresentação (15 min máx)
- Explicar desenvolvimento conceitual do projeto
- Demonstrar robô realizando tarefa na arena fornecida
- **PROIBIDO:** Mostrar código-fonte (desconto de 3-10 pontos)
- **Foco:** Imagens, processos, diagramas - MÍNIMO texto
- **Template:** `slides-template/main.tex` (LaTeX Beamer) já configurado
- **Submissão:** Link do Youtube + código desenvolvido

### Permitido Usar
- Modelos pré-treinados
- Trechos de código prontos
- Bibliotecas de terceiros
- Arquiteturas de RNA existentes
- **Condição:** Explicar tudo no vídeo

## Next Steps

1. **Estudo do ambiente:** Executar simulação base, entender sensores
2. **Implementação RNA:** Desenvolver/integrar modelo para processamento LIDAR/câmera
3. **Implementação Fuzzy:** Criar sistema de regras para controle
4. **Integração:** Conectar percepção → decisão → atuação
5. **Testes:** Validar coleta dos 15 cubos e deposição correta
6. **Otimização:** Refinar navegação e eficiência
7. **Documentação:** Preparar material visual para apresentação

## SpecKit Workflow (OBRIGATÓRIO)

**Workflow para cada fase:**
1. `/speckit.specify` → Criar spec para nova feature
2. `/speckit.clarify` → Resolver ambiguidades
3. `/speckit.plan` → Gerar plano detalhado
4. `/speckit.tasks` → Quebrar em tasks
5. `/speckit.implement` → Executar
6. `/speckit.analyze` → Validar consistência (opcional)

**Branch strategy:** `00X-feature-name` por fase
**Aprendizado:** Ler DECISIONS.md antes de cada nova decisão técnica

---

## 📋 Metodologia de Desenvolvimento

### Princípios Fundamentais

**1. Decisões Baseadas em Teoria**
- TODAS as escolhas técnicas devem ter fundamentação científica
- Antes de implementar, consultar REFERENCIAS.md
- Documentar decisão em DECISIONS.md ANTES de implementar
- Citar papers relevantes na justificativa

**2. Planejamento Incremental**
- Desenvolvimento dividido em 8 fases (ver TODO.md)
- Cada fase tem deliverable testável
- Não avançar sem concluir fase anterior
- Buffer de 1 semana antes da entrega

**3. Rastreabilidade Total**
- Toda mudança documentada em DECISIONS.md
- Git commits descritivos por fase
- Logs de execução em `logs/`
- Métricas de performance registradas

**4. Qualidade Senior**
- Código limpo e bem estruturado
- Testes para cada módulo crítico
- Documentação inline mínima (foco em DECISIONS.md)
- Performance otimizada

### Documentos Principais

**CLAUDE.md** (este arquivo)
- Contexto geral do projeto
- Especificações e requisitos
- Diretrizes de desenvolvimento

**REFERENCIAS.md**
- 80+ referências científicas organizadas
- Top 10 essenciais para apresentação
- Base teórica para todas as decisões
- Estratégia de citação no vídeo

**TODO.md**
- Plano detalhado em 8 fases
- Checklist de tarefas por fase
- Cronograma até 06/01/2026
- Critérios de sucesso

**DECISIONS.md**
- Registro de TODAS as decisões técnicas
- Formato: O que, Por quê, Base teórica, Alternativas, Impacto
- Atualizar ANTES de cada implementação
- Template padronizado

### Workflow de Desenvolvimento

**Para cada nova funcionalidade:**

1. **Planejar**
   - Consultar TODO.md para contexto da fase
   - Identificar decisões técnicas necessárias
   - Pesquisar em REFERENCIAS.md papers relevantes

2. **Decidir**
   - Avaliar alternativas (mín. 2)
   - Escolher baseado em teoria + requisitos
   - Documentar em DECISIONS.md usando template
   - Justificar com citações científicas

3. **Implementar**
   - Seguir arquitetura definida em TODO.md
   - Código em `src/` organizado por módulo
   - Testes em `tests/`
   - Commits descritivos

4. **Validar**
   - Testes unitários passando
   - Métricas de performance aceitáveis
   - Documentar resultados em DECISIONS.md
   - Atualizar TODO.md (marcar como concluído)

5. **Integrar**
   - Conectar com módulos existentes
   - Testes de integração
   - Update de documentação se necessário

### Regras de Ouro

✅ **SEMPRE:**
- Documentar decisões ANTES de implementar
- Citar papers ao justificar escolhas
- Testar antes de marcar como concluído
- Fazer backup (git push) ao final do dia
- Atualizar TODO.md com progresso

❌ **NUNCA:**
- Modificar supervisor.py (perda de pontos!)
- Mostrar código-fonte no vídeo (perda de 3-10 pontos!)
- Implementar sem fundamentação teórica
- Avançar com testes falhando
- Deixar documentação para depois

### Estrutura de Código Esperada

```
projeto-final-ia/
├── src/
│   ├── perception/           # RNA para LIDAR e câmera
│   │   ├── lidar_processor.py
│   │   ├── cube_detector.py
│   │   └── perception_system.py
│   ├── control/              # Lógica Fuzzy e estados
│   │   ├── fuzzy_controller.py
│   │   ├── state_machine.py
│   │   └── robot_controller.py
│   ├── navigation/           # Mapeamento e path planning
│   │   ├── local_map.py
│   │   └── odometry.py
│   ├── manipulation/         # Grasping e deposição
│   │   ├── grasping.py
│   │   └── depositing.py
│   └── main_controller.py    # Loop principal
├── models/                   # Modelos treinados (.pth)
├── tests/                    # Testes unitários
├── notebooks/                # Jupyter para exploração
├── docs/                     # Diagramas e análises
├── logs/                     # Logs de execução
├── media/                    # Material para apresentação
└── IA_20252/                 # Código base (não modificar supervisor!)
```

### Critérios de Excelência

**Implementação:**
- [ ] Sistema coleta 15/15 cubos consistentemente
- [ ] Identificação de cores >95% precisa
- [ ] Zero colisões com obstáculos
- [ ] Tempo otimizado (<5 min total)

**Código:**
- [ ] Arquitetura modular e bem organizada
- [ ] Testes com >80% cobertura
- [ ] Documentação clara em DECISIONS.md
- [ ] Código limpo (PEP8, type hints)

**Fundamentação:**
- [ ] Todas decisões justificadas cientificamente
- [ ] Top 10 papers citados na apresentação
- [ ] Trade-offs documentados
- [ ] Alternativas comparadas

**Apresentação:**
- [ ] Vídeo de 15 min sem código-fonte
- [ ] Figuras, gráficos e vídeos de qualidade
- [ ] Citações corretas (formato ABNT)
- [ ] Demonstração completa funcionando

---

## 🔬 Base Científica (Quick Reference)

**Top 10 Referências Essenciais:**

1. **Goodfellow et al. (2016)** - Deep Learning fundamentals
2. **Zadeh (1965)** - Fuzzy Sets theory
3. **Mamdani & Assilian (1975)** - Fuzzy Controller
4. **Qi et al. (2017)** - PointNet (LIDAR processing)
5. **Redmon et al. (2016)** - YOLO (object detection)
6. **Bischoff et al. (2011)** - YouBot specifications
7. **Thrun et al. (2005)** - Probabilistic Robotics
8. **Taheri et al. (2015)** - Mecanum kinematics
9. **Saffiotti (1997)** - Fuzzy navigation
10. **Craig (2005)** - Robot kinematics

Ver REFERENCIAS.md para lista completa (80+ papers organizados por tópico).

---

## 📅 Timeline e Checkpoints

**Fase 0 - Setup:** ✅ CONCLUÍDO (2025-11-18)
- [x] CLAUDE.md criado
- [x] REFERENCIAS.md compilado
- [x] TODO.md planejado
- [x] DECISIONS.md inicializado

**Próximas Fases:** Ver TODO.md

**Prazo Final:** 06/01/2026, 23:59 ⚠️

---

## ⚙️ Dependências e Setup

### Requisitos do Sistema
- Webots R2023a ou superior
- Python 3.8+
- CUDA (opcional, para treinamento de CNNs)

### Bibliotecas Python
```bash
pip install -r requirements.txt
```

Principais:
- `numpy`, `scipy`, `matplotlib`
- `torch`, `torchvision` (PyTorch)
- `scikit-fuzzy` (Lógica Fuzzy)
- `opencv-python` (Visão)
- `pytest` (Testes)

Ver `requirements.txt` para lista completa.

---

## 🎯 Próximos Passos Imediatos

1. **Setup do Webots** (Fase 1)
   - Instalar/verificar Webots
   - Testar simulação IA_20252.wbt
   - Explorar sensores (LIDAR, câmera)

2. **Familiarização** (Fase 1)
   - Testar controles básicos (base, arm, gripper)
   - Coletar dados de sensores
   - Criar notebook de exploração

3. **Decisão Arquitetural** (Fase 2)
   - Escolher abordagem para LIDAR (MLP vs PointNet)
   - Escolher modelo para detecção (YOLO vs SSD vs custom)
   - Documentar em DECISIONS.md

Ver TODO.md para plano completo detalhado.

---

**Última Atualização:** 2025-11-18
**Status:** Fase 0 (Setup) concluída, pronto para Fase 1

## Active Technologies
- Python 3.8+ (requirement for Webots R2023b controller compatibility) + Webots R2023b simulator, pytest (testing), numpy/scipy (sensor data processing) (001-webots-setup)
- File-based (world files .wbt, controller scripts, test logs) (001-webots-setup)
- Python 3.14.0 (validated in Phase 1.1) (002-sensor-exploration)
- File-based (test logs, sensor data CSVs, example images, Jupyter notebooks) (002-sensor-exploration)

## Recent Changes
- 001-webots-setup: Added Python 3.8+ (requirement for Webots R2023b controller compatibility) + Webots R2023b simulator, pytest (testing), numpy/scipy (sensor data processing)
