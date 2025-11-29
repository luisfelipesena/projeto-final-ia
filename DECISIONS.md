# Registro de Decisões Técnicas - YouBot Autônomo

**Projeto:** Sistema Autônomo de Coleta e Organização de Objetos com YouBot
**Aluno:** Luis Felipe Cordeiro Sena
**Disciplina:** MATA64 - Inteligência Artificial - UFBA

---

## Índice

| # | Decisão | Fase | Status |
|---|---------|------|--------|
| 001 | Estrutura do Projeto | Setup | ✅ |
| 002 | Sistema de Rastreamento | Setup | ✅ |
| 003 | Organização de Referências | Setup | ✅ |
| 004 | Planejamento por Fases | Setup | ✅ |
| 005 | Instalação Webots R2023b | Setup | ✅ |
| 006 | Integração Python-Webots | Fase 1 | ✅ |
| 007 | Framework de Testes | Fase 1 | ✅ |
| 008 | Validação de Sensores | Fase 1 | ✅ |
| 009 | Restrição GPS | Fase 1 | ✅ |
| 010 | World File Compatibility | Fase 1 | ✅ |
| 016 | Hybrid CNN+MLP para LIDAR | Fase 2 | ✅ |
| 017 | Lightweight CNN para Câmera | Fase 2 | ✅ |
| 018 | Mamdani Fuzzy 25 Regras | Fase 3 | ✅ |
| 019 | 7-State Machine | Fase 3 | ✅ |
| 020 | Mock Perception System | Fase 3 | ✅ |
| 021 | Controller Integration | Fase 6 | ✅ |

---

## DECISÃO 001: Estrutura do Projeto

**O que:** Criar documentação estruturada (CLAUDE.md, REFERENCIAS.md, TODO.md, DECISIONS.md) antes de implementar.

**Por quê:** Projeto longo (até 06/01/2026) requer rastreabilidade. Professor cobra fundamentação teórica.

**Base teórica:** Martin Fowler - "Documentation should live with the code"; IEEE Std 1016-2009.

**Alternativas:** (1) Só README ❌ insuficiente; (2) Wiki externa ❌ separação; (3) LaTeX único ❌ overhead.

**Impacto:** Clareza nas decisões, facilita apresentação, rastreabilidade completa.

---

## DECISÃO 002: Sistema de Rastreamento

**O que:** Usar DECISIONS.md como registro único de todas as decisões técnicas com 5 campos obrigatórios.

**Por quê:** Evitar retrabalho, garantir consistência, fundamentar apresentação.

**Base teórica:** Architecture Decision Records (ADR) - Michael Nygard.

**Alternativas:** (1) Git commits ❌ difícil navegar; (2) Issues ❌ efêmero.

**Impacto:** Histórico completo de decisões arquiteturais.

---

## DECISÃO 003: Organização de Referências

**O que:** REFERENCIAS.md com 80+ papers organizados por tópico + Top 10 essenciais para apresentação.

**Por quê:** Cumprir requisito de fundamentação científica. Facilitar citações no vídeo.

**Base teórica:** Método científico - citação de fontes primárias.

**Alternativas:** (1) BibTeX ❌ menos acessível; (2) Inline ❌ duplicação.

**Impacto:** Todas decisões têm respaldo científico citável.

---

## DECISÃO 004: Planejamento por Fases

**O que:** 8 fases (Setup→Webots→Percepção→Fuzzy→Navegação→Integração→Polish→Demo).

**Por quê:** Desenvolvimento incremental com deliverables testáveis por fase.

**Base teórica:** Agile/Scrum - sprints com entregas incrementais.

**Alternativas:** (1) Waterfall ❌ risco alto; (2) Kanban puro ❌ sem marcos.

**Impacto:** Progresso mensurável, riscos mitigados cedo.

---

## DECISÃO 005: Instalação Webots R2023b

**O que:** Usar Webots R2023b via DMG oficial no macOS.

**Por quê:** Compatibilidade garantida com Python 3.8+, documentação oficial.

**Base teórica:** Webots User Guide - Installation.

**Alternativas:** (1) Homebrew ❌ versão desatualizada; (2) Build source ❌ complexo.

**Impacto:** Ambiente funcional para desenvolvimento.

---

## DECISÃO 006: Integração Python-Webots

**O que:** Controller em Python puro usando API nativa do Webots (controller.Robot).

**Por quê:** Python é requisito do projeto. API Webots bem documentada.

**Base teórica:** Webots Reference Manual - Python API.

**Alternativas:** (1) C/C++ ❌ maior complexidade; (2) ROS ❌ overhead.

**Impacto:** Desenvolvimento rápido, fácil debugging.

---

## DECISÃO 007: Framework de Testes

**O que:** pytest com fixtures para mock de Robot/sensores. Testes em `tests/`.

**Por quê:** Testes unitários garantem qualidade. Mock permite testar sem Webots rodando.

**Base teórica:** TDD - Kent Beck; pytest documentation.

**Alternativas:** (1) unittest ❌ verbose; (2) Sem testes ❌ alto risco.

**Impacto:** 16 testes passando, cobertura de módulos críticos.

---

## DECISÃO 008: Validação de Sensores

**O que:** Coletar dados reais de LIDAR (667 pontos) e Câmera (512x512 RGB) para validar processamento.

**Por quê:** Garantir que pipeline funciona com dados reais antes de treinar modelos.

**Base teórica:** Webots LIDAR/Camera device specifications.

**Alternativas:** (1) Dados sintéticos ❌ não representativo.

**Impacto:** Pipeline validado com dados reais do simulador.

---

## DECISÃO 009: Restrição GPS

**O que:** GPS PROIBIDO na demo final. Navegação apenas com odometria + LIDAR + câmera.

**Por quê:** **REQUISITO OBRIGATÓRIO** do Final Project.pdf - perda de pontos se usar GPS.

**Base teórica:** Odometry - Thrun et al. (2005) Probabilistic Robotics.

**Alternativas:** Nenhuma - requisito fixo.

**Impacto:** Sistema usa odometria incremental + mapa local.

---

## DECISÃO 010: World File Compatibility

**O que:** World file R2025a funciona em Webots R2023b com warning de versão (ignorável).

**Por quê:** Evitar reescrever world file. Funcionalidade preservada.

**Base teórica:** Teste empírico - backward compatibility do Webots.

**Alternativas:** (1) Converter para R2023b ❌ risco de perder configurações.

**Impacto:** Usar world original sem modificações.

---

## DECISÃO 016: Hybrid CNN+MLP para LIDAR

**O que:** HybridLIDARNet com branch CNN 1D (667→64 features) + branch estatístico (6 features) → MLP classificador → 9 setores de ocupação.

**Por quê:** CNN captura padrões espaciais, features estatísticas complementam. Output em setores simplifica controle fuzzy.

**Base teórica:**
- Qi et al. (2017) PointNet - processamento de point clouds
- Goodfellow et al. (2016) Deep Learning - CNN architectures

**Alternativas:** (1) MLP puro ❌ perde padrões espaciais; (2) PointNet completo ❌ overkill para 1D.

**Impacto:** ~50K parâmetros, <100ms inferência, integra com fuzzy controller.

**Implementação:** `src/perception/models/lidar_net.py`

---

## DECISÃO 017: Lightweight CNN para Câmera

**O que:** CNN com 3 blocos Conv2D+BN+ReLU+Pool → GlobalAvgPool → FC(128→64→3) para classificar cores (verde/azul/vermelho). Fallback HSV implementado.

**Por quê:** Classificação simples de 3 cores não requer modelo pesado. HSV fallback garante funcionamento sem treinamento.

**Base teórica:**
- Goodfellow et al. (2016) Deep Learning - CNNs para visão
- OpenCV HSV color segmentation

**Alternativas:** (1) ResNet/YOLO ❌ overkill; (2) Só HSV ❌ menos robusto a variações.

**Impacto:** ~250K parâmetros, >10 FPS em CPU, fallback funcional.

**Implementação:** `src/perception/models/camera_net.py`

---

## DECISÃO 018: Mamdani Fuzzy 25 Regras

**O que:** Controlador fuzzy Mamdani com:
- Entradas: distance_to_obstacle, angle_to_obstacle, distance_to_cube, angle_to_cube
- Saídas: linear_velocity, angular_velocity, action
- 25 regras (15 safety weight=10, 5 task weight=5, 5 exploration weight=1)
- Defuzzificação: centróide

**Por quê:** **REQUISITO:** mínimo 20 regras fuzzy. Mamdani é padrão para controle robótico.

**Base teórica:**
- Zadeh (1965) Fuzzy Sets
- Mamdani & Assilian (1975) Fuzzy Logic Controller
- Saffiotti (1997) Fuzzy Navigation

**Alternativas:** (1) Takagi-Sugeno ❌ menos interpretável; (2) PID ❌ não é fuzzy.

**Impacto:** 25 regras implementadas (excede requisito), ~30ms/inferência.

**Implementação:** `src/control/fuzzy_controller.py`, `src/control/fuzzy_rules.py`

---

## DECISÃO 019: 7-State Machine

**O que:** FSM com estados: SEARCHING → APPROACHING → GRASPING → NAVIGATING_TO_BOX → DEPOSITING → AVOIDING → RECOVERY.

**Por quê:** Organiza comportamento complexo em estados discretos. AVOIDING tem prioridade máxima (safety).

**Base teórica:**
- Brooks (1986) Subsumption Architecture
- Finite State Machines for robotics

**Alternativas:** (1) Behavior trees ❌ complexidade desnecessária; (2) Reativo puro ❌ difícil coordenar sequências.

**Impacto:** Fluxo claro de coleta, tratamento robusto de obstáculos.

**Implementação:** `src/control/state_machine.py`

---

## DECISÃO 020: Mock Perception System

**O que:** MockPerceptionSystem que simula outputs de percepção com 10 cenários pré-definidos para testar fuzzy controller sem Webots.

**Por quê:** Permite desenvolvimento paralelo de controle e percepção. Testes rápidos e reproduzíveis.

**Base teórica:** Martin Fowler - Mocks Aren't Stubs; Contract-based testing.

**Alternativas:** (1) Aguardar percepção ❌ bloqueia; (2) Dados gravados ❌ menos flexível.

**Impacto:** Fuzzy controller testado independentemente. Migração fácil para percepção real.

**Implementação:** `tests/control/fixtures/mock_perception.py`

---

## DECISÃO 021: Controller Integration

**O que:** `youbot.py` importa e executa `MainController` de `src/` via sys.path manipulation. Fallback para teste de sensores se import falhar.

**Por quê:** **CRITICAL FIX** - `run()` tinha NotImplementedError, todo código de src/ era dead code.

**Base teórica:**
- Thrun et al. (2005) - sense-plan-act architecture
- Brooks (1986) - subsumption layers

**Alternativas:** (1) Herança ❌ acoplamento forte; (2) Copiar código ❌ duplicação.

**Impacto:** Sistema integrado end-to-end. Robô funciona no Webots.

**Implementação:** `IA_20252/controllers/youbot/youbot.py`

---

## Resumo de Compliance

| Requisito | Status |
|-----------|--------|
| RNA (MLP/CNN) para detecção | ✅ HybridLIDARNet + LightweightCNN |
| Lógica Fuzzy ≥20 regras | ✅ 25 regras Mamdani |
| 15 cubos coloridos | ✅ Supervisor spawna 15 |
| GPS proibido na demo | ✅ Odometria only |
| supervisor.py inalterado | ✅ ZERO modificações |
| Vídeo 15min sem código | ⏳ Template pronto |
