# Registro de Decisões Técnicas - YouBot Autônomo

**Projeto:** Sistema Autônomo de Coleta e Organização de Objetos com YouBot
**Aluno:** Luis Felipe Cordeiro Sena
**Disciplina:** MATA64 - Inteligência Artificial - UFBA

---

## Propósito deste Documento

Este arquivo rastreia **todas as decisões técnicas e teóricas** tomadas durante o desenvolvimento do projeto. Para cada decisão, documentamos:

1. **O que foi decidido**
2. **Por que foi decidido** (justificativa)
3. **Base teórica** (referências científicas)
4. **Alternativas consideradas**
5. **Impacto esperado**

**Regra:** Atualizar este arquivo **antes** de implementar qualquer mudança significativa.

---

## Índice de Decisões

1. [Estrutura do Projeto e Documentação](#decisão-001-estrutura-do-projeto-e-documentação)
2. [Sistema de Rastreamento de Decisões](#decisão-002-sistema-de-rastreamento-de-decisões)
3. [Organização de Referências Científicas](#decisão-003-organização-de-referências-científicas)
4. [Planejamento por Fases](#decisão-004-planejamento-por-fases)
5. [Método de Instalação do Webots R2023b](#decisão-005-método-de-instalação-do-webots-r2023b)
6. [Estratégia de Integração Python-Webots](#decisão-006-estratégia-de-integração-python-webots)
7. [Framework de Testes Automatizados](#decisão-007-framework-de-testes-automatizados)
8. [Abordagem de Validação de Sensores](#decisão-008-abordagem-de-validação-de-sensores)

---

## DECISÃO 001: Estrutura do Projeto e Documentação

**Data:** 2025-11-18
**Fase:** Setup (Fase 0)
**Status:** ✅ Implementado

### O que foi decidido

Criar estrutura de documentação completa antes de iniciar implementação:
- `CLAUDE.md` - Contexto e diretrizes do projeto
- `REFERENCIAS.md` - Base científica unificada
- `TODO.md` - Planejamento detalhado passo a passo
- `DECISIONS.md` - Este arquivo de rastreamento
- `.gitignore` - Proteção de credenciais

### Por que foi decidido

**Motivação:**
- Projeto de longo prazo (até 06/01/2026) requer organização rigorosa
- Professor cobra fundamentação teórica sólida
- Apresentação visual exige material bem estruturado
- Necessidade de rastrear decisões para evitar retrabalho

**Justificativa Técnica:**
1. **Metodologia ágil:** Documentação viva que evolui com o projeto
2. **Princípio DRY:** Evitar duplicação de informações
3. **Manutenibilidade:** Facilitar retomada após pausas
4. **Transparência:** Decisões justificadas e rastreáveis

### Base teórica

- **Software Engineering Best Practices:**
  - Martin Fowler: "Documentation should live with the code"
  - IEEE Std 1016-2009: Software Design Descriptions

### Alternativas consideradas

1. **Documentação mínima:** Apenas README
   - ❌ Insuficiente para projeto acadêmico rigoroso
2. **Wiki externa:** Notion, Confluence
   - ❌ Separação entre código e documentação
3. **LaTeX completo:** Documento formal único
   - ❌ Overhead desnecessário, dificulta iterações rápidas

### Impacto esperado

- ✅ Maior clareza nas decisões técnicas
- ✅ Facilita preparação da apresentação final
- ✅ Documentação serve como base para relatório/vídeo
- ✅ Rastreabilidade de mudanças ao longo do tempo

---

## DECISÃO 002: Sistema de Rastreamento de Decisões

**Data:** 2025-11-18
**Fase:** Setup (Fase 0)
**Status:** ✅ Implementado

### O que foi decidido

Criar `DECISIONS.md` como registro vivo de todas as decisões técnicas, com template padronizado:
- Data e fase
- Decisão, justificativa, base teórica
- Alternativas consideradas
- Impacto esperado

### Por que foi decidido

**Motivação:**
- Projetos de IA envolvem muitas escolhas de arquitetura (MLP vs CNN, Mamdani vs Sugeno, etc.)
- Necessidade de justificar escolhas com base científica na apresentação
- Evitar decisões "porque sim" - tudo deve ter fundamentação
- Facilitar retrospectiva e aprendizado

**Justificativa Técnica:**
1. **Design Rationale:** Rastrear "por quê" além de "o quê"
2. **Knowledge Management:** Decisões como artefatos de conhecimento
3. **Accountability:** Responsabilidade sobre escolhas técnicas

### Base teórica

- **Decision Documentation Patterns:**
  - Architecture Decision Records (ADR) - Michael Nygard
  - Design rationale capture methods

- **Relevant to AI/ML Projects:**
  - Model selection justification (Goodfellow et al., 2016, Cap. 11)
  - Hyperparameter choices documentation
  - Architecture search decision trees

### Alternativas consideradas

1. **Git commits apenas:**
   - ❌ Falta contexto de "por quê"
   - ❌ Difícil visualizar decisões de alto nível
2. **Comments no código:**
   - ❌ Fragmentado, difícil visão geral
   - ❌ Não permite comparação de alternativas
3. **Issue tracker (GitHub Issues):**
   - ❌ Overhead para projeto solo
   - ❌ Separação entre código e decisões

### Impacto esperado

- ✅ Apresentação no vídeo: "Escolhemos X baseado em Y (Autor, Ano)"
- ✅ Facilita debugging: entender por que algo foi feito
- ✅ Aprendizado: reflexão sobre trade-offs
- ✅ Reprodutibilidade: outros podem entender escolhas

---

## DECISÃO 003: Organização de Referências Científicas

**Data:** 2025-11-18
**Fase:** Setup (Fase 0)
**Status:** ✅ Implementado

### O que foi decidido

Unificar `REFERENCIAS.md` e `REFERENCIAS_CITACAO.md` em arquivo único com:
- Top 10 essenciais (para apresentação)
- Referências organizadas por tópico (12 seções)
- Aplicação prática de cada paper
- Estratégia de citação para vídeo
- BibTeX para possível LaTeX

### Por que foi decidido

**Motivação:**
- Projeto exige fundamentação teórica rigorosa
- Apresentação deve citar papers (proibido mostrar código)
- Evitar redundância entre arquivos de referências
- Facilitar consulta rápida durante implementação

**Justificativa Técnica:**
1. **Princípio DRY:** Single Source of Truth para referências
2. **Usabilidade:** Top 10 como quick reference
3. **Rastreabilidade:** Cada módulo ligado a papers específicos
4. **Academic Rigor:** Citações ABNT + BibTeX

### Base teórica

**Papers incluídos (Top 10):**
1. Goodfellow et al. (2016) - Deep Learning fundamentals
2. Zadeh (1965) - Fuzzy Sets
3. Mamdani & Assilian (1975) - Fuzzy Controller
4. Qi et al. (2017) - PointNet (LIDAR)
5. Redmon et al. (2016) - YOLO (detection)
6. Bischoff et al. (2011) - YouBot specs
7. Thrun et al. (2005) - Probabilistic Robotics
8. Taheri et al. (2015) - Mecanum kinematics
9. Saffiotti (1997) - Fuzzy navigation
10. Craig (2005) - Robot kinematics

**Total:** 80+ referências peer-reviewed

### Alternativas consideradas

1. **Referências separadas por módulo:**
   - ❌ Dificulta visão geral
   - ❌ Duplicação de papers comuns
2. **Apenas Top 5:**
   - ❌ Insuficiente para embasar todas as escolhas
3. **Zotero/Mendeley external:**
   - ❌ Separação entre documentação e refs

### Impacto esperado

- ✅ Apresentação bem fundamentada (cada slide com citações)
- ✅ Decisões técnicas justificadas cientificamente
- ✅ Facilita redação de possível artigo futuro
- ✅ Demonstra rigor acadêmico ao professor

---

## DECISÃO 004: Planejamento por Fases

**Data:** 2025-11-18
**Fase:** Setup (Fase 0)
**Status:** ✅ Planejado

### O que foi decidido

Dividir projeto em 8 fases sequenciais com critérios claros:
1. **Fase 0:** Setup e documentação (3 dias) ✅
2. **Fase 1:** Ambiente e exploração (3 dias)
3. **Fase 2:** Percepção com RNA (10 dias)
4. **Fase 3:** Controle Fuzzy (7 dias)
5. **Fase 4:** Navegação (5 dias)
6. **Fase 5:** Manipulação (4 dias)
7. **Fase 6:** Integração (5 dias)
8. **Fase 7:** Otimização (5 dias)
9. **Fase 8:** Documentação e vídeo (7 dias)

**Total:** ~8 semanas + 1 buffer = até 06/01/2026

### Por que foi decidido

**Motivação:**
- Projeto complexo com múltiplos componentes (RNA, Fuzzy, navegação, manipulação)
- Prazo fixo de entrega (06/01/2026)
- Requisito obrigatório: RNA + Fuzzy
- Necessidade de tempo para testes e otimização

**Justificativa Técnica:**
1. **Incremental Development:** Cada fase tem deliverable testável
2. **Risk Management:** Fases críticas (RNA, Fuzzy) com mais tempo
3. **Dependency Management:** Ordem respeita dependências técnicas
4. **Buffer:** 1 semana de margem para imprevistos

### Base teórica

**Metodologia de Desenvolvimento:**
- **Agile/Scrum adaptado:** Sprints temáticos
- **V-Model:** Cada fase tem verificação
- **Robotic Development Methodology:**
  - Perception → Decision → Action (pipeline clássico)
  - Thrun et al. (2005): "Sense-Plan-Act paradigm"

### Alternativas consideradas

1. **Desenvolvimento linear sem fases:**
   - ❌ Difícil rastrear progresso
   - ❌ Alto risco de atraso
2. **Fases paralelas (RNA + Fuzzy simultaneamente):**
   - ❌ Sobrecarga cognitiva
   - ❌ Difícil debugar problemas de integração
3. **Waterfall puro (tudo planejado antecipadamente):**
   - ❌ Inflexível para ajustes
   - ❌ Não permite aprendizado iterativo

### Impacto esperado

- ✅ Progresso mensurável (X% de tarefas completadas)
- ✅ Identificação precoce de problemas
- ✅ Possibilidade de ajustar escopo se necessário
- ✅ Entrega no prazo (06/01/2026)

**Checkpoints:**
- Final de cada fase: revisar TODO.md
- Atualizar DECISIONS.md com escolhas feitas
- Commit no git com tag da fase

---

## DECISÃO 005: Método de Instalação do Webots R2023b

**Data:** 2025-11-18
**Fase:** Fase 1.1 - Setup do Webots
**Status:** ✅ Implementado

### O que foi decidido

Utilizar instaladores oficiais do Webots R2023b:
- **macOS**: DMG universal (Intel/Apple Silicon)
- **Linux Ubuntu 22.04+**: Pacote Debian (.deb)
- **Método**: Download direto do GitHub releases (R2023b tag)
- **Pré-requisito**: Desinstalar versões anteriores antes da instalação

### Por que foi decidido

**Motivação:**
- Projeto exige versão específica (R2023b) devido à compatibilidade com IA_20252.wbt
- API do Webots pode ter mudanças incompatíveis entre versões
- Instaladores oficiais são mais confiáveis e bem testados

**Justificativa Técnica:**
1. **Estabilidade**: Instaladores oficiais têm resolução automática de dependências
2. **Suporte**: Documentação oficial alinhada com releases oficiais
3. **Reprodutibilidade**: Mesmo método funciona em todas as máquinas do time

### Base teórica

**Referências:**
- **Michel, O. (2004)**: "Webots: Professional Mobile Robot Simulation" - Estabelece Webots como simulador bem testado e mantido
- **Cyberbotics (2023)**: Documentação oficial R2023b - Procedimentos de instalação

**Análise da Pesquisa** (research.md Seção 1):
- DMG/DEB testado por comunidade durante 7+ anos
- Problemas conhecidos documentados (Gatekeeper macOS, drivers Linux)
- Universal Binary para Apple Silicon nativamente suportado

### Alternativas consideradas

1. **Docker Container:**
   - ✅ Isolamento total, CI/CD friendly
   - ❌ Complexidade de X11 forwarding para GUI
   - ❌ Overhead de performance
   - **Veredicto**: Adequado para CI/CD, não para desenvolvimento interativo

2. **Compilação do source:**
   - ✅ Máxima customização
   - ❌ Tempo de build ~1-2 horas
   - ❌ Complexidade de gerenciar dependências manualmente
   - **Veredicto**: Overhead desnecessário para versão estável

3. **APT Repository (Linux):**
   - ✅ Integração com sistema de pacotes
   - ❌ Risco de auto-upgrade para R2024a+ (quebra compatibilidade)
   - **Veredicto**: Aceitável se version pinning configurado

### Impacto esperado

- ✅ Setup reproduzível em <10 min (excluindo download)
- ✅ Todos desenvolvedores na mesma versão R2023b
- ✅ Compatibilidade garantida com world file IA_20252.wbt
- ✅ Menos troubleshooting de problemas de versão

**Métricas de sucesso:**
- `webots --version` retorna "Webots R2023b"
- World file IA_20252.wbt carrega em <30s sem erros

---

## DECISÃO 006: Estratégia de Integração Python-Webots

**Data:** 2025-11-18
**Fase:** Fase 1.1 - Setup do Webots
**Status:** ✅ Implementado

### O que foi decidido

Utilizar **abordagem híbrida**:
- **Python System-wide**: 3.8+ instalado no sistema (não só em venv)
- **Virtual Environment (venv)**: Para dependências de desenvolvimento (pytest, numpy, scipy)
- **PYTHONPATH**: Configurado para incluir biblioteca controller do Webots
- **Workflow**: Webots lançado do sistema, venv ativado para testes/desenvolvimento

### Por que foi decidido

**Motivação:**
- Webots R2021b+ tem problemas conhecidos com virtual environments
- Controladores Python executados pelo Webots precisam acessar módulo `controller`
- Desenvolvimento requer isolamento de dependências (pytest, linting)

**Justificativa Técnica:**
1. **Compatibilidade**: Webots ignora venv quando lançado de dentro dele (Issue #3462)
2. **Isolamento**: Venv protege sistema de conflitos de versões
3. **Flexibilidade**: Permite usar ferramentas de dev sem poluir sistema
4. **Padrão da Comunidade**: FAIRIS project e ROS2-Webots usam abordagem similar

### Base teórica

**Referências Técnicas:**
- **Webots GitHub Issue #3462**: "Python virtual environments don't work with R2021b"
- **PyPA (2023)**: Python Packaging Best Practices - Recomenda venv para projetos
- **FAIRIS Project (GitHub)**: Exemplo de integração Webots R2023b + venv

**Análise da Pesquisa** (research.md Seção 2):
- Configuração PYTHONPATH é prática padrão para external controllers
- Sistema Python + venv é única solução confiável para R2023b
- Conda tem mesmos problemas que venv padrão

### Alternativas consideradas

1. **Virtual Environment Only (sem Python system):**
   - ❌ Incompatível com Webots R2021b+
   - ❌ Controllers falham ao importar `controller` module
   - **Veredicto**: Não viável

2. **Conda Environment:**
   - ✅ Melhor isolamento cross-platform
   - ❌ Mesmos problemas de venv com Webots
   - ❌ Overhead adicional de gerenciamento
   - **Veredicto**: Sem vantagens práticas para este projeto

3. **System-wide pip install (sem isolamento):**
   - ✅ Simples, sem problemas de venv
   - ❌ Polui Python do sistema
   - ❌ Conflitos de versão entre projetos
   - **Veredicto**: Viola best practices

### Impacto esperado

- ✅ Controllers Webots funcionam sem modificações
- ✅ Testes isolados em venv (não afetam sistema)
- ✅ Setup documentado claramente (evita confusão)
- ⚠️ Trade-off: Requer Python system + venv (setup um pouco mais complexo)

**Métricas de sucesso:**
- `python3 --version` (sistema) retorna 3.8+
- `source venv/bin/activate && pip list` mostra pytest
- Controller em Webots importa `controller` sem erros

---

## DECISÃO 007: Framework de Testes Automatizados

**Data:** 2025-11-18
**Fase:** Fase 1.1 - Setup do Webots
**Status:** ✅ Implementado

### O que foi decidido

Utilizar **pytest com multi-layer testing**:
- **Framework**: pytest 7.4+
- **Estrutura**: Pirâmide de testes (Unit → Functional → Integration)
- **Markers**: `@pytest.mark.fast`, `@pytest.mark.slow`, `@pytest.mark.requires_webots`
- **Coverage**: pytest-cov com target >80%
- **CI/CD**: GitHub Actions com Xvfb para headless testing

### Por que foi decidido

**Motivação:**
- FR-012 exige testes automatizados para validação de setup
- Simuladores robóticos requerem testes em múltiplas camadas (env, sensores, integração)
- Reprodutibilidade: setup deve ser testável em novas máquinas
- Constitution Principle IV: Qualidade Senior (>80% coverage)

**Justificativa Técnica:**
1. **Pytest é padrão**: Comunidade Python robotics prefere pytest
2. **Flexibilidade**: Markers permitem selecionar testes (fast vs slow)
3. **Fixtures**: Gerenciamento de ciclo de vida do Webots em batch mode
4. **Plugins**: pytest-cov integra cobertura, pytest-xdist para paralelização

### Base teórica

**Referências:**
- **TestRiq (2023)**: "Robotic Software Testing: ROS2, Gazebo, and Motion Planning Validation" - Estabelece pirâmide de testes para sistemas robóticos
- **RobotPy Documentation (2025)**: "Unit Testing Robot Code" - Pytest como padrão para robotics
- **Webots Community**: Batch mode (`--batch --mode=fast`) é pattern para automated testing

**Análise da Pesquisa** (research.md Seção 3):
- Pirâmide: Fast (<5s) → Medium (10-30s) → Slow (1-5min)
- Webots headless com Xvfb permite CI/CD
- Markers melhoram developer experience (rodar só fast tests localmente)

### Alternativas consideradas

1. **unittest (Python standard library):**
   - ✅ Sem dependências externas
   - ❌ Sintaxe verbose, fixtures limitadas
   - **Veredicto**: pytest é mais moderno e flexível

2. **ROS2 Testing Framework (ros2test):**
   - ✅ Ferramentas ricas para robotics
   - ❌ Requer instalação ROS2 (overhead)
   - ❌ Projeto não usa ROS
   - **Veredicto**: Over-engineered para Python-only project

3. **Manual Testing Only:**
   - ❌ Não reproduzível
   - ❌ Não integra com CI/CD
   - **Veredicto**: Insuficiente para production-grade project

### Impacto esperado

- ✅ 100% pass rate quando setup correto (SC-003)
- ✅ Detecta problemas antes de manual testing
- ✅ CI/CD valida PRs automaticamente
- ✅ Novos desenvolvedores validam setup rapidamente

**Métricas de sucesso:**
- `pytest tests/test_webots_setup.py` passa 4/4 testes
- Execução completa em <5min
- Coverage >80% dos scripts de setup

**Estrutura de Testes Phase 1.1:**
```
tests/
├── test_webots_setup.py        # 4 testes (installation, env validation)
├── fixtures/
│   └── conftest.py             # pytest fixtures (webots_process, temp_venv)
└── pytest.ini                  # Configuração de markers
```

---

## DECISÃO 008: Abordagem de Validação de Sensores

**Data:** 2025-11-18
**Fase:** Fase 1.1 - Setup do Webots (DEFERRED para Fase 2)
**Status:** 📋 Planejado (implementação em Fase 2)

### O que foi decidido

Utilizar **validação multi-estágio**:
1. **Format Validation**: Verificar array size, data types, resolução
2. **Range Validation**: Verificar valores estão em ranges físicos plausíveis
3. **Temporal Consistency**: Verificar estabilidade ao longo do tempo
4. **Content Validation**: Verificar dados fazem sentido (obstáculos detectados, cores visíveis)

**LIDAR (512 pontos)**:
- Array size: Exatamente 512 floats
- Range values: [0.01m, 10m] para finitos
- Obstacle detection: >10% de raios finitos (não todos `inf`)
- Temporal variance: <0.01 para robô estacionário

**Camera (128x128 BGRA)**:
- Resolution: width=128, height=128
- Format: 128×128×4 bytes (BGRA)
- Pixel range: [0, 255] uint8
- Content: Não monochrome (variance RGB channels)
- Temporal stability: <5.0 mean pixel diff entre frames

### Por que foi decidido

**Motivação:**
- Sensores devem retornar dados válidos ANTES de desenvolver percepção (Fase 2)
- Validação precoce evita debugging complexo depois
- User Story 3 (P1) requer validação de sensores funcionais

**Justificativa Técnica:**
1. **Multi-stage**: Detecta problemas em níveis diferentes (format vs content)
2. **Physical Plausibility**: Arena 7x4m → ranges >10m são implausíveis
3. **Temporal Checks**: Robô parado deve ter leituras estáveis
4. **Statistical Validation**: Variance/mean detecta dados degenerados

### Base teórica

**Referências:**
- **Claytex (2023)**: "LiDAR Sensor Validation: How to Ensure Accurate Virtual Models" - Estabelece necessidade de validação multi-estágio
- **Springer (2020)**: "Sequential lidar sensor system simulation: a modular approach" - Valida 512-point arrays e ranges plausíveis
- **PMC/NIH (2023)**: "LiMOX—A Point Cloud Lidar Model Toolbox" - Documenta configuração 512-point padrão
- **Webots Documentation (2023)**: "Camera Sensors Guide" - BGRA format, ranges [0,255]

**Análise da Pesquisa** (research.md Seção 4):
- Sensor initialization: LIDAR <1s (10 steps), Camera <1s (20 steps)
- Performance benchmarks: Both <1s first valid data
- Validation patterns: Format → Range → Temporal → Content

### Alternativas consideradas

1. **Visual Inspection Only:**
   - ❌ Não reproduzível
   - ❌ Subjetivo, tempo-consuming
   - **Veredicto**: Inaceitável para production testing

2. **Statistical Distribution Tests (Chi-square, KS test):**
   - ✅ Rigor estatístico
   - ❌ Requer ground truth distribution
   - ❌ Overkill para setup phase
   - **Veredicto**: Defer para Fase 2 (perception validation)

3. **Sensor Fusion Validation (LIDAR + Camera alignment):**
   - ✅ Valida calibração extrinsic
   - ❌ Complexo, requer geometria de cena conhecida
   - **Veredicto**: Out of scope para Phase 1.1, defer para Fase 6 (integração)

### Impacto esperado

- ✅ Detecta problemas de sensor ANTES de implementar RNA
- ✅ SC-005 & SC-006: Dados válidos em <1s (verificável)
- ✅ Baseline para Fase 2: sensores funcionais garantidos
- ⚠️ Requer controller implementation (por isso DEFERRED)

**Métricas de sucesso (Fase 2):**
- LIDAR: 512 pontos, >10% finite, variance <0.01
- Camera: 128x128x4, pixels [0,255], color variance >100
- Init time: Both <1s from enable

**Nota**: User Story 3 (Sensor Validation) é P1 (Critical), mas implementação requer controllers que serão desenvolvidos na Fase 2. Por isso, tasks T028-T031 estão marcadas como DEFERRED no tasks.md.

---

## Template para Novas Decisões

```markdown
## DECISÃO XXX: [Título da Decisão]

**Data:** YYYY-MM-DD
**Fase:** [Nome da Fase]
**Status:** [Planejado / Em implementação / Implementado / Revisado]

### O que foi decidido

[Descrição clara e objetiva da decisão]

### Por que foi decidido

**Motivação:**
[Contexto e razões para a decisão]

**Justificativa Técnica:**
1. [Razão 1]
2. [Razão 2]

### Base teórica

**Referências:**
- [Autor et al. (Ano)]: [Contribuição]
- [Paper/livro relevante]

**Conceitos aplicados:**
- [Teoria X aplicada no contexto Y]

### Alternativas consideradas

1. **[Alternativa 1]:**
   - ❌ [Por que foi descartada]
2. **[Alternativa 2]:**
   - ❌ [Por que foi descartada]
3. **[Alternativa escolhida]:**
   - ✅ [Vantagens]

### Impacto esperado

- ✅ [Benefício 1]
- ✅ [Benefício 2]
- ⚠️ [Possível trade-off]

**Métricas de sucesso:**
- [Como medir se decisão foi boa]

### Notas adicionais

[Qualquer informação relevante não coberta acima]

---
```

---

## Próximas Decisões a Documentar

**Fase 1 (Exploração):**
- [x] Versão do Webots e Python escolhidas (DECISÃO 005, 006)
- [x] Estrutura de testes inicial (DECISÃO 007, 008)

**Fase 2 (Percepção):**
- [ ] Arquitetura RNA para LIDAR (MLP simples vs PointNet adaptado)
- [ ] Modelo CNN para detecção (YOLO vs SSD vs custom)
- [ ] Framework de deep learning (PyTorch vs TensorFlow)
- [ ] Estratégia de treinamento (dados sintéticos vs reais)

**Fase 3 (Controle):**
- [ ] Tipo de controlador fuzzy (Mamdani vs Sugeno)
- [ ] Número e tipo de variáveis linguísticas
- [ ] Funções de pertinência (triangular vs gaussiana)
- [ ] Total de regras fuzzy

**Fase 4 (Navegação):**
- [ ] Estratégia de navegação (reativa vs path planning)
- [ ] Mapeamento local (occupancy grid vs landmark-based)
- [ ] Localização (odometria vs SLAM)

**Fase 5 (Manipulação):**
- [ ] Sequência de grasping (posições do braço)
- [ ] Estratégia para identificar caixas (hardcode vs detecção visual)

---

## Registro de Mudanças neste Documento

| Data | Mudança | Autor |
|------|---------|-------|
| 2025-11-18 | Criação inicial com decisões 001-004 | Luis Felipe |
| 2025-11-18 | Adicionadas decisões 005-008 (Fase 1.1 - Setup do Webots) | Luis Felipe |

---

**Nota:** Este documento deve ser atualizado **ANTES** de cada implementação significativa. Decisões tomadas "no calor do momento" devem ser documentadas retrospectivamente no mesmo dia.
