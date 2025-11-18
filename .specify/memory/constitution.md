# YouBot Autônomo - Constitution

**Projeto:** Sistema Autônomo de Coleta e Organização de Objetos com YouBot
**Disciplina:** MATA64 - Inteligência Artificial - UFBA
**Aluno:** Luis Felipe Cordeiro Sena
**Data de Entrega:** 06/01/2026, 23:59

---

## Core Principles

### I. Fundamentação Científica (NON-NEGOTIABLE)

**Todas decisões técnicas devem ter base teórica científica peer-reviewed.**

- Antes de implementar qualquer funcionalidade, consultar REFERENCIAS.md
- Documentar decisão em DECISIONS.md com:
  - Alternativas consideradas (mín. 2)
  - Justificativa baseada em papers (citação ABNT)
  - Base teórica aplicável
  - Trade-offs analisados
- Proibido decisões "porque sim" ou "achismo"
- Top 10 referências essenciais devem ser citadas na apresentação final

**Requisitos Obrigatórios da Disciplina:**
1. **RNA (MLP ou CNN):** Processamento LIDAR e/ou detecção visual
2. **Lógica Fuzzy:** Controle de ações do robô

### II. Rastreabilidade Total

**Cada mudança significativa deve ser documentada antes da implementação.**

- DECISIONS.md é o single source of truth para escolhas técnicas
- Template obrigatório: O que, Por quê, Base teórica, Alternativas, Impacto
- Atualizar ANTES de implementar, não depois
- Git commits descritivos por fase com referência a decisões
- Métricas de performance registradas em logs/

**Hierarquia de Documentação:**
1. CLAUDE.md → Contexto geral e diretrizes
2. REFERENCIAS.md → Base científica (80+ papers)
3. TODO.md → Plano em 8 fases com tarefas
4. DECISIONS.md → Registro cronológico de decisões
5. spec/* → Especificações geradas pelo SpecKit

### III. Desenvolvimento Incremental por Fases

**Proibido avançar para próxima fase sem concluir anterior.**

**Fases do Projeto (8 semanas + 1 buffer):**
1. **Fase 0 - Setup:** ✅ Documentação e estrutura
2. **Fase 1 - Exploração:** Setup Webots, análise de sensores (3 dias)
3. **Fase 2 - Percepção RNA:** LIDAR + Detecção de cubos (10 dias)
4. **Fase 3 - Controle Fuzzy:** Navegação e decisões (7 dias)
5. **Fase 4 - Navegação:** Mapeamento e path planning (5 dias)
6. **Fase 5 - Manipulação:** Grasping e deposição (4 dias)
7. **Fase 6 - Integração:** Sistema completo end-to-end (5 dias)
8. **Fase 7 - Otimização:** Refinamento e métricas (5 dias)
9. **Fase 8 - Apresentação:** Vídeo 15 min SEM CÓDIGO (7 dias)

**Deliverable Obrigatório por Fase:**
- Código testado e funcional
- Decisões documentadas em DECISIONS.md
- Testes unitários passando
- TODO.md atualizado com progresso

### IV. Qualidade Senior

**Código deve seguir padrões de excelência acadêmica e industrial.**

- **Arquitetura Modular:**
  - `src/perception/` - RNA para LIDAR e câmera
  - `src/control/` - Lógica Fuzzy e estados
  - `src/navigation/` - Mapeamento local
  - `src/manipulation/` - Grasping/deposição
  - `src/main_controller.py` - Loop principal

- **Testes Obrigatórios:**
  - Testes unitários para cada módulo crítico
  - Target: >80% cobertura
  - Testes passando antes de marcar tarefa como concluída

- **Code Quality:**
  - PEP8 compliance (usar black)
  - Type hints quando aplicável
  - Docstrings nos módulos principais
  - Logs estruturados em `logs/`

### V. Restrições Disciplinares (NON-NEGOTIABLE)

**Violação resulta em perda de pontos na avaliação.**

❌ **PROIBIDO:**
1. **Modificar `supervisor.py`** - Spawn de cubos é fixo
2. **Mostrar código-fonte no vídeo** - Perda de 3-10 pontos
3. **Mostrar texto excessivo nos slides** - Foco em imagens e processos
4. **Usar GPS na demo final** - Pode treinar, mas apresentação SEM GPS
5. **Código sem fundamentação teórica** - Toda escolha justificada

✅ **PERMITIDO:**
- Modelos pré-treinados (transfer learning)
- Bibliotecas de terceiros (PyTorch, scikit-fuzzy, etc.)
- Trechos de código prontos
- Arquiteturas de RNA já existentes
- **Condição:** Tudo explicado no vídeo com citações

### VI. Workflow SpecKit (MANDATÓRIO)

**Para cada fase do TODO.md, seguir ciclo completo:**

1. **Specify** (`/speckit.specify`):
   - Branch: `00X-feature-name`
   - Criar `specs/<feature>/spec.md`
   - Consultar DECISIONS.md para contexto de decisões anteriores
   - Referenciar REFERENCIAS.md para base teórica

2. **Clarify** (`/speckit.clarify`):
   - Identificar ambiguidades
   - Perguntas de esclarecimento
   - Atualizar spec.md

3. **Plan** (`/speckit.plan`):
   - Gerar `specs/<feature>/plan.md`
   - Pesquisa (research.md) e design (data-model.md)
   - Ordem de implementação

4. **Tasks** (`/speckit.tasks`):
   - Gerar `specs/<feature>/tasks.md`
   - Checklist granular (~40-50 tasks)
   - Dependências claras

5. **Implement** (`/speckit.implement`):
   - Executar tasks sequencialmente
   - **ANTES de decisões técnicas:** Ler DECISIONS.md seções relevantes
   - **DURANTE implementação:** Documentar novas decisões em DECISIONS.md
   - Commit incremental por milestone

6. **Validate**:
   - Testes passando (pytest)
   - `/speckit.analyze` para consistência
   - Merge to main após validação

**Aprendizado Contínuo:**
- Ler DECISIONS.md completo antes de começar nova fase
- Cada decisão técnica = consultar papers em REFERENCIAS.md
- Atualizar DECISIONS.md ANTES de implementar

---

## Requisitos Técnicos

### Objetivo do Sistema

**Robô autônomo que:**
- Coleta 15 cubos coloridos (verde, azul, vermelho) aleatórios
- Identifica cor via câmera RGB
- Deposita cada cubo na caixa correspondente
- Evita obstáculos (caixotes de madeira) usando LIDAR
- **SEM GPS** - navegação baseada apenas em sensores

### Hardware e Sensores

**Plataforma:** KUKA YouBot (simulador Webots)
- Base omnidirecional (rodas mecanum)
- Braço robótico 5-DOF
- Garra paralela
- **LIDAR** para detecção de obstáculos
- **Câmera RGB** para identificação de cores

**Referências:**
- Bischoff et al. (2011): Especificações do YouBot
- Taheri et al. (2015): Cinemática mecanum wheels

### Arquitetura Obrigatória

**Pipeline Sense-Plan-Act:**
1. **Percepção:**
   - LIDAR → RNA → Detecção de obstáculos
   - Câmera → CNN → Classificação de cores
2. **Decisão:**
   - Fuzzy Controller → Ações (velocidade, direção)
   - State Machine → Estados (buscar, aproximar, pegar, levar, soltar)
3. **Atuação:**
   - Base (vx, vy, omega)
   - Arm (posições preset + cinemática inversa)
   - Gripper (grip/release)

**Referências:**
- Thrun et al. (2005): Sense-Plan-Act paradigm
- Goodfellow et al. (2016): Deep Learning para percepção
- Zadeh (1965), Mamdani (1975): Lógica Fuzzy para controle

---

## Critérios de Sucesso

### Mínimo Viável (Aprovação)
- [ ] Sistema coleta ≥10/15 cubos
- [ ] Identificação de cores >80% precisa
- [ ] Evitação de obstáculos funcional (sem colisões críticas)
- [ ] RNA para LIDAR implementada e funcional
- [ ] Lógica Fuzzy implementada e funcional
- [ ] Vídeo 15min explicando (SEM CÓDIGO!)
- [ ] Código entregue e documentado

### Excelência (Nota Máxima)
- [ ] Sistema coleta 15/15 cubos consistentemente (3+ runs)
- [ ] Identificação de cores >95% precisa
- [ ] Zero colisões com obstáculos
- [ ] Tempo otimizado (<5 min total)
- [ ] Apresentação visual impecável
- [ ] Top 10 papers citados corretamente
- [ ] Código modular e bem testado (>80% coverage)
- [ ] DECISIONS.md completo com todas escolhas justificadas

---

## Material de Apresentação

**Vídeo de 15 minutos (SEM CÓDIGO-FONTE!):**

**Estrutura Obrigatória:**
1. **Intro (1 min):** Problema e objetivos
2. **Fundamentação (3 min):** RNA + Fuzzy com citações
3. **Arquitetura (2 min):** Diagrama do sistema
4. **Percepção (2 min):** LIDAR + Detecção visual
5. **Controle (2 min):** Fuzzy rules + State machine
6. **Demo (4 min):** Robô coletando 15 cubos
7. **Resultados (1 min):** Métricas e conclusões

**Template de Slides:**
- **Arquivo:** `slides-template/main.tex` (LaTeX Beamer)
- **Tema:** DCC (já configurado)
- **Formato:** 16:9, português, bibliografia integrada

**Material Visual Permitido:**
- Diagramas de arquitetura e processos (Draw.io, TikZ)
- Gráficos de performance (matplotlib, seaborn)
- Vídeos da simulação (OBS Studio)
- Visualizações de dados (LIDAR point clouds, detecções)
- Equações matemáticas (LaTeX)
- Funções de pertinência fuzzy (plots)
- Tabelas comparativas (resultados, métricas)
- Fluxogramas e state machines

**Material PROIBIDO (PERDE PONTOS):**
- Código-fonte Python/C/C++
- Screenshots de código ou IDE
- Prints de terminal com código
- Texto excessivo (máx 3-4 bullet points por slide)
- Parágrafos longos (usar tópicos curtos)
- Slides "wall of text" (texto deve ser APOIO, não foco)

**FILOSOFIA:** Apresentador (Luis Felipe) explica verbalmente, slides só reforçam visualmente

---

## Governance

### Priorização de Documentos

1. **CLAUDE.md** - Diretrizes gerais (este arquivo referencia)
2. **TODO.md** - Plano de execução (fases e tarefas)
3. **DECISIONS.md** - Decisões técnicas (NON-NEGOTIABLE atualizar)
4. **REFERENCIAS.md** - Base científica (consulta obrigatória)
5. **spec/** - Especificações SpecKit (geradas por comandos)

### Conflitos

- Constitution (este arquivo) supersede tudo
- Em caso de conflito, DECISIONS.md documenta escolha
- Decisões devem referenciar esta constitution

### Amendments

- Mudanças na constitution requerem:
  - Justificativa científica
  - Documentação em DECISIONS.md
  - Não podem violar requisitos da disciplina

### Compliance

- Todo PR/commit deve verificar compliance com:
  - Fundamentação científica (Princípio I)
  - Rastreabilidade (Princípio II)
  - Restrições disciplinares (Princípio V)
- Decisões complexas exigem revisão de trade-offs

---

**Version**: 1.0.0
**Ratified**: 2025-11-18
**Last Amended**: 2025-11-18
**Next Review**: Após Fase 2 (conclusão da percepção)
