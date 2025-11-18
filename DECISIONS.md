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
- [ ] Versão do Webots e Python escolhidas
- [ ] Estrutura de testes inicial

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

---

**Nota:** Este documento deve ser atualizado **ANTES** de cada implementação significativa. Decisões tomadas "no calor do momento" devem ser documentadas retrospectivamente no mesmo dia.
