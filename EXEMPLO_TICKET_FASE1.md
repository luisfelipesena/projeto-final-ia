# Exemplo de Ticket - Fase 1.1: Setup e Exploração do Webots

**Baseado em:** TODO.md → Fase 1 → Seção 1.1 "Setup do Webots"
**Data:** 2025-11-18
**Status:** Exemplo de workflow SpecKit

---

## Contexto da Tarefa

Primeira tarefa técnica do projeto após a fase de documentação.

**Objetivo:** Configurar ambiente Webots e validar que simulação roda corretamente.

**Referência em TODO.md:**
```
### Fase 1: Ambiente e Exploração Inicial
Prazo: 3 dias

#### 1.1 Setup do Webots
- [ ] Instalar/atualizar Webots
- [ ] Verificar versão do Python (compatibilidade com Webots)
- [ ] Testar abertura do mundo IA_20252.wbt
- [ ] Verificar spawn de cubos pelo supervisor
- [ ] Documentar setup em DECISIONS.md

Deliverable: Simulação rodando sem erros
```

---

## Workflow SpecKit - Passo a Passo

### Passo 1: Specify (`/speckit.specify`)

**Prompt Inicial:**
```
Criar especificação para tarefa de Setup do Webots conforme Fase 1.1 do TODO.md.

Contexto:
- Primeira tarefa técnica do projeto YouBot Autônomo
- Objetivo: garantir ambiente Webots funcional
- Requisitos: Python compatível, simulação IA_20252.wbt rodando, spawn de 15 cubos
- Deliverable: Simulação validada sem erros

Consultar:
- TODO.md (Fase 1.1)
- CLAUDE.md (requisitos de sistema)
- Constitution (Princípio IV - Qualidade)

Gerar spec/webots-setup/spec.md
```

**Output Esperado:** `spec/webots-setup/spec.md`

```markdown
# Webots Setup - Especificação

## Objetivo

Configurar e validar ambiente de simulação Webots para desenvolvimento do projeto YouBot Autônomo.

## Requisitos Funcionais

### RF1: Instalação do Webots
- Webots R2023a ou superior instalado
- Licença ativa (gratuita para uso educacional)
- Interface gráfica funcional

### RF2: Compatibilidade Python
- Python 3.8+ disponível no sistema
- Webots detecta interpretador Python corretamente
- Módulo `controller` do Webots acessível

### RF3: Simulação Base Funcional
- Mundo IA_20252.wbt abre sem erros
- Supervisor spawna 15 cubos aleatoriamente
- YouBot aparece na posição inicial
- LIDAR e câmera RGB funcionais

## Requisitos Não-Funcionais

### RNF1: Documentação
- Setup documentado em DECISIONS.md
- Versões de software registradas
- Problemas encontrados e soluções documentadas

### RNF2: Reprodutibilidade
- Outro desenvolvedor consegue reproduzir setup
- Instruções claras em README.md (seção Setup)

## Critérios de Aceitação

- [ ] Webots abre normalmente
- [ ] Simulação IA_20252.wbt carrega em <30s
- [ ] 15 cubos spawned com cores corretas (5 verde, 5 azul, 5 vermelho)
- [ ] Nenhum erro no console do Webots
- [ ] Python controller conecta com robô
- [ ] DECISIONS.md atualizado com DECISÃO 005

## Restrições

- Não modificar supervisor.py (Constitution, Princípio V)
- Usar apenas versões estáveis do Webots (não dev/nightly)

## Referências

- Michel (2004): Webots simulator
- Cyberbotics Documentation: https://cyberbotics.com/doc/
```

---

### Passo 2: Clarify (`/speckit.clarify`)

**Prompt Automático (gerado pelo SpecKit):**
```
Analisar spec/webots-setup/spec.md e identificar ambiguidades.

Fazer até 5 perguntas de esclarecimento sobre:
- Requisitos não claros
- Casos de borda
- Decisões técnicas necessárias
```

**Output Esperado:** Perguntas de esclarecimento

```markdown
# Clarificações Necessárias - Webots Setup

## Questões Identificadas

### Q1: Versão específica do Webots?
**Contexto:** Spec diz "R2023a ou superior"
**Pergunta:** Qual versão exata devemos usar? R2023a, R2023b, R2024a?
**Impacto:** Compatibilidade de API, bugs conhecidos

### Q2: Sistema operacional alvo?
**Contexto:** Setup pode variar entre macOS/Linux/Windows
**Pergunta:** Desenvolveremos em qual OS? Precisamos suportar múltiplos?
**Impacto:** Instruções de instalação, paths, comandos

### Q3: Distribuição de cores dos cubos?
**Contexto:** Spec diz "5 verde, 5 azul, 5 vermelho"
**Pergunta:** Essa distribuição é garantida pelo supervisor ou esperada?
**Impacto:** Testes de validação

### Q4: Ambiente virtual Python?
**Contexto:** Boas práticas recomendam venv
**Pergunta:** Usar venv/conda ou Python global?
**Impacto:** Gerenciamento de dependências, reprodutibilidade

### Q5: Testes automatizados de setup?
**Contexto:** Queremos validar setup programaticamente
**Pergunta:** Criar script de validação ou checklist manual?
**Impacto:** Tempo de desenvolvimento, confiabilidade
```

**Respostas (fornecidas por você):**
```markdown
## Respostas

### A1: Webots R2023b
- Versão estável mais recente da família 2023
- Boa documentação disponível
- Sem bugs críticos conhecidos

### A2: macOS (desenvolvimento), documentar para Linux
- Desenvolvimento primário em macOS
- Instruções devem funcionar em Linux (Ubuntu 22.04+)
- Windows: best-effort

### A3: Distribuição aleatória
- Supervisor usa random.choice() - não garante 5-5-5
- Validar que TOTAL = 15 e cores corretas
- Aceitar distribuição desbalanceada

### A4: Ambiente virtual (venv)
- Criar venv em projeto: `projeto-final-ia/venv/`
- Adicionar ao .gitignore
- Documentar ativação em README

### A5: Script de validação Python
- Criar `tests/test_webots_setup.py`
- Validar: Webots instalado, Python 3.8+, mundo carrega
- Executar via pytest
```

**Atualização:** `spec/webots-setup/spec.md` incorpora respostas

---

### Passo 3: Plan (`/speckit.plan`)

**Prompt Automático:**
```
Baseado em spec/webots-setup/spec.md (atualizado com clarificações),
gerar plano de implementação detalhado.

Dividir em subtarefas ordenadas com dependências.
```

**Output Esperado:** `spec/webots-setup/plan.md`

```markdown
# Plano de Implementação - Webots Setup

## Visão Geral

Configuração do ambiente em 3 etapas principais:
1. Instalação e configuração
2. Validação da simulação
3. Documentação

**Tempo Estimado:** 4-6 horas
**Complexidade:** Baixa (setup padrão)

## Subtarefas

### Tarefa 1: Instalação do Webots
**Prioridade:** P0 (bloqueante)
**Dependências:** Nenhuma
**Tempo:** 1h

**Passos:**
1. Download Webots R2023b do site oficial
2. Instalação conforme OS (macOS: .dmg, Linux: .deb)
3. Primeira execução (aceitar licença educacional)
4. Verificar instalação: `webots --version`

**Critérios de Sucesso:**
- Webots abre interface gráfica
- Versão exibida: R2023b

### Tarefa 2: Configuração do Python
**Prioridade:** P0
**Dependências:** Tarefa 1
**Tempo:** 30min

**Passos:**
1. Verificar Python: `python3 --version` (≥3.8)
2. Criar venv: `python3 -m venv venv`
3. Ativar: `source venv/bin/activate`
4. Instalar dependências base: `pip install -r requirements.txt`
5. Verificar módulo controller (fornecido por Webots)

**Critérios de Sucesso:**
- Python 3.8+ confirmado
- venv ativo
- Pip packages instalados

### Tarefa 3: Teste da Simulação Base
**Prioridade:** P0
**Dependências:** Tarefas 1, 2
**Tempo:** 1h

**Passos:**
1. Abrir Webots
2. File → Open World → navegar para `IA_20252/worlds/IA_20252.wbt`
3. Executar simulação (botão Play)
4. Observar:
   - Supervisor executa (console output)
   - 15 cubos spawned
   - YouBot na posição inicial
5. Verificar cores dos cubos visualmente
6. Parar simulação (botão Stop)

**Critérios de Sucesso:**
- Mundo carrega em <30s
- 15 cubos visíveis
- Sem erros no console Webots
- Robô aparece corretamente

### Tarefa 4: Validação dos Sensores
**Prioridade:** P1
**Dependências:** Tarefa 3
**Tempo:** 1h

**Passos:**
1. Modificar `IA_20252/controllers/youbot/youbot.py` temporariamente:
   - Adicionar prints dos dados do LIDAR
   - Adicionar print da resolução da câmera
2. Executar simulação
3. Verificar output:
   - LIDAR retorna ranges (array numérico)
   - Câmera retorna imagem (width x height)
4. Reverter modificações temporárias

**Critérios de Sucesso:**
- LIDAR retorna dados válidos
- Câmera retorna imagem (ex: 640x480)
- Valores fazem sentido (LIDAR: 0-5m)

### Tarefa 5: Script de Validação
**Prioridade:** P2
**Dependências:** Tarefas 2, 3
**Tempo:** 1.5h

**Passos:**
1. Criar `tests/test_webots_setup.py`:
   ```python
   import subprocess
   import sys

   def test_webots_installed():
       """Verifica instalação do Webots"""
       result = subprocess.run(['webots', '--version'], ...)
       assert 'R2023b' in result.stdout

   def test_python_version():
       """Verifica Python 3.8+"""
       assert sys.version_info >= (3, 8)

   def test_world_file_exists():
       """Verifica arquivo do mundo"""
       assert Path('IA_20252/worlds/IA_20252.wbt').exists()
   ```
2. Executar: `pytest tests/test_webots_setup.py -v`

**Critérios de Sucesso:**
- Todos testes passam
- Output claro do pytest

### Tarefa 6: Documentação
**Prioridade:** P1
**Dependências:** Tarefas 1-5 concluídas
**Tempo:** 1h

**Passos:**
1. Atualizar DECISIONS.md:
   - Criar DECISÃO 005: "Escolha de Webots R2023b"
   - Justificar: estabilidade, documentação, compatibilidade
   - Alternativas: R2024a (muito recente), R2023a (bugs)
2. Criar README.md (ou atualizar):
   - Seção "Setup do Ambiente"
   - Comandos de instalação (macOS + Linux)
   - Troubleshooting comum
3. Registrar versões em `docs/environment.md`:
   - Webots: R2023b
   - Python: 3.x.y (specific)
   - OS: macOS versão / Ubuntu versão

**Critérios de Sucesso:**
- DECISIONS.md atualizado
- README com instruções claras
- Versões documentadas

## Ordem de Execução

```
[Tarefa 1] → [Tarefa 2] → [Tarefa 3] → [Tarefa 4]
                              ↓
                          [Tarefa 5]
                              ↓
                    [Tarefa 6] (paralela após 1-5)
```

## Riscos e Mitigações

### Risco 1: Webots não instala (incompatibilidade OS)
**Probabilidade:** Baixa
**Impacto:** Alto
**Mitigação:** Usar VM com Ubuntu 22.04 como fallback

### Risco 2: Python não reconhecido pelo Webots
**Probabilidade:** Média (macOS)
**Impacto:** Médio
**Mitigação:** Configurar WEBOTS_PYTHON_PATH manualmente

### Risco 3: Simulação lenta (hardware insuficiente)
**Probabilidade:** Baixa
**Impacto:** Médio
**Mitigação:** Reduzir qualidade gráfica, fechar apps

## Rollback

Se setup falhar criticamente:
1. Desinstalar Webots
2. Remover venv
3. Documentar problema em DECISIONS.md
4. Buscar alternativa (versão anterior, Gazebo)

## Próximos Passos

Após conclusão desta tarefa:
→ Fase 1.2: Exploração dos Controles Base
→ Fase 1.3: Análise dos Sensores
```

---

### Passo 4: Tasks (`/speckit.tasks`)

**Prompt Automático:**
```
Baseado em spec/webots-setup/plan.md, gerar checklist detalhado de tarefas.

Formato: Markdown com checkboxes [ ].
```

**Output Esperado:** `spec/webots-setup/tasks.md`

```markdown
# Tarefas - Webots Setup

**Feature:** Setup e Validação do Ambiente Webots
**Prazo:** 2025-11-21 (3 dias)
**Responsável:** Luis Felipe Cordeiro Sena

---

## Checklist de Implementação

### 🔧 Instalação do Webots

- [ ] Baixar Webots R2023b do site oficial (https://cyberbotics.com)
- [ ] Instalar Webots conforme sistema operacional
  - macOS: Abrir .dmg e arrastar para /Applications
  - Linux: `sudo apt install ./webots_2023b_amd64.deb`
- [ ] Executar Webots pela primeira vez
- [ ] Aceitar licença educacional
- [ ] Verificar versão: `webots --version` exibe "R2023b"
- [ ] Testar abertura de mundo exemplo (File → Open World → samples)

### 🐍 Configuração do Python

- [ ] Verificar Python instalado: `python3 --version`
- [ ] Confirmar versão ≥3.8
- [ ] Criar ambiente virtual: `python3 -m venv venv`
- [ ] Ativar venv: `source venv/bin/activate` (macOS/Linux)
- [ ] Atualizar pip: `pip install --upgrade pip`
- [ ] Instalar dependências: `pip install -r requirements.txt`
- [ ] Verificar instalações: `pip list`
- [ ] Adicionar venv/ ao .gitignore (se já não estiver)

### 🌍 Teste da Simulação

- [ ] Abrir Webots
- [ ] Carregar mundo: File → Open → `IA_20252/worlds/IA_20252.wbt`
- [ ] Aguardar carregamento (<30s)
- [ ] Pressionar Play (▶️)
- [ ] Observar console:
  - [ ] Supervisor inicia
  - [ ] Mensagem "Spawn complete. Spawned X/15 objects"
- [ ] Observar arena 3D:
  - [ ] 15 cubos visíveis
  - [ ] Cores variadas (verde, azul, vermelho)
  - [ ] YouBot na posição inicial
  - [ ] Caixas de depósito visíveis (verde, azul, vermelha)
- [ ] Verificar ausência de erros no console
- [ ] Parar simulação (⏹️)
- [ ] Fechar Webots

### 📡 Validação dos Sensores

- [ ] Criar branch git: `git checkout -b test/sensor-validation`
- [ ] Modificar `IA_20252/controllers/youbot/youbot.py`:
  ```python
  # Adicionar após linha 19 (self.lidar.enable)
  print(f"LIDAR enabled. FOV: {self.lidar.getFov()}")
  print(f"LIDAR points: {self.lidar.getNumberOfPoints()}")

  # Adicionar após linha 16 (self.camera.enable)
  print(f"Camera enabled. Resolution: {self.camera.getWidth()}x{self.camera.getHeight()}")
  ```
- [ ] Executar simulação
- [ ] Verificar output no console:
  - [ ] LIDAR FOV exibido (ex: 3.14)
  - [ ] LIDAR points exibido (ex: 512)
  - [ ] Camera resolution exibida (ex: 640x480)
- [ ] Adicionar leitura de dados:
  ```python
  # No método run() (criar se não existir)
  ranges = self.lidar.getRangeImage()
  print(f"LIDAR sample: {ranges[:5]}")  # Primeiros 5 pontos
  ```
- [ ] Executar novamente
- [ ] Confirmar ranges numéricos (ex: [2.34, 5.12, inf, ...])
- [ ] Reverter modificações: `git checkout youbot.py`
- [ ] Deletar branch: `git branch -D test/sensor-validation`

### ✅ Script de Validação

- [ ] Criar diretório: `mkdir -p tests`
- [ ] Criar `tests/__init__.py` (vazio)
- [ ] Criar `tests/test_webots_setup.py`:
  ```python
  import subprocess
  import sys
  from pathlib import Path

  def test_webots_installed():
      """Testa se Webots está instalado"""
      result = subprocess.run(
          ['webots', '--version'],
          capture_output=True,
          text=True
      )
      assert result.returncode == 0
      assert 'R2023b' in result.stdout

  def test_python_version():
      """Testa versão do Python"""
      assert sys.version_info >= (3, 8), \
          f"Python 3.8+ required, found {sys.version_info}"

  def test_world_file_exists():
      """Testa existência do arquivo do mundo"""
      world_path = Path('IA_20252/worlds/IA_20252.wbt')
      assert world_path.exists(), f"World file not found at {world_path}"

  def test_controller_files_exist():
      """Testa existência dos controllers Python"""
      controllers = [
          'IA_20252/controllers/youbot/youbot.py',
          'IA_20252/controllers/youbot/base.py',
          'IA_20252/controllers/youbot/arm.py',
          'IA_20252/controllers/youbot/gripper.py',
          'IA_20252/controllers/supervisor/supervisor.py'
      ]
      for ctrl in controllers:
          assert Path(ctrl).exists(), f"Controller not found: {ctrl}"
  ```
- [ ] Executar testes: `pytest tests/test_webots_setup.py -v`
- [ ] Confirmar 4/4 testes passam
- [ ] Commit: `git add tests/ && git commit -m "Add Webots setup validation tests"`

### 📝 Documentação

- [ ] Atualizar DECISIONS.md:
  - [ ] Adicionar DECISÃO 005: "Escolha de Webots R2023b"
  - [ ] Seção "O que foi decidido": Usar Webots R2023b
  - [ ] Seção "Por que": Estabilidade, documentação, sem bugs críticos
  - [ ] Seção "Base teórica": Michel (2004) - Webots simulator
  - [ ] Seção "Alternativas":
    - R2024a: Muito recente, possíveis bugs
    - R2023a: Bugs conhecidos corrigidos em R2023b
  - [ ] Seção "Impacto": Ambiente reprodutível, compatibilidade API

- [ ] Criar/atualizar README.md:
  - [ ] Seção "Requisitos do Sistema"
  - [ ] Seção "Instalação do Webots" (instruções macOS + Linux)
  - [ ] Seção "Configuração do Ambiente Python"
  - [ ] Seção "Executando a Simulação"
  - [ ] Seção "Troubleshooting":
    - Python não reconhecido → configurar WEBOTS_PYTHON_PATH
    - Simulação lenta → reduzir qualidade gráfica
    - Mundo não carrega → verificar permissões de arquivo

- [ ] Criar `docs/environment.md`:
  ```markdown
  # Ambiente de Desenvolvimento

  **Data:** 2025-11-18

  ## Software
  - Webots: R2023b
  - Python: 3.x.y (registrar versão específica)
  - Sistema Operacional: macOS X.Y / Ubuntu 22.04

  ## Bibliotecas Python
  (ver requirements.txt para versões exatas)
  ```

- [ ] Commit documentação: `git add DECISIONS.md README.md docs/ && git commit -m "Document Webots setup (DECISION 005)"`

### ✅ Validação Final

- [ ] Executar pytest novamente: `pytest tests/test_webots_setup.py -v`
- [ ] Executar simulação uma última vez
- [ ] Confirmar 15 cubos spawned
- [ ] Sem erros no console
- [ ] Atualizar TODO.md:
  - [ ] Marcar Fase 1.1 como concluída: `- [x] Setup do Webots`
- [ ] Git push: `git push origin main`

---

## Critérios de Aceitação (Checklist Final)

Antes de marcar tarefa como CONCLUÍDA, validar:

- [x] Webots R2023b instalado e funcional
- [x] Python 3.8+ configurado em venv
- [x] Simulação IA_20252.wbt carrega sem erros
- [x] 15 cubos spawned corretamente
- [x] LIDAR e câmera retornam dados válidos
- [x] Testes automatizados passam (pytest)
- [x] DECISIONS.md atualizado (DECISÃO 005)
- [x] README.md com instruções de setup
- [x] docs/environment.md criado
- [x] TODO.md atualizado
- [x] Código commitado e pushed

---

**Próxima Tarefa:** Fase 1.2 - Exploração dos Controles Base
```

---

### Passo 5: Implement (`/speckit.implement`)

**Prompt Automático:**
```
Executar implementação guiada baseada em spec/webots-setup/tasks.md.

Seguir checklist passo a passo, perguntando antes de cada ação significativa.
```

**Interação Esperada:**

```
🤖 SpecKit Agent: Iniciando implementação de Webots Setup.

Primeira tarefa: Instalação do Webots

Vou executar:
1. Verificar se Webots já está instalado
2. Se não, fornecer instruções de download

Prosseguir? [s/n]

→ Você: s

🤖 Agent: Executando `which webots`...

Output: /Applications/Webots.app/Contents/MacOS/webots

Webots já instalado! Verificando versão...

🤖 Agent: `webots --version`

Output: Webots R2023b

✅ Webots R2023b confirmado.

Próxima tarefa: Configuração do Python

Vou executar:
1. `python3 --version`
2. Criar venv se não existir
3. Instalar requirements.txt

Prosseguir? [s/n]

→ Você: s

[... e assim por diante para cada tarefa do checklist ...]

🤖 Agent: Todas tarefas concluídas!

Executando validação final...
✅ pytest tests/test_webots_setup.py: 4/4 passed
✅ Simulação executada com sucesso
✅ DECISIONS.md atualizado
✅ README.md criado

Tarefa "Webots Setup" CONCLUÍDA.

Arquivo de resumo salvo em: spec/webots-setup/implementation_log.md
```

---

## Estrutura de Arquivos Gerada

Após completar o workflow SpecKit, teremos:

```
projeto-final-ia/
├── spec/
│   └── webots-setup/
│       ├── spec.md              # Especificação (Passo 1)
│       ├── clarifications.md    # Perguntas e respostas (Passo 2)
│       ├── plan.md              # Plano detalhado (Passo 3)
│       ├── tasks.md             # Checklist (Passo 4)
│       └── implementation_log.md # Log da execução (Passo 5)
├── tests/
│   ├── __init__.py
│   └── test_webots_setup.py     # Criado durante implement
├── docs/
│   └── environment.md           # Criado durante implement
├── DECISIONS.md                 # Atualizado: DECISÃO 005
├── README.md                    # Atualizado: Setup instructions
├── TODO.md                      # Atualizado: Fase 1.1 ✅
└── .gitignore                   # Já existe
```

---

## Como Executar Este Ticket

### Comando Inicial (você digita):
```bash
/speckit.specify

# Prompt:
# Criar especificação para Setup do Webots conforme Fase 1.1 do TODO.md.
# Contexto: Primeira tarefa técnica, validar ambiente Webots funcional.
# Consultar TODO.md, CLAUDE.md, constitution.
# Gerar spec/webots-setup/spec.md
```

### Depois (sequencialmente):
```bash
/speckit.clarify     # Analisa spec.md, faz perguntas
/speckit.plan        # Gera plan.md baseado em spec.md atualizado
/speckit.tasks       # Gera tasks.md baseado em plan.md
/speckit.implement   # Executa tasks.md de forma guiada
```

### Ao Final:
```bash
# Validar tudo funcionando
pytest tests/test_webots_setup.py -v

# Marcar como concluído no TODO.md
# Commit e push
git add .
git commit -m "feat(setup): Complete Webots R2023b setup and validation

- Install and configure Webots R2023b
- Setup Python 3.8+ venv
- Validate simulation loads correctly
- Add automated setup tests
- Document environment (DECISION 005)

Refs: spec/webots-setup/, TODO.md Phase 1.1"

git push origin main
```

---

## Próximos Tickets

Após concluir este ticket, seguir para:

**Ticket 2:** Fase 1.2 - Exploração dos Controles Base
**Ticket 3:** Fase 1.3 - Análise dos Sensores
**Ticket 4:** Fase 1.4 - Mapeamento da Arena

Cada um seguirá o mesmo workflow SpecKit.

---

**Nota:** Este é um exemplo completo. Na prática, o SpecKit gerará os documentos automaticamente conforme você executa os comandos. Este arquivo serve como referência de como será o fluxo.
