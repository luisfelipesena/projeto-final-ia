# Feature Specification: Phase 2 Dataset Script Enhancements

**Feature Branch**: `002-script-updates`  
**Created**: 2025-11-21  
**Status**: Draft  
**Input**: User description: "Continuar com T011-T017 (melhorias nos scripts) sem Webots @TODO.md @DECISIONS.md"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Instrumentar scripts de captura (Priority: P1)

Como engenheiro de percepção, posso executar os scripts de coleta de LIDAR e câmera e receber arquivos JSON enriquecidos com pose do robô, tags de cenário, rotulagem setorial e metadados de iluminação sem precisar rodar o Webots agora, garantindo que quando a coleta real acontecer os dados já terão todos os campos exigidos pelos requisitos FR-001—FR-007.

**Why this priority**: Esses campos são pré-requisito direto para gerar datasets treináveis (DECISÃO 016-017). Sem eles, a coleta manual planejada para T018-T019 não produz dados utilizáveis.

**Independent Test**: Executar os scripts contra arquivos de amostra (mock scans e frames) e verificar, via testes automatizados, que o JSON por registro contém pose `{x,y,theta}`, `scenario_tag`, `sector_labels[9]`, tags de iluminação e blobs de anotação.

**Acceptance Scenarios**:

1. **Given** um arquivo de ranges brutos e a pose mockada, **When** `scripts/collect_lidar_data.py` é executado com o novo parâmetro `--pose-log pose.json`, **Then** cada entrada exportada inclui `robot_pose`, `scenario_tag` e `sector_labels` preenchidos automaticamente.
2. **Given** um conjunto de frames RGB e um arquivo `lighting_profile.json`, **When** `scripts/collect_camera_data.py` roda com `--metadata lighting_profile.json`, **Then** o output JSON contém `robot_pose`, `lighting_tag` e placeholders de bounding boxes compatíveis com o pipeline de anotação.

---

### User Story 2 - Automatizar manifestos e splits (Priority: P2)

Como responsável por curadoria de dados, posso gerar manifestos consistentes e dividir automaticamente o dataset em train/val/test balanceados sem depender do simulador, para que a futura coleta apenas alimente o pipeline pronto e documentado.

**Why this priority**: Os manifestos e splits são exigidos pelos requisitos FR-003 e FR-004 do plano da Fase 2. Sem automação, a geração manual seria lenta e sujeita a erros quando os dados reais chegarem.

**Independent Test**: Usar um diretório de amostras mockadas, rodar `scripts/generate_dataset_manifest.py` seguido de `scripts/split_dataset.py`, e validar que os arquivos de saída contêm hashes, contagens por classe/setor e splits estritamente mutuamente exclusivos.

**Acceptance Scenarios**:

1. **Given** um diretório `data/lidar/mock/` com 60 amostras e metadados, **When** `scripts/generate_dataset_manifest.py --input data/lidar/mock --output data/lidar/dataset_manifest.json` é executado, **Then** o manifesto contém lista de `samples`, hash SHA256 e contagens por `scenario_tag`.
2. **Given** um manifesto JSON válido, **When** `scripts/split_dataset.py --manifest dataset_manifest.json --strategy balanced` roda, **Then** cada sample_id aparece em apenas um split e o desvio percentual por setor/cor fica abaixo de 5%.

---

### User Story 3 - [Brief Title] (Priority: P3)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

### Edge Cases

- Amostras sem pose registrada: scripts devem inserir `robot_pose` com `null` e registrar no manifesto para revisão.
- Arquivos duplicados (mesmo `sample_id`) detectados ao gerar manifesto: processo deve abortar com mensagem clara.
- Splits com poucas amostras por cor ou setor: ferramenta deve sugerir estratégia `stratified` e impedir saída inconsistente.
- Inputs externos corrompidos (JSON inválido): scripts precisam falhar rápido e registrar erro no console.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: `scripts/collect_lidar_data.py` deve aceitar arquivos de pose/log e enriquecer cada amostra com `robot_pose`, `timestamp` ISO8601 e `scenario_tag` configurável.
- **FR-002**: `scripts/annotate_lidar.py` deve gerar `sector_labels[9]` automaticamente a partir dos ranges, usando thresholds definidos em DECISÃO 016, e inserir o array no JSON final.
- **FR-003**: `scripts/collect_camera_data.py` deve registrar `robot_pose`, `lighting_tag` e placeholders de bounding boxes para cada frame coletado.
- **FR-004**: `scripts/annotate_camera.py` deve suportar pipeline HSV → bounding box → cor → distância, produzindo JSON compatível com o data model (CameraSample).
- **FR-005**: `scripts/generate_dataset_manifest.py` deve produzir manifesto JSON com hash, contagens por classe/setor, caminhos dos arquivos e validação de schema.
- **FR-006**: `scripts/split_dataset.py` deve criar splits `train/val/test` estritamente mutuamente exclusivos mantendo desvio ≤5% por classe/setor e salvar artefatos individuais.
- **FR-007**: Todos os scripts devem funcionar com dados mockados (sem Webots), usando entradas exemplo fornecidas em `data/mock/`, permitindo testes locais automatizados.

### Key Entities *(include if feature involves data)*

- **LidarSampleMock**: versão sintética com campos `ranges`, `robot_pose`, `scenario_tag`, `sector_labels` (após enriquecimento).
- **CameraSampleMock**: objeto com `image_path`, `robot_pose`, `lighting_tag`, `bounding_boxes` artificiais e `distance_estimates`.
- **DatasetManifest**: estrutura JSON contendo `samples`, `hash`, `counts`, `splits` e metadados gerais.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Scripts de coleta devem gerar JSON com 100% dos campos obrigatórios (`robot_pose`, `scenario_tag`, `lighting_tag`, `sector_labels`) preenchidos quando executados com dados mock.
- **SC-002**: Manifestos gerados devem registrar 0 erros de schema e incluir contagens agregadas por classe/setor, verificadas via testes automatizados.
- **SC-003**: Splits automáticos devem manter desvio ≤5% por classe/setor, comprovado por relatório gerado pelo próprio script.
- **SC-004**: Pipeline completo (coleta ≥ anotação ≥ manifesto ≥ split) deve terminar em <2 minutos sobre o dataset mock com logs claros de sucesso.

## Assumptions & Dependencies *(optional but recommended)*

- Os dados mockados em `data/mock/` representam adequadamente a estrutura real e podem ser ajustados conforme necessário.
- O Webots não é requerido para esta feature; portanto, nenhuma métrica de tempo real de simulação será considerada.
- DECISÕES 016 e 017 definem os campos e formatos alvo, e TODO.md (T011-T017) é a referência de escopo.

