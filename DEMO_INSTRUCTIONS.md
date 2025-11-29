# Instruções para Gravar Vídeo do Projeto Final

## Requisitos do Vídeo (Final Project.pdf)

| Requisito | Especificação |
|-----------|---------------|
| **Duração** | Máximo 15 minutos |
| **Conteúdo** | Explicar desenvolvimento conceitual + demonstração |
| **PROIBIDO** | Mostrar código-fonte (desconto 3-10 pontos) |
| **Foco** | Figuras, diagramas, processos - MÍNIMO texto |
| **Submissão** | Link YouTube + código desenvolvido |

---

## Estrutura do Vídeo (15 min)

### Parte 1: Slides (12 min)
Usar `slides-template/main.tex` compilado

| Tempo | Slide | Conteúdo |
|-------|-------|----------|
| 0:00-0:30 | 1-2 | Título + Agenda |
| 0:30-1:30 | 3-4 | O Problema + Sensores |
| 1:30-3:00 | 5-6 | Pipeline + Máquina de Estados |
| 3:00-5:00 | 7-9 | RNA LIDAR + Câmera + HSV |
| 5:00-7:00 | 10-13 | Fuzzy Controller (variáveis, MFs, regras, inferência) |
| 7:00-8:00 | 14 | Métricas de Performance |
| 8:00-9:00 | 15 | Demonstração (transição) |
| 9:00-10:00 | 16-17 | Lições + Referências |

### Parte 2: Demo ao Vivo ou Gravada (3 min)
| Tempo | Ação |
|-------|------|
| 10:00-10:30 | Mostrar arena com 15 cubos |
| 10:30-11:00 | Robô buscando (estado SEARCHING) |
| 11:00-11:30 | Detecção de cubo (mostrar console) |
| 11:30-12:00 | Aproximação + Grasping |
| 12:00-12:30 | Navegação até caixa correta |
| 12:30-13:00 | Depósito + voltar a buscar |
| 13:00-14:00 | Mostrar ciclo completo de mais cubos |
| 14:00-14:30 | Estatísticas finais (X/15 cubos) |
| 14:30-15:00 | Encerramento |

---

## O Que Gravar

### Screenshots Necessários (antes de gravar vídeo)
```
slides-template/imgs/
├── arena_screenshot.png      # Visão geral da arena
├── cube_detection.png        # Robô detectando cubo
├── robot_grasp.png           # Momento do grasping
└── hsv_pipeline.png          # Segmentação HSV (opcional)
```

### Vídeo da Demo
- **Resolução:** 1920x1080
- **FPS:** 30
- **Duração demo:** 3-5 minutos
- **Áudio:** Narração explicando o que acontece

---

## Como Gravar

### Opção A: Gravação Webots + OBS
1. Abrir OBS Studio
2. Capturar janela Webots + microfone
3. Gravar slides primeiro (compartilhar tela do PDF)
4. Depois gravar demo no Webots

### Opção B: Webots Movie + Edição
1. Webots: `File → Make Movie...`
2. Gravar demo completa
3. Editar com slides usando software de vídeo

---

## Checklist Antes de Gravar

### Verificações Técnicas
- [ ] GPS **DESABILITADO** no world file
- [ ] Supervisor funcionando (15 cubos aparecem)
- [ ] Console mostra inicialização OK
- [ ] Robô se move e detecta cubos
- [ ] Sem erros de import

### Verificações de Conteúdo
- [ ] Slides compilados (`pdflatex main.tex`)
- [ ] Todas figuras inseridas nos slides
- [ ] Script de fala revisado (`falas.txt`)
- [ ] Métricas preenchidas na tabela

### Verificações de Requisitos
- [ ] **ZERO código-fonte** visível em qualquer momento
- [ ] Citações científicas mencionadas (Zadeh, Mamdani, etc.)
- [ ] Demonstração mostra ciclo completo
- [ ] Tempo total ≤ 15 minutos

---

## Pontos Importantes na Narração

### O que DEVE explicar:
1. **RNA:** "Usamos CNN 1D para processar 667 pontos LIDAR em 8 setores"
2. **Fuzzy:** "25 regras Mamdani com defuzzificação por centróide"
3. **Cores:** "Classificação RGB com fallback HSV calibrado"
4. **Estados:** "Máquina de 7 estados com AVOIDING prioritário"
5. **GPS:** "Navegação apenas com odometria - GPS desabilitado"

### O que NÃO fazer:
- ❌ Mostrar arquivos .py
- ❌ Mostrar terminal com código
- ❌ Ler código em voz alta
- ❌ Ultrapassar 15 minutos

---

## Métricas para Reportar

Preencher durante demo e mostrar no slide 14:

| Métrica | Meta | Resultado |
|---------|------|-----------|
| Cubos coletados | 15/15 | ___/15 |
| Precisão de cor | >95% | ___% |
| Colisões | 0 | ___ |
| Tempo total | <5 min | ___ min |
| FPS controle | >10 Hz | ___ Hz |

---

## Entrega Final

### Arquivos para Submeter
1. **Link YouTube** do vídeo (unlisted ou public)
2. **Código fonte** (zip do repositório sem .git)

### Prazo
**06/01/2026, 23:59**

---

## Troubleshooting

### Robô não funciona:
```bash
# Verificar dependências
pip install numpy scipy scikit-fuzzy torch opencv-python
```

### Cubos não aparecem:
- Verificar Supervisor habilitado no world
- Recarregar world: `Ctrl+Shift+L`

### Performance lenta:
- Desativar ray tracing: `View → Rendering`
- Reduzir qualidade de gravação
