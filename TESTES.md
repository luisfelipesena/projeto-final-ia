# Guia de Testes no Webots

## Pré-requisitos

Antes de começar, verifique:

```bash
# 1. Dependências Python instaladas
pip install numpy scipy scikit-fuzzy torch opencv-python

# 2. Testes unitários passando
python -m pytest tests/ -v
# Esperado: 16 passed, 31 skipped
```

---

## TESTE 1: Inicialização Básica

### Objetivo
Verificar se o sistema inicializa corretamente no Webots.

### Passos

1. **Abrir Webots**
   ```bash
   open -a Webots  # macOS
   ```

2. **Carregar World**
   - Menu: `File → Open World...`
   - Navegar: `IA_20252/worlds/IA_20252.wbt`

3. **Executar**
   - Clicar Play (▶)

### O que observar no Console

**SUCESSO:**
```
[youbot] Starting Autonomous YouBot Controller
[youbot] MATA64 Final Project - Cube Collection Task
[youbot] GPS Disabled - Odometry Navigation Only

[youbot] Initializing MainController...
Initializing Main Controller...
  Perception system initialized
  Fuzzy controller initialized (25 rules)
  State machine initialized
  Navigation initialized
  Manipulation initialized
Main Controller ready!
[youbot] MainController ready - starting autonomous operation
```

**FALHA - Erro de Import:**
```
[youbot] ERROR: Failed to import MainController: ...
[youbot] Falling back to basic sensor test mode...
```
→ Copiar mensagem de erro completa

**FALHA - Outro erro:**
→ Copiar todo o traceback

### O que me enviar
- [ ] Screenshot do console com mensagens de inicialização
- [ ] Se erro: copiar texto completo do erro

---

## TESTE 2: Verificar 15 Cubos

### Objetivo
Confirmar que o Supervisor spawna os 15 cubos corretamente.

### Passos

1. Após inicialização, olhar a arena
2. Contar cubos visíveis (verde + azul + vermelho)

### O que observar

**SUCESSO:**
- 15 cubos espalhados pela arena
- Cores variadas (não todos da mesma cor)
- Cubos em posições aleatórias

**FALHA:**
- Menos de 15 cubos
- Nenhum cubo visível
- Cubos empilhados/sobrepostos

### O que me enviar
- [ ] Screenshot da arena mostrando cubos
- [ ] Contagem aproximada por cor: Verde___ Azul___ Vermelho___

---

## TESTE 3: Movimento do Robô

### Objetivo
Verificar se o robô está se movendo na arena.

### Passos

1. Observar robô por 30 segundos
2. Verificar se há movimento

### O que observar

**SUCESSO:**
- Robô girando/movendo
- Console mostra: `State: SEARCHING`
- Rodas girando

**FALHA:**
- Robô parado
- Nenhuma mensagem de estado
- Erro no console

### O que me enviar
- [ ] Robô se move? SIM / NÃO
- [ ] Console mostra estados? SIM / NÃO
- [ ] Se parado: copiar últimas mensagens do console

---

## TESTE 4: Detecção de Cubo

### Objetivo
Verificar se o sistema detecta cubos via câmera/LIDAR.

### Passos

1. Esperar robô se aproximar de um cubo
2. Observar console para mensagens de detecção

### O que observar no Console

**SUCESSO:**
```
[INFO] Cube detected: GREEN at 1.5m, 30°
[INFO] State transition: SEARCHING → APPROACHING
```

**FALHA:**
- Robô passa perto de cubo sem detectar
- Nenhuma mensagem de detecção
- Cor identificada errada

### O que me enviar
- [ ] Screenshot do console mostrando detecção
- [ ] Cores detectadas corretamente? SIM / NÃO / PARCIAL
- [ ] Se não detecta: distância aproximada do cubo quando passou

---

## TESTE 5: Ciclo Completo (Grasping)

### Objetivo
Verificar ciclo completo: detectar → aproximar → pegar → depositar.

### Passos

1. Esperar robô completar um ciclo
2. Pode demorar 1-3 minutos

### O que observar

**SUCESSO:**
```
[INFO] State: APPROACHING
[INFO] State: GRASPING
[INFO] Cube grasped successfully
[INFO] State: NAVIGATING_TO_BOX
[INFO] State: DEPOSITING
[INFO] Cube deposited in GREEN box
[INFO] State: SEARCHING
[INFO] Cubes collected: 1/15
```

**FALHA PARCIAL:**
- Pega cubo mas não deposita
- Deposita na caixa errada
- Fica preso em algum estado

**FALHA TOTAL:**
- Não consegue pegar cubo
- Colide com obstáculo
- Sistema trava

### O que me enviar
- [ ] Ciclo completo? SIM / PARCIAL / NÃO
- [ ] Se parcial: em qual estado parou?
- [ ] Screenshots do console mostrando transições
- [ ] Cubo depositado na caixa correta? SIM / NÃO

---

## TESTE 6: Desvio de Obstáculos

### Objetivo
Verificar se robô desvia de caixotes de madeira.

### Passos

1. Observar quando robô se aproxima de obstáculo
2. Verificar se entra em estado AVOIDING

### O que observar

**SUCESSO:**
```
[WARNING] Obstacle detected at 0.25m
[INFO] State: AVOIDING
[INFO] Turning away from obstacle
[INFO] State: SEARCHING (clear)
```

**FALHA:**
- Robô colide com obstáculo
- Não entra em AVOIDING
- Fica preso em loop

### O que me enviar
- [ ] Desvia de obstáculos? SIM / NÃO
- [ ] Houve colisão? SIM / NÃO
- [ ] Screenshot se houver problema

---

## TESTE 7: Performance (Tempo)

### Objetivo
Medir métricas de performance.

### Passos

1. Deixar rodar por 5 minutos
2. Anotar métricas

### Métricas para coletar

| Métrica | Valor |
|---------|-------|
| Tempo decorrido | ___ min |
| Cubos coletados | ___/15 |
| Colisões observadas | ___ |
| Estados travados | SIM/NÃO |

### O que me enviar
- [ ] Tabela preenchida acima
- [ ] Screenshot final do console
- [ ] Se travou: em qual estado e após quanto tempo

---

## Resumo do que me enviar

Após os testes, me envie:

### Obrigatório
1. **Screenshot console inicialização** (Teste 1)
2. **Screenshot arena com cubos** (Teste 2)
3. **Robô se move?** SIM/NÃO (Teste 3)
4. **Detecta cubos?** SIM/NÃO (Teste 4)
5. **Ciclo completo?** SIM/PARCIAL/NÃO (Teste 5)

### Se houver problemas
6. **Texto completo de erros/tracebacks**
7. **Estado onde travou**
8. **Comportamento inesperado descrito**

### Opcional (para métricas)
9. **Tabela de performance** (Teste 7)
10. **Screenshots de detecção/grasping**

---

## Troubleshooting Rápido

### Erro: "No module named 'controller'"
→ Está rodando fora do Webots. Execute DENTRO do Webots.

### Erro: "No module named 'skfuzzy'"
```bash
pip install scikit-fuzzy
```

### Erro: "No module named 'torch'"
```bash
pip install torch
```

### Robô não se move
1. Verificar se simulação está rodando (não pausada)
2. Verificar se há erros no console
3. Recarregar world: `Ctrl+Shift+L`

### Cubos não aparecem
1. Verificar se Supervisor está habilitado
2. Recarregar world

### Performance muito lenta
1. `View → Rendering` → desativar efeitos
2. Reduzir qualidade de textura
