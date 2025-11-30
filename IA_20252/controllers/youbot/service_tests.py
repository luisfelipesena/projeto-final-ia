"""
Service Tests - Testes isolados para cada serviço modular

USO:
1. Abrir Webots
2. Carregar IA_20252/worlds/IA_20252.wbt
3. No youbot.py, mudar para chamar este arquivo
4. Pausar simulação (||)
5. Posicionar cubo manualmente se necessário (para teste de ARM)
6. Rodar simulação (▶)

TESTES DISPONÍVEIS:
- arm_positions: Testa posições do braço (sem cubo)
- arm_grasp: Testa ciclo de grasp (precisa cubo posicionado)
- movement_square: Testa movimento em quadrado
- vision_tracking: Testa estabilidade de tracking
"""

import sys
from pathlib import Path

# Add paths
src_path = Path(__file__).resolve().parent.parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper


def test_arm_positions(robot, arm, gripper, time_step):
    """
    TESTE 1: Posições do braço

    Cicla pelas posições do braço para verificar se funciona.
    NÃO precisa de cubo.
    """
    print("\n" + "="*50)
    print("TESTE: ARM POSITIONS")
    print("="*50)
    print("Observar braço movendo para cada posição...")

    def wait(seconds):
        steps = int(seconds * 1000 / time_step)
        for _ in range(steps):
            if robot.step(time_step) == -1:
                return False
        return True

    positions = [
        ("RESET (tucked)", arm.RESET, arm.FRONT),
        ("FRONT_PLATE (raised)", arm.FRONT_PLATE, arm.FRONT),
        ("FRONT_FLOOR (lowered)", arm.FRONT_FLOOR, arm.FRONT),
        ("FRONT_PLATE (raised)", arm.FRONT_PLATE, arm.FRONT),
        ("RESET (tucked)", arm.RESET, arm.FRONT),
    ]

    for name, height, orientation in positions:
        print(f"\n  → Movendo para: {name}")
        arm.set_height(height)
        arm.set_orientation(orientation)
        if not wait(2.5):
            return False
        print(f"    ✓ Chegou em {name}")

    print("\n" + "="*50)
    print("TESTE ARM POSITIONS: COMPLETO")
    print("="*50)
    return True


def test_arm_grasp(robot, arm, gripper, time_step):
    """
    TESTE 2: Ciclo de grasp

    ANTES DE RODAR:
    1. Pausar simulação (||)
    2. No Webots, arrastar um cubo para ~25cm na frente do robot
    3. Rodar simulação (▶)

    Sequência:
    1. Abre gripper
    2. Move braço para frente (raised)
    3. Abaixa braço (floor level)
    4. Fecha gripper
    5. Verifica sensor (has_object?)
    6. Levanta braço
    7. Abre gripper (deposita)
    """
    print("\n" + "="*50)
    print("TESTE: ARM GRASP CYCLE")
    print("="*50)
    print("IMPORTANTE: Cubo deve estar ~25cm na frente do robot!")
    print("  Se não posicionou, pause agora e arraste um cubo.")
    print("="*50)

    def wait(seconds):
        steps = int(seconds * 1000 / time_step)
        for _ in range(steps):
            if robot.step(time_step) == -1:
                return False
        return True

    print("\n  Iniciando em 3 segundos...")
    if not wait(3.0):
        return False

    # Step 1: Open gripper
    print("\n  [1/7] Abrindo gripper...")
    gripper.release()
    if not wait(1.0):
        return False
    print("    ✓ Gripper aberto")

    # Step 2: Move arm to front raised
    print("\n  [2/7] Movendo braço para frente (raised)...")
    arm.set_orientation(arm.FRONT)
    arm.set_height(arm.FRONT_PLATE)
    if not wait(2.5):
        return False
    print("    ✓ Braço em FRONT_PLATE")

    # Step 3: Lower arm to floor
    print("\n  [3/7] Abaixando braço para o chão...")
    arm.set_height(arm.FRONT_FLOOR)
    if not wait(2.0):
        return False
    print("    ✓ Braço em FRONT_FLOOR")

    # Step 4: Close gripper
    print("\n  [4/7] Fechando gripper...")
    gripper.grip()
    if not wait(1.5):
        return False
    print("    ✓ Gripper fechado")

    # Step 5: Check sensor
    print("\n  [5/7] Verificando sensor...")
    has_object = gripper.has_object()
    print(f"    → has_object() = {has_object}")

    if has_object:
        print("    ✓✓✓ CUBO DETECTADO! Grasp funcionou!")
    else:
        print("    ✗ Nenhum objeto detectado")
        print("    Possíveis causas:")
        print("    - Cubo não estava posicionado corretamente")
        print("    - Braço não alcançou o cubo")
        print("    - Gripper não fechou no cubo")

    # Step 6: Lift arm
    print("\n  [6/7] Levantando braço...")
    arm.set_height(arm.FRONT_PLATE)
    if not wait(2.0):
        return False
    print("    ✓ Braço levantado")

    # Step 7: Release (deposit)
    print("\n  [7/7] Abrindo gripper (depositar)...")
    gripper.release()
    if not wait(1.0):
        return False
    print("    ✓ Gripper aberto")

    # Reset arm
    print("\n  Retornando braço para posição inicial...")
    arm.set_height(arm.RESET)
    if not wait(2.0):
        return False

    print("\n" + "="*50)
    if has_object:
        print("TESTE ARM GRASP: SUCESSO!")
    else:
        print("TESTE ARM GRASP: FALHOU (sem objeto)")
    print("="*50)

    return has_object


def test_movement_square(robot, base, time_step):
    """
    TESTE 3: Movimento em quadrado

    Robot move 0.5m para frente, gira 90°, repete 4x.
    Deve retornar aproximadamente ao ponto inicial.
    """
    print("\n" + "="*50)
    print("TESTE: MOVEMENT SQUARE")
    print("="*50)
    print("Robot vai andar em quadrado (0.5m x 0.5m)")

    def wait(seconds):
        steps = int(seconds * 1000 / time_step)
        for _ in range(steps):
            if robot.step(time_step) == -1:
                return False
        return True

    def move_forward(distance, speed=0.15):
        duration = distance / speed
        base.move(vx=speed, vy=0, omega=0)
        return wait(duration)

    def turn_left(angle_deg, speed=0.4):
        import math
        angle_rad = math.radians(angle_deg)
        duration = angle_rad / speed
        base.move(vx=0, vy=0, omega=speed)
        return wait(duration)

    def stop():
        base.move(vx=0, vy=0, omega=0)

    print("\n  Iniciando em 2 segundos...")
    if not wait(2.0):
        return False

    for i in range(4):
        print(f"\n  Lado {i+1}/4:")
        print(f"    → Frente 0.5m...")
        if not move_forward(0.5):
            return False
        stop()
        if not wait(0.5):
            return False

        print(f"    → Girando 90°...")
        if not turn_left(90):
            return False
        stop()
        if not wait(0.5):
            return False

    print("\n" + "="*50)
    print("TESTE MOVEMENT SQUARE: COMPLETO")
    print("Verificar: robot voltou ao ponto inicial?")
    print("="*50)
    return True


def test_vision_tracking(robot, time_step):
    """
    TESTE 4: Estabilidade de tracking de visão

    Verifica se o VisionService mantém tracking estável
    mesmo com múltiplos cubos visíveis.
    """
    print("\n" + "="*50)
    print("TESTE: VISION TRACKING")
    print("="*50)

    try:
        from services.vision_service import VisionService
        from perception.cube_detector import CubeDetector
        import numpy as np
    except ImportError as e:
        print(f"  ERRO: Não foi possível importar VisionService: {e}")
        return False

    camera = robot.getDevice("camera")
    camera.enable(time_step)

    detector = CubeDetector()
    vision = VisionService(detector, time_step)

    print("  Rodando por 100 frames...")
    print("  Observar: track_id deve permanecer constante")

    initial_track_id = None
    switches = 0

    for i in range(100):
        if robot.step(time_step) == -1:
            return False

        # Get camera image
        image = camera.getImage()
        if image:
            width = camera.getWidth()
            height = camera.getHeight()
            image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            image_rgb = image_array[:, :, :3]

            vision.update(image_rgb)

        target = vision.get_target()

        if target:
            if initial_track_id is None:
                initial_track_id = target.track_id
                print(f"\n  Frame {i}: Primeiro target: {target.color} (id={target.track_id})")
            elif target.track_id != initial_track_id:
                switches += 1
                print(f"  Frame {i}: SWITCH! {initial_track_id} → {target.track_id}")
                initial_track_id = target.track_id

        # Log every 20 frames
        if (i + 1) % 20 == 0:
            if target:
                print(f"  Frame {i+1}: {target.color} id={target.track_id} "
                      f"dist={target.distance:.2f}m angle={target.angle:.1f}°")
            else:
                print(f"  Frame {i+1}: Sem target")

    print("\n" + "="*50)
    print(f"TESTE VISION TRACKING: {switches} switches")
    if switches == 0:
        print("  ✓ ESTÁVEL - Tracking não oscilou")
    else:
        print("  ✗ INSTÁVEL - Tracking oscilou entre alvos")
    print("="*50)

    return switches == 0


def run_all_tests():
    """Menu de testes"""
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    base = Base(robot)
    arm = Arm(robot)
    gripper = Gripper(robot)

    print("\n" + "="*60)
    print("  SERVICE TESTS - YouBot Modular Architecture")
    print("  DECISÃO 028: Testando serviços isoladamente")
    print("="*60)

    # Warmup
    print("\nWarmup (10 frames)...")
    for _ in range(10):
        robot.step(time_step)

    # Menu
    print("\nTestes disponíveis:")
    print("  1. ARM POSITIONS - Testa movimentos do braço (sem cubo)")
    print("  2. ARM GRASP - Testa ciclo de grasp (PRECISA cubo posicionado)")
    print("  3. MOVEMENT - Testa movimento em quadrado")
    print("  4. VISION - Testa estabilidade de tracking")
    print("  5. TODOS - Roda todos os testes")

    # Por padrão, roda ARM POSITIONS que não precisa de setup
    print("\n→ Rodando teste padrão: ARM POSITIONS")
    print("  (Para outros testes, edite este arquivo)\n")

    # MUDE AQUI para rodar teste diferente:
    TEST_TO_RUN = "arm_positions"  # Opções: arm_positions, arm_grasp, movement, vision

    if TEST_TO_RUN == "arm_positions":
        test_arm_positions(robot, arm, gripper, time_step)
    elif TEST_TO_RUN == "arm_grasp":
        test_arm_grasp(robot, arm, gripper, time_step)
    elif TEST_TO_RUN == "movement":
        test_movement_square(robot, base, time_step)
    elif TEST_TO_RUN == "vision":
        test_vision_tracking(robot, time_step)
    elif TEST_TO_RUN == "all":
        test_arm_positions(robot, arm, gripper, time_step)
        test_movement_square(robot, base, time_step)
        test_vision_tracking(robot, time_step)
        # arm_grasp requer setup manual, não roda automático

    # Keep simulation running
    print("\nTeste finalizado. Simulação continua rodando...")
    while robot.step(time_step) != -1:
        pass


if __name__ == "__main__":
    run_all_tests()
