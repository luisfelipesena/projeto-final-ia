"""
Pytest configuration and fixtures for YouBot control validation tests.
"""

import pytest
import sys
import os

# Add controller path for importing YouBot modules
CONTROLLER_PATH = os.path.join(os.path.dirname(__file__), '..', 'IA_20252', 'controllers', 'youbot')
sys.path.insert(0, CONTROLLER_PATH)


@pytest.fixture(scope="module")
def robot():
    """Initialize YouBot robot for all tests in module.

    Yields:
        Robot: Webots Robot instance

    Note:
        This fixture requires Webots simulation to be running.
        Tests will fail if executed outside Webots environment.
    """
    try:
        from controller import Robot
    except ImportError:
        pytest.skip("Webots controller module not available. Tests must run inside Webots simulation.")

    robot_instance = Robot()
    time_step = int(robot_instance.getBasicTimeStep())

    # Wait for initialization
    robot_instance.step(time_step)

    yield robot_instance

    # Cleanup - stop all motion
    # (Base, arm, gripper reset handled by autouse fixture)


@pytest.fixture(scope="module")
def youbot(robot):
    """Initialize YouBot controller modules.

    Args:
        robot: Webots Robot instance from robot fixture

    Yields:
        tuple: (robot, base, arm, gripper) instances
    """
    from youbot import YouBotController

    controller = YouBotController()

    yield (controller.robot, controller.base, controller.arm, controller.gripper)

    # Cleanup
    controller.base.reset()
    controller.arm.reset()
    controller.gripper.release()
    controller.robot.step(controller.time_step)


@pytest.fixture(autouse=True)
def reset_robot(youbot):
    """Reset robot to initial state before each test.

    This fixture runs automatically before every test function.

    Args:
        youbot: tuple (robot, base, arm, gripper) from youbot fixture
    """
    robot, base, arm, gripper = youbot

    # Reset all subsystems
    base.reset()
    arm.reset()
    gripper.release()

    # Wait for reset to complete
    time_step = int(robot.getBasicTimeStep())
    for _ in range(10):  # 10 steps = ~320ms
        robot.step(time_step)

    yield

    # Teardown: ensure robot stops after test
    base.reset()
    robot.step(time_step)


@pytest.fixture(scope="session")
def velocity_limits():
    """Storage for measured velocity limits (shared across tests).

    Returns:
        dict: mutable dictionary to store velocity limit measurements
    """
    return {
        'vx_max': None,
        'vx_min': None,
        'vy_max': None,
        'vy_min': None,
        'omega_max': None,
        'omega_min': None,
    }


@pytest.fixture(scope="session")
def joint_limits():
    """Storage for measured arm joint limits (shared across tests).

    Returns:
        dict: mutable dictionary to store joint limit measurements
    """
    return {
        'joint1': (None, None),
        'joint2': (None, None),
        'joint3': (None, None),
        'joint4': (None, None),
        'joint5': (None, None),
        'gripper': (None, None),
    }
