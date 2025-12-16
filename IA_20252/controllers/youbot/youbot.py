"""
YouBot Controller Entry Point.
Minimal entry point that imports and runs the main controller.
"""

from youbot_controller import YouBotController


if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
