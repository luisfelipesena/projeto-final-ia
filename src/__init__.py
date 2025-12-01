"""
YouBot Autonomous Controller - MATA64 UFBA
==========================================

Sistema autônomo de coleta e organização de cubos coloridos.

Requisitos MATA64:
- RNA (MLP/CNN) para detecção de obstáculos via LIDAR
- Lógica Fuzzy para controle de navegação e manipulação
- Sem GPS - apenas LIDAR + Câmera RGB

Autor: Luis Felipe Cordeiro Sena
Disciplina: MATA64 - Inteligência Artificial
Professor: Luciano Oliveira
Semestre: 2025.2
"""

from .main_controller import YouBotController, main

__version__ = "1.0.0"
__author__ = "Luis Felipe Cordeiro Sena"
__all__ = ['YouBotController', 'main']
