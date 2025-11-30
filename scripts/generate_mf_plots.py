"""
Generate Membership Function Plots for Presentation Slides

Creates publication-quality figures for fuzzy controller MFs.
Output: slides-template/imgs/mf_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for LaTeX-quality output
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = Path(__file__).parent.parent / 'slides-template' / 'imgs'


def trimf(x, params):
    """Triangular membership function"""
    a, b, c = params
    y = np.zeros_like(x)
    # Rising edge
    mask_rise = (x >= a) & (x <= b) & (b != a)
    y[mask_rise] = (x[mask_rise] - a) / (b - a)
    # Falling edge
    mask_fall = (x > b) & (x <= c) & (c != b)
    y[mask_fall] = (c - x[mask_fall]) / (c - b)
    # Peak
    y[x == b] = 1.0
    return y


def trapmf(x, params):
    """Trapezoidal membership function"""
    a, b, c, d = params
    y = np.zeros_like(x)
    # Rising edge
    if b != a:
        mask = (x >= a) & (x < b)
        y[mask] = (x[mask] - a) / (b - a)
    # Flat top
    mask = (x >= b) & (x <= c)
    y[mask] = 1.0
    # Falling edge
    if d != c:
        mask = (x > c) & (x <= d)
        y[mask] = (d - x[mask]) / (d - c)
    return y


def plot_distance_to_obstacle():
    """Plot MFs for distance_to_obstacle [0, 5] meters"""
    x = np.linspace(0, 5, 500)

    fig, ax = plt.subplots(figsize=(10, 5))

    mfs = {
        'Muito Perto': ('trimf', (0.0, 0.3, 0.6), '#e74c3c'),
        'Perto': ('trimf', (0.4, 0.8, 1.2), '#f39c12'),
        'Médio': ('trimf', (1.0, 1.8, 2.6), '#2ecc71'),
        'Longe': ('trimf', (2.2, 3.5, 4.3), '#3498db'),
        'Muito Longe': ('trapmf', (4.0, 5.0, 5.0, 5.0), '#9b59b6'),
    }

    for label, (mf_type, params, color) in mfs.items():
        if mf_type == 'trimf':
            y = trimf(x, params)
        else:
            y = trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_xlabel('Distância ao Obstáculo (m)')
    ax.set_ylabel('Grau de Pertinência μ(x)')
    ax.set_title('Funções de Pertinência - Distância ao Obstáculo')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_distance_obstacle.png')
    plt.close()
    print(f"Saved: mf_distance_obstacle.png")


def plot_angle_to_obstacle():
    """Plot MFs for angle_to_obstacle [-135, 135] degrees"""
    x = np.linspace(-135, 135, 500)

    fig, ax = plt.subplots(figsize=(12, 5))

    mfs = {
        'Neg. Grande': ('trapmf', (-135, -135, -90, -45), '#e74c3c'),
        'Neg. Médio': ('trimf', (-90, -60, -30), '#f39c12'),
        'Neg. Pequeno': ('trimf', (-45, -15, 0), '#f1c40f'),
        'Zero': ('trimf', (-15, 0, 15), '#2ecc71'),
        'Pos. Pequeno': ('trimf', (0, 15, 45), '#1abc9c'),
        'Pos. Médio': ('trimf', (30, 60, 90), '#3498db'),
        'Pos. Grande': ('trapmf', (45, 90, 135, 135), '#9b59b6'),
    }

    for label, (mf_type, params, color) in mfs.items():
        if mf_type == 'trimf':
            y = trimf(x, params)
        else:
            y = trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.12, color=color)

    ax.set_xlabel('Ângulo ao Obstáculo (°)')
    ax.set_ylabel('Grau de Pertinência μ(x)')
    ax.set_title('Funções de Pertinência - Ângulo ao Obstáculo')
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-135, 135)
    ax.set_ylim(0, 1.1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_angle_obstacle.png')
    plt.close()
    print(f"Saved: mf_angle_obstacle.png")


def plot_linear_velocity():
    """Plot MFs for linear_velocity [0, 0.3] m/s"""
    x = np.linspace(0, 0.3, 500)

    fig, ax = plt.subplots(figsize=(10, 5))

    mfs = {
        'Parar': ('trimf', (0.0, 0.0, 0.05), '#e74c3c'),
        'Devagar': ('trimf', (0.03, 0.08, 0.13), '#f39c12'),
        'Médio': ('trimf', (0.10, 0.18, 0.25), '#2ecc71'),
        'Rápido': ('trapmf', (0.20, 0.30, 0.30, 0.30), '#3498db'),
    }

    for label, (mf_type, params, color) in mfs.items():
        if mf_type == 'trimf':
            y = trimf(x, params)
        else:
            y = trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_xlabel('Velocidade Linear (m/s)')
    ax.set_ylabel('Grau de Pertinência μ(x)')
    ax.set_title('Funções de Pertinência - Velocidade Linear (Saída)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_linear_velocity.png')
    plt.close()
    print(f"Saved: mf_linear_velocity.png")


def plot_angular_velocity():
    """Plot MFs for angular_velocity [-0.5, 0.5] rad/s"""
    x = np.linspace(-0.5, 0.5, 500)

    fig, ax = plt.subplots(figsize=(10, 5))

    mfs = {
        'Girar Esq. Forte': ('trapmf', (-0.5, -0.5, -0.3, -0.15), '#e74c3c'),
        'Girar Esquerda': ('trimf', (-0.3, -0.15, 0.0), '#f39c12'),
        'Reto': ('trimf', (-0.1, 0.0, 0.1), '#2ecc71'),
        'Girar Direita': ('trimf', (0.0, 0.15, 0.3), '#3498db'),
        'Girar Dir. Forte': ('trapmf', (0.15, 0.3, 0.5, 0.5), '#9b59b6'),
    }

    for label, (mf_type, params, color) in mfs.items():
        if mf_type == 'trimf':
            y = trimf(x, params)
        else:
            y = trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_xlabel('Velocidade Angular (rad/s)')
    ax.set_ylabel('Grau de Pertinência μ(x)')
    ax.set_title('Funções de Pertinência - Velocidade Angular (Saída)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_angular_velocity.png')
    plt.close()
    print(f"Saved: mf_angular_velocity.png")


def plot_combined_inputs():
    """Combined plot with distance and angle MFs side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distance to obstacle
    ax = axes[0]
    x = np.linspace(0, 5, 500)
    mfs = {
        'Muito Perto': ('trimf', (0.0, 0.3, 0.6), '#e74c3c'),
        'Perto': ('trimf', (0.4, 0.8, 1.2), '#f39c12'),
        'Médio': ('trimf', (1.0, 1.8, 2.6), '#2ecc71'),
        'Longe': ('trimf', (2.2, 3.5, 4.3), '#3498db'),
        'Muito Longe': ('trapmf', (4.0, 5.0, 5.0, 5.0), '#9b59b6'),
    }
    for label, (mf_type, params, color) in mfs.items():
        y = trimf(x, params) if mf_type == 'trimf' else trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2, color=color)
        ax.fill_between(x, y, alpha=0.12, color=color)
    ax.set_xlabel('Distância (m)')
    ax.set_ylabel('μ(x)')
    ax.set_title('(a) Distância ao Obstáculo')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.1)

    # Angle to obstacle
    ax = axes[1]
    x = np.linspace(-135, 135, 500)
    mfs = {
        'NB': ('trapmf', (-135, -135, -90, -45), '#e74c3c'),
        'NM': ('trimf', (-90, -60, -30), '#f39c12'),
        'NP': ('trimf', (-45, -15, 0), '#f1c40f'),
        'Zero': ('trimf', (-15, 0, 15), '#2ecc71'),
        'PP': ('trimf', (0, 15, 45), '#1abc9c'),
        'PM': ('trimf', (30, 60, 90), '#3498db'),
        'PB': ('trapmf', (45, 90, 135, 135), '#9b59b6'),
    }
    for label, (mf_type, params, color) in mfs.items():
        y = trimf(x, params) if mf_type == 'trimf' else trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2, color=color)
        ax.fill_between(x, y, alpha=0.12, color=color)
    ax.set_xlabel('Ângulo (°)')
    ax.set_ylabel('μ(x)')
    ax.set_title('(b) Ângulo ao Obstáculo')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-135, 135)
    ax.set_ylim(0, 1.1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_inputs_combined.png')
    plt.close()
    print(f"Saved: mf_inputs_combined.png")


def plot_combined_outputs():
    """Combined plot with velocity MFs side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear velocity
    ax = axes[0]
    x = np.linspace(0, 0.3, 500)
    mfs = {
        'Parar': ('trimf', (0.0, 0.0, 0.05), '#e74c3c'),
        'Devagar': ('trimf', (0.03, 0.08, 0.13), '#f39c12'),
        'Médio': ('trimf', (0.10, 0.18, 0.25), '#2ecc71'),
        'Rápido': ('trapmf', (0.20, 0.30, 0.30, 0.30), '#3498db'),
    }
    for label, (mf_type, params, color) in mfs.items():
        y = trimf(x, params) if mf_type == 'trimf' else trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)
    ax.set_xlabel('Velocidade Linear (m/s)')
    ax.set_ylabel('μ(x)')
    ax.set_title('(a) Velocidade Linear (Saída)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.1)

    # Angular velocity
    ax = axes[1]
    x = np.linspace(-0.5, 0.5, 500)
    mfs = {
        'Esq. Forte': ('trapmf', (-0.5, -0.5, -0.3, -0.15), '#e74c3c'),
        'Esquerda': ('trimf', (-0.3, -0.15, 0.0), '#f39c12'),
        'Reto': ('trimf', (-0.1, 0.0, 0.1), '#2ecc71'),
        'Direita': ('trimf', (0.0, 0.15, 0.3), '#3498db'),
        'Dir. Forte': ('trapmf', (0.15, 0.3, 0.5, 0.5), '#9b59b6'),
    }
    for label, (mf_type, params, color) in mfs.items():
        y = trimf(x, params) if mf_type == 'trimf' else trapmf(x, params)
        ax.plot(x, y, label=label, linewidth=2.5, color=color)
        ax.fill_between(x, y, alpha=0.15, color=color)
    ax.set_xlabel('Velocidade Angular (rad/s)')
    ax.set_ylabel('μ(x)')
    ax.set_title('(b) Velocidade Angular (Saída)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mf_outputs_combined.png')
    plt.close()
    print(f"Saved: mf_outputs_combined.png")


if __name__ == '__main__':
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Membership Function Plots...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    plot_distance_to_obstacle()
    plot_angle_to_obstacle()
    plot_linear_velocity()
    plot_angular_velocity()
    plot_combined_inputs()
    plot_combined_outputs()

    print()
    print("All plots generated successfully!")
