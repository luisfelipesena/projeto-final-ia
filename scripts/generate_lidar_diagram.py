"""
Generate LIDAR 9-Sector Diagram for Presentation Slides

Shows the 9-sector division around the YouBot for obstacle detection.
Output: slides-template/imgs/lidar_sectors.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'slides-template' / 'imgs'


def draw_lidar_sectors():
    """Create diagram showing 9-sector LIDAR coverage"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Colors for sectors
    sector_colors = [
        '#e74c3c',  # 0: Front-Right
        '#f39c12',  # 1: Front
        '#f1c40f',  # 2: Front-Left
        '#2ecc71',  # 3: Left
        '#1abc9c',  # 4: Back-Left
        '#3498db',  # 5: Back
        '#9b59b6',  # 6: Back-Right
        '#e91e63',  # 7: Right
        '#95a5a6',  # 8: Center (unused/reference)
    ]

    # Sector labels (Portuguese)
    sector_labels = [
        'Setor 0\nFrente-Dir.',
        'Setor 1\nFrente',
        'Setor 2\nFrente-Esq.',
        'Setor 3\nEsquerda',
        'Setor 4\nTrás-Esq.',
        'Setor 5\nTrás',
        'Setor 6\nTrás-Dir.',
        'Setor 7\nDireita',
    ]

    # LIDAR covers 270 degrees (-135 to +135)
    # We divide this into 8 sectors (each ~33.75 degrees) + center zone
    # Sector angles (start, end) in degrees
    sector_start = -135.0
    sector_width = 270.0 / 8.0  # 33.75 degrees per sector

    max_radius = 4.5  # Max LIDAR range
    min_radius = 0.5  # Inner exclusion zone

    # Draw sectors as wedges
    for i in range(8):
        start_angle = sector_start + i * sector_width
        end_angle = start_angle + sector_width

        # Draw wedge
        wedge = patches.Wedge(
            center=(0, 0),
            r=max_radius,
            theta1=start_angle + 90,  # Rotate to have front at top
            theta2=end_angle + 90,
            width=max_radius - min_radius,
            facecolor=sector_colors[i],
            edgecolor='white',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(wedge)

        # Calculate label position
        mid_angle = np.radians((start_angle + end_angle) / 2 + 90)
        label_radius = (max_radius + min_radius) / 2 + 0.3
        label_x = label_radius * np.cos(mid_angle)
        label_y = label_radius * np.sin(mid_angle)

        ax.text(
            label_x, label_y, sector_labels[i],
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
        )

    # Draw robot body (simplified YouBot shape)
    robot_width = 0.5
    robot_length = 0.6
    robot = patches.FancyBboxPatch(
        (-robot_width/2, -robot_length/2),
        robot_width, robot_length,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor='#34495e',
        edgecolor='white',
        linewidth=3
    )
    ax.add_patch(robot)

    # Add robot front indicator (arrow)
    arrow = patches.FancyArrow(
        0, robot_length/2 - 0.1,
        0, 0.25,
        width=0.15,
        head_width=0.25,
        head_length=0.1,
        facecolor='#2ecc71',
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(arrow)

    # LIDAR sensor position (center of robot)
    ax.plot(0, 0, 'o', markersize=12, color='#e74c3c', markeredgecolor='white', markeredgewidth=2)
    ax.text(0, -0.15, 'LIDAR', ha='center', va='top', fontsize=9, fontweight='bold', color='white')

    # Draw range circles
    for r in [1, 2, 3, 4]:
        circle = plt.Circle((0, 0), r, fill=False, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.add_artist(circle)
        ax.text(r, 0.15, f'{r}m', fontsize=9, color='white', alpha=0.8)

    # Draw "blind zone" indicator (back 90 degrees)
    blind_wedge = patches.Wedge(
        center=(0, 0),
        r=max_radius + 0.3,
        theta1=-45 + 90,  # From back-left
        theta2=45 + 90,   # To back-right
        width=0.2,
        facecolor='#7f8c8d',
        edgecolor='white',
        linewidth=1,
        alpha=0.3
    )
    ax.add_patch(blind_wedge)

    # Add legend/title
    ax.text(0, max_radius + 1.0, 'Divisão em 9 Setores do LIDAR Hokuyo',
            ha='center', va='center', fontsize=16, fontweight='bold', color='black')

    ax.text(0, max_radius + 0.5, '270° de cobertura | 667 pontos | Alcance: 5m',
            ha='center', va='center', fontsize=12, color='#666')

    # Add info box
    info_text = (
        "• CNN processa 667 pontos LIDAR\n"
        "• Saída: probabilidade de ocupação por setor\n"
        "• Usado para navegação segura sem GPS"
    )
    ax.text(-max_radius - 0.8, -max_radius - 0.5, info_text,
            fontsize=10, color='#333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9))

    # Configure axes
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#2c3e50')
    fig.patch.set_facecolor('#2c3e50')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lidar_sectors.png', dpi=300, facecolor='#2c3e50', bbox_inches='tight')
    plt.close()
    print(f"Saved: lidar_sectors.png")


def draw_lidar_sectors_light():
    """Create diagram showing 9-sector LIDAR coverage (light theme for slides)"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Colors for sectors
    sector_colors = [
        '#e74c3c',  # 0: Front-Right
        '#f39c12',  # 1: Front
        '#f1c40f',  # 2: Front-Left
        '#2ecc71',  # 3: Left
        '#1abc9c',  # 4: Back-Left
        '#3498db',  # 5: Back
        '#9b59b6',  # 6: Back-Right
        '#e91e63',  # 7: Right
    ]

    # Sector labels (Portuguese)
    sector_labels = [
        'S0\nFrente-Dir.',
        'S1\nFrente',
        'S2\nFrente-Esq.',
        'S3\nEsquerda',
        'S4\nTrás-Esq.',
        'S5\nTrás',
        'S6\nTrás-Dir.',
        'S7\nDireita',
    ]

    sector_start = -135.0
    sector_width = 270.0 / 8.0

    max_radius = 4.0
    min_radius = 0.6

    # Draw sectors
    for i in range(8):
        start_angle = sector_start + i * sector_width
        end_angle = start_angle + sector_width

        wedge = patches.Wedge(
            center=(0, 0),
            r=max_radius,
            theta1=start_angle + 90,
            theta2=end_angle + 90,
            width=max_radius - min_radius,
            facecolor=sector_colors[i],
            edgecolor='black',
            linewidth=2,
            alpha=0.6
        )
        ax.add_patch(wedge)

        mid_angle = np.radians((start_angle + end_angle) / 2 + 90)
        label_radius = (max_radius + min_radius) / 2 + 0.2
        label_x = label_radius * np.cos(mid_angle)
        label_y = label_radius * np.sin(mid_angle)

        ax.text(
            label_x, label_y, sector_labels[i],
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='black'
        )

    # Robot body
    robot_width = 0.4
    robot_length = 0.5
    robot = patches.FancyBboxPatch(
        (-robot_width/2, -robot_length/2),
        robot_width, robot_length,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor='#34495e',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(robot)

    # Front indicator
    ax.annotate(
        '', xy=(0, robot_length/2 + 0.3), xytext=(0, robot_length/2),
        arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3)
    )
    ax.text(0, robot_length/2 + 0.4, 'FRENTE', ha='center', fontsize=10, fontweight='bold')

    # LIDAR position
    ax.plot(0, 0, 'o', markersize=10, color='#e74c3c', markeredgecolor='black', markeredgewidth=2)

    # Range circles
    for r in [1, 2, 3, 4]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.add_artist(circle)
        ax.text(r + 0.1, 0.1, f'{r}m', fontsize=9, color='gray')

    # Title
    ax.set_title('Divisão LIDAR em 8 Setores\n(270° de cobertura)', fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lidar_sectors_light.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: lidar_sectors_light.png")


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating LIDAR Sector Diagrams...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    draw_lidar_sectors()
    draw_lidar_sectors_light()

    print()
    print("All diagrams generated successfully!")
