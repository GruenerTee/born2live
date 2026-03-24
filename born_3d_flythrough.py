#!/usr/bin/env python3
import pygame
import numpy as np
import math

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FOV = 400
FPS = 60

# Colors (RGB)
COLOR_TI = (180, 180, 190)      # Silver
COLOR_NI = (80, 80, 100)        # Dark Metallic
COLOR_VACUUM = (100, 150, 255)  # Sky blue
COLOR_SI = (40, 40, 40)         # Dark Substrate

def project(x, y, z):
    """Simple 3D to 2D projection."""
    factor = FOV / (z + FOV)
    x_2d = x * factor + WIDTH // 2
    y_2d = -y * factor + HEIGHT // 2
    return int(x_2d), int(y_2d)

def draw_block(surface, x, y, z, w, h, d, color, alpha=180):
    """Draws a 3D wireframe/solid block."""
    # Corners: front-bottom-left, etc.
    points = [
        (x-w, y-h, z), (x+w, y-h, z), (x+w, y+h, z), (x-w, y+h, z),
        (x-w, y-h, z+d), (x+w, y-h, z+d), (x+w, y+h, z+d), (x-w, y+h, z+d)
    ]
    
    # Project points to 2D
    proj_points = []
    for p in points:
        if p[2] + FOV <= 0: return # Behind camera
        proj_points.append(project(*p))

    # Define faces (indices of points)
    faces = [
        [0, 1, 2, 3], # Front
        [4, 5, 6, 7], # Back
        [0, 1, 5, 4], # Bottom
        [2, 3, 7, 6], # Top
        [0, 3, 7, 4], # Left
        [1, 2, 6, 5]  # Right
    ]

    # Draw faces with depth sorting (simplistic)
    for face in faces:
        face_pts = [proj_points[i] for i in face]
        # Create a surface for transparency
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, (*color, alpha), face_pts)
        pygame.draw.polygon(s, (255, 255, 255, 50), face_pts, 1) # Outline
        surface.blit(s, (0, 0))

def run_viz():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    # Layer parameters (units: pixels)
    TI_D = 30
    NI_D = 70
    N_REP = 10
    BLOCK_SIZE = 200
    
    camera_z = -200
    running = True
    saved = False

    while running:
        screen.fill((20, 20, 30)) # Dark background
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move camera forward (passthrough effect)
        camera_z += 2
        # Reset camera if we passed the whole stack
        total_depth = (TI_D + NI_D) * N_REP + 500
        if camera_z > total_depth:
            camera_z = -200

        # Draw Layers in reverse order for simple Z-buffering
        # 1. Substrate
        substrate_z = (TI_D + NI_D) * N_REP
        draw_block(screen, 0, 0, substrate_z - camera_z, BLOCK_SIZE, BLOCK_SIZE, 500, COLOR_SI)

        # 2. Ti/Ni Stack
        for i in range(N_REP-1, -1, -1):
            base_z = i * (TI_D + NI_D)
            # Ni Layer
            draw_block(screen, 0, 0, base_z + TI_D - camera_z, BLOCK_SIZE, BLOCK_SIZE, NI_D, COLOR_NI)
            # Ti Layer
            draw_block(screen, 0, 0, base_z - camera_z, BLOCK_SIZE, BLOCK_SIZE, TI_D, COLOR_TI)

        # 3. Vacuum (indicator)
        draw_block(screen, 0, 0, -500 - camera_z, BLOCK_SIZE, BLOCK_SIZE, 500, COLOR_VACUUM, alpha=50)

        # UI
        font = pygame.font.SysFont('Arial', 24)
        text = font.render(f"Z-Position: {int(camera_z)} | 3D Stack Fly-through", True, (255, 255, 255))
        screen.blit(text, (20, 20))

        # Save the first frame as PNG
        if not saved:
            pygame.image.save(screen, "3d_stack_viz.png")
            print("3D visualization saved as 3d_stack_viz.png")
            saved = True

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run_viz()
