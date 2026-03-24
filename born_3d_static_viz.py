#!/usr/bin/env python3
import pygame
import math

# --- Configuration ---
N_REPETITIONS = 10  # <--- Change this to adjust the number of layers
WIDTH, HEIGHT = 800, 600
FOV = 600

# Colors (RGB)
COLOR_TI = (180, 180, 190)      # Silver
COLOR_NI = (80, 80, 100)        # Dark Metallic
COLOR_SI = (40, 40, 40)         # Dark Substrate (Si)
COLOR_BG = (30, 30, 45)         # Background

def project(x, y, z, angle_x, angle_y):
    """3D to 2D projection with rotation."""
    # Rotate around Y axis
    rad_y = math.radians(angle_y)
    nx = x * math.cos(rad_y) + z * math.sin(rad_y)
    nz = -x * math.sin(rad_y) + z * math.cos(rad_y)
    
    # Rotate around X axis
    rad_x = math.radians(angle_x)
    ny = y * math.cos(rad_x) - nz * math.sin(rad_x)
    nz = y * math.sin(rad_x) + nz * math.cos(rad_x)

    # Perspective projection
    nz += 1000 # Distance from camera
    
    factor = FOV / nz
    x_2d = nx * factor + WIDTH // 2
    y_2d = -ny * factor + HEIGHT // 2
    return int(x_2d), int(y_2d), nz

def draw_layer(surface, x, y, z, w, h, d, color, angle_x, angle_y, alpha=220):
    """
    Draws a 3D box representing a layer.
    x, y, z: Center of the bottom face
    w, d: width and depth (X and Z)
    h: height (Y)
    """
    # 8 corners of the box (extending UP from y)
    corners = [
        (x-w, y, z-d), (x+w, y, z-d), (x+w, y, z+d), (x-w, y, z+d),     # Bottom
        (x-w, y+h, z-d), (x+w, y+h, z-d), (x+w, y+h, z+d), (x-w, y+h, z+d) # Top
    ]
    
    proj_points = []
    for c in corners:
        proj_points.append(project(*c, angle_x, angle_y))

    # Faces ordered for vertical rendering (bottom-to-top)
    faces = [
        [0, 1, 2, 3], # Bottom
        [0, 1, 5, 4], # Front
        [2, 3, 7, 6], # Back
        [0, 3, 7, 4], # Left
        [1, 2, 6, 5], # Right
        [4, 5, 6, 7], # Top
    ]

    for face in faces:
        face_pts = [(proj_points[i][0], proj_points[i][1]) for i in face]
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, (*color, alpha), face_pts)
        pygame.draw.polygon(s, (255, 255, 255, 40), face_pts, 1)
        surface.blit(s, (0, 0))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D Vertical Layer Stack")
    
    # Dimensions
    TI_H = 15
    NI_H = 35
    BLOCK_W = 150
    BLOCK_D = 150
    SUBSTRATE_H = 100
    
    ANG_X = 20  # Look down slightly
    ANG_Y = 45  # Isometric-like rotation

    screen.fill(COLOR_BG)
    
    # Calculate starting Y so stack is centered
    total_height = SUBSTRATE_H + (TI_H + NI_H) * N_REPETITIONS
    current_y = -total_height / 2

    # 1. Substrate
    draw_layer(screen, 0, current_y, 0, BLOCK_W, SUBSTRATE_H, BLOCK_D, COLOR_SI, ANG_X, ANG_Y)
    current_y += SUBSTRATE_H

    # 2. Ti/Ni Layers (rendered from bottom to top)
    for i in range(N_REPETITIONS):
        # Ti
        draw_layer(screen, 0, current_y, 0, BLOCK_W, TI_H, BLOCK_D, COLOR_TI, ANG_X, ANG_Y)
        current_y += TI_H
        # Ni
        draw_layer(screen, 0, current_y, 0, BLOCK_W, NI_H, BLOCK_D, COLOR_NI, ANG_X, ANG_Y)
        current_y += NI_H

    # Save and Show
    out_file = "3d_stack_vertical.png"
    pygame.image.save(screen, out_file)
    print(f"Vertical visualization saved as {out_file}")

    pygame.display.flip()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
