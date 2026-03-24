#!/usr/bin/env python3
import pygame
import sys

def draw_stack(N_REPETITIONS= 10, TOP_LAYER='TI' ,FACTOR=1, SIM_NUMBER=1):
    # --- Configuration ---
    SCALE = 0.5 # 1 Angstrom = 4 pixels (adjust for visibility)
    TI_THICKNESS = 30  # Angstroms
    NI_THICKNESS = 70  # Angstroms
    
    
    # Colors (RGB)
    COLOR_VACUUM = (200, 230, 255)  # Light Blue
    COLOR_TI = (192, 192, 192)      # Silver/Grey
    COLOR_NI = (100, 100, 120)      # Dark Metallic Blue/Grey
    COLOR_SI = (50, 50, 50)         # Dark Grey (Substrate)
    COLOR_TEXT = (0, 0, 0)

    # Calculate Dimensions
    stack_height_px = (TI_THICKNESS + NI_THICKNESS) * N_REPETITIONS * SCALE
    vacuum_height_px = 100
    substrate_height_px = 150
    width_px = 600
    total_height_px = vacuum_height_px + stack_height_px + substrate_height_px

    # Initialize Pygame
    pygame.init()
    screen = pygame.Surface((width_px, total_height_px)) # Render to surface first
    font = pygame.font.SysFont('Arial', 18)

    # 1. Draw Vacuum (Top)
    pygame.draw.rect(screen, COLOR_VACUUM, (0, 0, width_px, vacuum_height_px))
    screen.blit(font.render("Vacuum", True, COLOR_TEXT), (10, 10))

    # 2. Draw Periodic Stack
    current_y = vacuum_height_px
    ti_px = TI_THICKNESS * SCALE
    ni_px = NI_THICKNESS * SCALE


    if (TOP_LAYER == 'Ti'):
        pygame.draw.rect(screen, COLOR_TI, (0, current_y, width_px, FACTOR*ti_px))
        screen.blit(font.render(f"Ti {str(FACTOR*ti_px)} )", True, COLOR_TEXT), (width_px-100, current_y + 5))
        current_y+=int(FACTOR*ti_px)
        pygame.draw.rect(screen, COLOR_NI, (0, current_y, width_px, ni_px))
        screen.blit(font.render(f"Ni ({NI_THICKNESS}A)", True, (255,255,255)), (width_px-100, current_y + 5))
        current_y += ni_px
    else:
        pygame.draw.rect(screen, COLOR_NI, (0, current_y, width_px, FACTOR*ni_px))
        screen.blit(font.render(f"Ni {str(FACTOR*ni_px)}A)", True, (255,255,255)), (width_px-100, current_y + 5))
        current_y += int(FACTOR*ni_px)

          

    for i in range(N_REPETITIONS):
        # Ti Layer
        pygame.draw.rect(screen, COLOR_TI, (0, current_y, width_px, ti_px))
        screen.blit(font.render(f"Ti ({TI_THICKNESS}A)", True, COLOR_TEXT), (width_px-100, current_y + 5))
        current_y += ti_px
        
        # Ni Layer
        pygame.draw.rect(screen, COLOR_NI, (0, current_y, width_px, ni_px))
        screen.blit(font.render(f"Ni ({NI_THICKNESS}A)", True, (255,255,255)), (width_px-100, current_y + 5))
        current_y += ni_px

    # 3. Draw Substrate (Bottom)
    pygame.draw.rect(screen, COLOR_SI, (0, current_y, width_px, substrate_height_px))
    screen.blit(font.render("Si Substrate", True, (255, 255, 255)), (10, current_y + 10))

    # Save the rendered image
    output_filename = "layer_stack_viz-"+f"{SIM_NUMBER:02d}"+".png"
    pygame.image.save(screen, './sim/'+output_filename)
    print(f"Visualization saved as {output_filename}")

    # Display in a window
    
    
    # Simple loop to show the image
    running = False
    if running:
        display_screen = pygame.display.set_mode((width_px, min(total_height_px, 900)))
        pygame.display.set_caption("BornAgain Layer Stack Visualization")
    offset_y = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: offset_y += 20
                if event.key == pygame.K_DOWN: offset_y -= 20

        display_screen.fill((255, 255, 255))
        display_screen.blit(screen, (0, offset_y))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    draw_stack(0, 'NI')
