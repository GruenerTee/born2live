import pygame
import math
import sys

# Define constants if not imported
nm = 1.0
deg = math.pi / 180.0

def visualize_lattice_pygame(path, radius=5*nm, height=5*nm, a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg ):
    """
    Visualizes a 2D lattice of cylinders using Pygame.
    
    Parameters:
    radius: Radius of the cylinders
    height: Height of the cylinders (used for color/shading info)
    a, b: Lattice constants
    alpha: Angle between lattice vectors (in radians)
    xi: Rotation of the first lattice vector (in radians)
    """
    
    # --- Configuration ---
    WIDTH, HEIGHT_WINDOW = 800, 800

    
    SCALE = 10.0  # Pixels per nm
    while SCALE*SCALE < 2*a+2*b:
        SCALE+=5
    
    # Colors
    COLOR_BG = (30, 30, 45)         # Dark background
    COLOR_LATTICE = (100, 100, 100) # Grey for lattice lines
    COLOR_PARTICLE = (200, 180, 100)# Gold-ish for cylinders
    COLOR_V1 = (255, 50, 50)        # Red for vector a
    COLOR_V2 = (50, 255, 50)        # Green for vector b
    COLOR_TEXT = (255, 255, 255)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT_WINDOW))
    pygame.display.set_caption("BornAgain Lattice Visualization")
    font = pygame.font.SysFont('Arial', 16)

    # Lattice vectors
    v1 = (a * math.cos(xi), a * math.sin(xi))
    v2 = (b * math.cos(xi + alpha), b * math.sin(xi + alpha))

    def to_screen(x, y):
        """Convert lattice coordinates to screen pixels."""
        return int(x * SCALE + WIDTH // 2), int(-y * SCALE + HEIGHT_WINDOW // 2)

    
    
    screen.fill(COLOR_BG)

    # Draw lattice grid and particles
    N = 10  # Number of unit cells to show in each direction
    
    # 1. Draw Lattice Lines
    for i in range(-N, N + 1):
        # Lines along v2
        p1 = to_screen(-N * v1[0] + i * v2[0], -N * v1[1] + i * v2[1])
        p2 = to_screen( N * v1[0] + i * v2[0],  N * v1[1] + i * v2[1])
        pygame.draw.line(screen, COLOR_LATTICE, p1, p2, 1)
        
        # Lines along v1
        p1 = to_screen(i * v1[0] - N * v2[0], i * v1[1] - N * v2[1])
        p2 = to_screen(i * v1[0] + N * v2[0], i * v1[1] + N * v2[1])
        pygame.draw.line(screen, COLOR_LATTICE, p1, p2, 1)

    # 2. Draw Particles (Cylinders as circles in top-down)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            pos_x = i * v1[0] + j * v2[0]
            pos_y = i * v1[1] + j * v2[1]
            
            # Check if visible on screen to optimize
            screen_pos = to_screen(pos_x, pos_y)
            if 0 <= screen_pos[0] <= WIDTH and 0 <= screen_pos[1] <= HEIGHT_WINDOW:
                # Draw cylinder top
                pygame.draw.circle(screen, COLOR_PARTICLE, screen_pos, int(radius * SCALE))
                # Add a small highlight to give it volume
                pygame.draw.circle(screen, (255, 255, 255), (screen_pos[0]-int(radius*SCALE*0.3), screen_pos[1]-int(radius*SCALE*0.3)), int(radius * SCALE * 0.1))

    # 3. Draw Basis Vectors at Origin
    origin = to_screen(0, 0)
    end_v1 = to_screen(v1[0], v1[1])
    end_v2 = to_screen(v2[0], v2[1])
    
    pygame.draw.line(screen, COLOR_V1, origin, end_v1, 3)
    pygame.draw.line(screen, COLOR_V2, origin, end_v2, 3)
    
    # Labels
    screen.blit(font.render(f"a = {a:.1f} nm", True, COLOR_V1), (end_v1[0] + 5, end_v1[1]))
    screen.blit(font.render(f"b = {b:.1f} nm", True, COLOR_V2), (end_v2[0] + 5, end_v2[1]))
    
    # Info Panel
    info_text = [
        f"Radius: {radius:.1f} nm",
        f"Height: {height:.1f} nm",
        f"Alpha: {math.degrees(alpha):.1f}°",
        f"Xi: {math.degrees(xi):.1f}°",
        "Controls: ESC to quit"
    ]
    for idx, text in enumerate(info_text):
        screen.blit(font.render(text, True, COLOR_TEXT), (10, 10 + idx * 20))

    #pygame.display.flip()
    pygame.image.save(screen, str(path))

    pygame.quit()

if __name__ == "__main__":
    # Test with default values from creat_sample_test.py
    visualize_lattice_pygame(
        path="test-image.png",
        radius=5*nm, 
        height=5*nm, 
        a=20*nm, 
        b=20*nm, 
        alpha=120*deg, 
        xi=0*deg
    )
