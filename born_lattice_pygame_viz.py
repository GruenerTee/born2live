import os
# Set SDL to use the dummy video driver for headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import math
import sys

# Define constants if not imported
nm = 1.0
deg = math.pi / 180.0

def visualize_lattice_pygame(path, radius=5*nm, height=5*nm, a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg):
    """
    Visualizes a 2D lattice of cylinders on a substrate box in 3D.
    """
    
    # --- Configuration ---
    WIDTH, HEIGHT_WINDOW = 800, 800
    FOV = 3500      # Increased for more zoom
    CAMERA_DIST = 1000 # Decreased to bring camera closer
    ANG_X = 35.26  
    ANG_Y = 45.0   
    
    # Colors
    COLOR_BG = (30, 30, 45)         
    COLOR_LATTICE = (80, 80, 100) 
    COLOR_PARTICLE = (200, 180, 100)
    COLOR_SUBSTRATE = (60, 60, 70)  # Dark grey substrate
    COLOR_TEXT = (255, 255, 255)

    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT_WINDOW))
    font = pygame.font.SysFont('Arial', 16)

    # Lattice vectors
    v1 = (a * math.cos(xi), a * math.sin(xi))
    v2 = (b * math.cos(xi + alpha), b * math.sin(xi + alpha))

    def project(x, y, z, angle_x, angle_y):
        """3D to 2D projection."""
        ry = math.radians(angle_y)
        x_rot = x * math.cos(ry) + z * math.sin(ry)
        z_rot = -x * math.sin(ry) + z * math.cos(ry)
        
        rx = math.radians(angle_x)
        y_final = y * math.cos(rx) + z_rot * math.sin(rx)
        z_final = -y * math.sin(rx) + z_rot * math.cos(rx)

        z_final += CAMERA_DIST
        factor = FOV / max(0.1, z_final)
        x_2d = x_rot * factor + WIDTH // 2
        y_2d = -y_final * factor + HEIGHT_WINDOW // 2
        return int(x_2d), int(y_2d), z_final

    def draw_box(surface, x, y, z, w, h, d, color, angle_x, angle_y):
        """Draws a 3D box (substrate). y is the top surface height."""
        # 8 corners
        corners = [
            (x-w, y-h, z-d), (x+w, y-h, z-d), (x+w, y-h, z+d), (x-w, y-h, z+d), # Bottom
            (x-w, y,   z-d), (x+w, y,   z-d), (x+w, y,   z+d), (x-w, y,   z+d)  # Top
        ]
        proj = [project(*c, angle_x, angle_y) for c in corners]
        
        # Faces: [indices], average_z
        faces = [
            ([4, 5, 6, 7], sum(proj[i][2] for i in [4,5,6,7])/4, 1.0), # Top
            ([0, 1, 5, 4], sum(proj[i][2] for i in [0,1,5,4])/4, 0.8), # Front
            ([1, 2, 6, 5], sum(proj[i][2] for i in [1,2,6,5])/4, 0.9), # Right
            ([2, 3, 7, 6], sum(proj[i][2] for i in [2,3,7,6])/4, 0.7), # Back
            ([3, 0, 4, 7], sum(proj[i][2] for i in [3,0,4,7])/4, 0.6), # Left
        ]
        # Sort faces back-to-front
        faces.sort(key=lambda f: f[1], reverse=True)
        for f_indices, _, shade in faces:
            pts = [(proj[i][0], proj[i][1]) for i in f_indices]
            shaded_color = tuple(int(c * shade) for c in color)
            pygame.draw.polygon(surface, shaded_color, pts)
            pygame.draw.polygon(surface, (255, 255, 255, 50), pts, 1)

    def draw_cylinder(surface, x, z, r, h, color, angle_x, angle_y):
        """Draws a cylinder prism."""
        num_sides = 12
        bottom_pts = [project(x + r*math.cos(2*math.pi*i/num_sides), 0, z + r*math.sin(2*math.pi*i/num_sides), angle_x, angle_y) for i in range(num_sides)]
        top_pts = [project(x + r*math.cos(2*math.pi*i/num_sides), h, z + r*math.sin(2*math.pi*i/num_sides), angle_x, angle_y) for i in range(num_sides)]
        
        side_faces = []
        for i in range(num_sides):
            next_i = (i + 1) % num_sides
            avg_z = (bottom_pts[i][2] + bottom_pts[next_i][2] + top_pts[i][2] + top_pts[next_i][2]) / 4
            side_faces.append((avg_z, i, next_i))
        side_faces.sort(key=lambda x: x[0], reverse=True)

        for _, i, next_i in side_faces:
            pts = [(bottom_pts[i][0], bottom_pts[i][1]), (bottom_pts[next_i][0], bottom_pts[next_i][1]), (top_pts[next_i][0], top_pts[next_i][1]), (top_pts[i][0], top_pts[i][1])]
            shade = 0.7 + 0.3 * math.cos(2 * math.pi * i / num_sides)
            pygame.draw.polygon(surface, [int(c * shade) for c in color], pts)

        top_poly = [(p[0], p[1]) for p in top_pts]
        pygame.draw.polygon(surface, color, top_poly)
        pygame.draw.polygon(surface, (255, 255, 255, 100), top_poly, 1)

    screen.fill(COLOR_BG)

    # Prepare objects
    N = 6
    objects_to_draw = []
    
    # Substrate
    # Make the substrate significantly larger and thicker
    SUB_W = (N + 10) * a
    SUB_D = (N + 10) * b
    objects_to_draw.append(('substrate', 0, 0, 0, SUB_W, 40, SUB_D, COLOR_SUBSTRATE))

    # Lattice lines
    for i in range(-N, N + 1):
        objects_to_draw.append(('line', (-N*v1[0]+i*v2[0],0,-N*v1[1]+i*v2[1]), (N*v1[0]+i*v2[0],0,N*v1[1]+i*v2[1]), COLOR_LATTICE))
        objects_to_draw.append(('line', (i*v1[0]-N*v2[0],0,i*v1[1]-N*v2[1]), (i*v1[0]+N*v2[0],0,i*v1[1]+N*v2[1]), COLOR_LATTICE))

    # Particles
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            objects_to_draw.append(('cylinder', i*v1[0]+j*v2[0], i*v1[1]+j*v2[1], radius, height, COLOR_PARTICLE))

    def get_sort_z(obj):
        if obj[0] == 'substrate': return 5000 # Draw substrate first (or at the back)
        if obj[0] == 'line': return (project(*obj[1], ANG_X, ANG_Y)[2] + project(*obj[2], ANG_X, ANG_Y)[2]) / 2
        return project(obj[1], 0, obj[2], ANG_X, ANG_Y)[2]

    objects_to_draw.sort(key=get_sort_z, reverse=True)

    for obj in objects_to_draw:
        if obj[0] == 'substrate':
            draw_box(screen, *obj[1:8], ANG_X, ANG_Y)
        elif obj[0] == 'line':
            p1, p2 = project(*obj[1], ANG_X, ANG_Y), project(*obj[2], ANG_X, ANG_Y)
            pygame.draw.line(screen, obj[3], (p1[0], p1[1]), (p2[0], p2[1]), 1)
        elif obj[0] == 'cylinder':
            draw_cylinder(screen, *obj[1:6], ANG_X, ANG_Y)

    # Info Panel
    info_text = [f"3D Lattice with Substrate", f"Radius: {radius:.1f} nm", f"Height: {height:.1f} nm", f"a: {a:.1f}, b: {b:.1f} nm"]
    for idx, text in enumerate(info_text):
        screen.blit(font.render(text, True, COLOR_TEXT), (10, 10 + idx * 20))

    pygame.image.save(screen, str(path))
    pygame.quit()

if __name__ == "__main__":
    visualize_lattice_pygame_3d(path="test-image-3d.png", radius=5*nm, height=15*nm, a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg)
