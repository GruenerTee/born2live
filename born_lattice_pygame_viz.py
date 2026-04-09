import os
# Set SDL to use dummy drivers for headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import math
import sys

# Define constants if not imported
nm = 1.0
deg = math.pi / 180.0

class ShapeRenderer:
    """Consolidated 3D projection and drawing logic for all shapes."""
    def __init__(self, width, height_window, fov=3500, camera_dist=1000):
        self.WIDTH = width
        self.HEIGHT = height_window
        self.FOV = fov
        self.CAMERA_DIST = camera_dist

    def project(self, x, y, z, angle_x, angle_y):
        """3D to 2D projection."""
        ry = math.radians(angle_y)
        x_rot = x * math.cos(ry) + z * math.sin(ry)
        z_rot = -x * math.sin(ry) + z * math.cos(ry)
        
        rx = math.radians(angle_x)
        y_final = y * math.cos(rx) + z_rot * math.sin(rx)
        z_final = -y * math.sin(rx) + z_rot * math.cos(rx)

        z_final += self.CAMERA_DIST
        factor = self.FOV / max(0.1, z_final)
        x_2d = x_rot * factor + self.WIDTH // 2
        y_2d = -y_final * factor + self.HEIGHT // 2
        return int(x_2d), int(y_2d), z_final

    def draw_polygon(self, surface, pts_3d, color, angle_x, angle_y, border=True):
        proj = [self.project(*p, angle_x, angle_y) for p in pts_3d]
        avg_z = sum(p[2] for p in proj) / len(proj)
        pts_2d = [(p[0], p[1]) for p in proj]
        pygame.draw.polygon(surface, color, pts_2d)
        
        if border:
            if len(pts_2d) >= 3:
                v1 = (pts_2d[1][0] - pts_2d[0][0], pts_2d[1][1] - pts_2d[0][1])
                v2 = (pts_2d[2][0] - pts_2d[1][0], pts_2d[2][1] - pts_2d[1][1])
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                if cross_product > 0:
                    pygame.draw.polygon(surface, (255, 255, 255, 50), pts_2d, 1)
            else:
                pygame.draw.polygon(surface, (255, 255, 255, 50), pts_2d, 1)
        return avg_z

    def render_shape(self, surface, shape, x, z, size, H, slope, detail, color, angle_x, angle_y):
        """Dispatches to specific shape drawing methods."""
        if shape == "HemiEllipsoid":
            self.draw_hemiellipsoid(surface, x, z, size, H, color, angle_x, angle_y, detail)
        elif shape in ["Sphere", "SphericalSegment"]:
            self.draw_ellipsoid(surface, x, z, size, 2*size, color, angle_x, angle_y, detail, is_segment=("Segment" in shape))
        elif shape in ["Spheroid", "SpheroidalSegment"]:
            self.draw_ellipsoid(surface, x, z, size, H, color, angle_x, angle_y, detail, is_segment=("Segment" in shape))
        elif shape in ["Cylinder", "Box", "Prism3", "Prism6"]:
            self.draw_prism(surface, x, z, size, H, color, angle_x, angle_y, shape, detail)
        elif shape in ["Cone", "Pyramid2", "Pyramid3", "Pyramid4", "Pyramid6", "Bipyramid4"]:
            self.draw_pyramid(surface, x, z, size, H, slope, color, angle_x, angle_y, shape, detail)
        elif shape == "HorizontalCylinder":
            s_bottom = -size if detail <= 0 else -size * detail
            s_top = size if detail <= 0 else size * (1.0 - detail)
            self.draw_horizontal_cylinder(surface, x, z, size, H, s_bottom, s_top, color, angle_x, angle_y)
        elif shape in ["Dodecahedron", "Icosahedron", "PlatonicOctahedron", "PlatonicTetrahedron"]:
            self.draw_platonic(surface, x, z, size, color, angle_x, angle_y, shape)
        elif shape == "CantellatedCube":
            self.draw_box(surface, x, 0, z, size, size*2, size, color, angle_x, angle_y)
        else:
            self.draw_prism(surface, x, z, size, H, color, angle_x, angle_y, "Cylinder", detail)

    def draw_platonic(self, surface, x, z, size, color, angle_x, angle_y, shape_type):
        if shape_type == "PlatonicTetrahedron":
            v = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
            f = [(0,1,2), (0,2,3), (0,3,1), (1,3,2)]
            s = size / 1.5
        elif shape_type == "PlatonicOctahedron":
            v = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
            f = [(0,2,4), (0,4,3), (0,3,5), (0,5,2), (1,2,5), (1,5,3), (1,3,4), (1,4,2)]
            s = size / 1.2
        elif shape_type == "Icosahedron":
            phi = (1 + math.sqrt(5)) / 2
            v = [(0,1,phi), (0,1,-phi), (0,-1,phi), (0,-1,-phi), (1,phi,0), (1,-phi,0), (-1,phi,0), (-1,-phi,0), (phi,0,1), (phi,0,-1), (-phi,0,1), (-phi,0,-1)]
            f = [(0,8,4), (0,4,6), (0,6,10), (0,10,2), (0,2,8), (1,9,5), (1,5,11), (1,11,7), (1,7,3), (1,3,9), (2,10,11), (2,11,5), (2,5,8), (3,7,6), (3,6,4), (3,4,9), (4,8,9), (5,9,8), (6,7,11), (6,11,10)]
            s = size / 2.0
        elif shape_type == "Dodecahedron":
            phi = (1 + math.sqrt(5)) / 2
            v = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1), (0,1/phi,phi), (0,1/phi,-phi), (0,-1/phi,phi), (0,-1/phi,-phi), (1/phi,phi,0), (1/phi,-phi,0), (-1/phi,phi,0), (-1/phi,-phi,0), (phi,0,1/phi), (phi,0,-1/phi), (-phi,0,1/phi), (-phi,0,-1/phi)]
            f = [(0,8,10,2,16), (0,16,17,1,12), (0,12,14,4,8), (1,17,3,13,9), (1,9,11,5,12), (2,10,6,15,13), (2,13,3,17,16), (3,11,7,15,13), (4,14,15,6,10), (4,10,8,0,12), (5,11,9,1,17), (5,17,14,12,4)]
            s = size / 1.8
        else: return
        face_list = []
        for face_indices in f:
            pts_3d = [(x + v[i][0]*s, v[i][1]*s + s, z + v[i][2]*s) for i in face_indices if i < len(v)]
            if len(pts_3d) < 3: continue
            mx = sum(p[0] for p in pts_3d)/len(pts_3d)
            my = sum(p[1] for p in pts_3d)/len(pts_3d)
            mz = sum(p[2] for p in pts_3d)/len(pts_3d)
            _, _, depth = self.project(mx, my, mz, angle_x, angle_y)
            face_list.append((depth, pts_3d))
        face_list.sort(key=lambda x: x[0], reverse=True)
        for _, pts in face_list: self.draw_polygon(surface, pts, color, angle_x, angle_y)

    def draw_prism(self, surface, x, z, r, h, color, angle_x, angle_y, type="Cylinder", detail=0.0):
        if type == "Box":
            w = r if detail <= 0 else r * detail
            self.draw_box(surface, x, 0, z, r, h, w, color, angle_x, angle_y)
            return
        num_sides = 32 if type == "Cylinder" else (3 if type == "Prism3" else 6)
        bottom_pts = [(x + r*math.cos(2*math.pi*i/num_sides), 0, z + r*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
        top_pts = [(x + r*math.cos(2*math.pi*i/num_sides), h, z + r*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
        for i in range(num_sides):
            next_i = (i + 1) % num_sides
            side_pts = [bottom_pts[i], bottom_pts[next_i], top_pts[next_i], top_pts[i]]
            shade = 0.7 + 0.3 * math.cos(2 * math.pi * i / num_sides)
            self.draw_polygon(surface, side_pts, [int(c * shade) for c in color], angle_x, angle_y)
        self.draw_polygon(surface, top_pts, color, angle_x, angle_y)

    def draw_pyramid(self, surface, x, z, r, h, slope, color, angle_x, angle_y, type="Pyramid4", detail=0.0):
        num_sides = 4
        if "Pyramid2" in type: num_sides = 4
        elif "Pyramid3" in type: num_sides = 3
        elif "Pyramid6" in type: num_sides = 6
        elif "Cone" in type: num_sides = 16
        is_bipyramid = "Bipyramid" in type
        h_mid = h * (1.0 - detail) if is_bipyramid else 0
        r_peak = r - (h * detail / math.tan(slope)) if slope < math.pi/2 else 0
        r_peak = max(0, r_peak)
        r_bottom = r - (h * (1.0 - detail) / math.tan(slope)) if slope < math.pi/2 else 0
        r_bottom = max(0, r_bottom)
        bottom_pts = [(x + r_bottom*math.cos(2*math.pi*i/num_sides), 0, z + r_bottom*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
        mid_pts = [(x + r*math.cos(2*math.pi*i/num_sides), h_mid, z + r*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
        top_pts = [(x + r_peak*math.cos(2*math.pi*i/num_sides), h, z + r_peak*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
        if is_bipyramid:
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                self.draw_polygon(surface, [bottom_pts[i], bottom_pts[next_i], mid_pts[next_i], mid_pts[i]], color, angle_x, angle_y)
                self.draw_polygon(surface, [mid_pts[i], mid_pts[next_i], top_pts[next_i], top_pts[i]], color, angle_x, angle_y)
            if r_peak > 0: self.draw_polygon(surface, top_pts, color, angle_x, angle_y)
        else:
            r_top = r - (h / math.tan(slope)) if slope < math.pi/2 else 0
            r_top = max(0, r_top)
            base_pts = [(x + r*math.cos(2*math.pi*i/num_sides), 0, z + r*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
            peak_pts = [(x + r_top*math.cos(2*math.pi*i/num_sides), h, z + r_top*math.sin(2*math.pi*i/num_sides)) for i in range(num_sides)]
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                self.draw_polygon(surface, [base_pts[i], base_pts[next_i], peak_pts[next_i], peak_pts[i]], color, angle_x, angle_y)
            if r_top > 0: self.draw_polygon(surface, peak_pts, color, angle_x, angle_y)

    def draw_hemiellipsoid(self, surface, x, z, r, h, color, angle_x, angle_y, detail=0.0):
        num_stacks, num_segments = 12, 24
        ry = r if detail <= 0 else r * detail
        prev_pts = None
        for i in range(num_stacks + 1):
            phi = (math.pi / 2) * (1 - i / num_stacks)
            curr_y = h * math.sin((math.pi/2) * (i/num_stacks))
            curr_rx, curr_rz = r * math.cos((math.pi/2) * (i/num_stacks)), ry * math.cos((math.pi/2) * (i/num_stacks))
            pts = [(x + curr_rx*math.cos(2*math.pi*j/num_segments), curr_y, z + curr_rz*math.sin(2*math.pi*j/num_segments)) for j in range(num_segments)]
            if prev_pts:
                for j in range(num_segments):
                    next_j = (j + 1) % num_segments
                    quad = [prev_pts[j], prev_pts[next_j], pts[next_j], pts[j]]
                    shade = 0.5 + 0.5 * math.sin(phi) * (0.7 + 0.3 * math.cos(2*math.pi*j/num_segments))
                    self.draw_polygon(surface, quad, [int(c * shade) for c in color], angle_x, angle_y, border=False)
            prev_pts = pts
        base_pts = [(x + r*math.cos(2*math.pi*j/num_segments), 0, z + ry*math.sin(2*math.pi*j/num_segments)) for j in range(num_segments)]
        self.draw_polygon(surface, base_pts, color, angle_x, angle_y, border=True)

    def draw_ellipsoid(self, surface, x, z, r, h, color, angle_x, angle_y, detail=0.0, is_segment=False):
        num_stacks, num_segments = 16, 24
        for i in range(num_stacks + 1):
            phi = math.pi * i / num_stacks
            if is_segment and phi < detail * math.pi: continue
            curr_h, curr_r = (h/2) * (1 - math.cos(phi)), r * math.sin(phi)
            pts = [(x + curr_r*math.cos(2*math.pi*j/num_segments), curr_h, z + curr_r*math.sin(2*math.pi*j/num_segments)) for j in range(num_segments)]
            shade = 0.6 + 0.4 * (i / num_stacks)
            self.draw_polygon(surface, pts, [int(c * shade) for c in color], angle_x, angle_y, border=(i == num_stacks))

    def draw_horizontal_cylinder(self, surface, x, z, r, length, s_bottom, s_top, color, angle_x, angle_y):
        num_sides = 32
        x_start, quads = x - length/2, []
        for i in range(num_sides):
            phi, phi_next = 2 * math.pi * i / num_sides, 2 * math.pi * (i+1) / num_sides
            z1, y1 = r * math.sin(phi), r * math.cos(phi)
            z2, y2 = r * math.sin(phi_next), r * math.cos(phi_next)
            if s_bottom <= z1 <= s_top or s_bottom <= z2 <= s_top:
                z1_c, z2_c = max(s_bottom, min(s_top, z1)), max(s_bottom, min(s_top, z2))
                pts = [(x_start, z1_c + r, z + y1), (x_start + length, z1_c + r, z + y1), (x_start + length, z2_c + r, z + y2), (x_start, z2_c + r, z + y2)]
                _, _, depth = self.project(x, (z1_c+z2_c)/2 + r, z + (y1+y2)/2, angle_x, angle_y)
                quads.append((depth, pts, [int(c * (0.5 + 0.5 * math.cos(phi))) for c in color]))
        quads.sort(key=lambda q: q[0], reverse=True)
        for _, pts, c in quads: self.draw_polygon(surface, pts, c, angle_x, angle_y, border=False)
        if s_top < r:
            w = math.sqrt(max(0, r**2 - s_top**2))
            top_rect = [(x_start, s_top + r, z - w), (x_start + length, s_top + r, z - w), (x_start + length, s_top + r, z + w), (x_start, s_top + r, z + w)]
            self.draw_polygon(surface, top_rect, [int(c * 1.1) for c in color], angle_x, angle_y)
        for end_x in [x_start, x_start + length]:
            cap_pts = []
            for i in range(num_sides + 1):
                phi = 2 * math.pi * i / num_sides
                zv, yv = r * math.sin(phi), r * math.cos(phi)
                if s_bottom <= zv <= s_top: cap_pts.append((end_x, zv + r, z + yv))
            if len(cap_pts) >= 3: self.draw_polygon(surface, cap_pts, color, angle_x, angle_y)

    def draw_box(self, surface, x, y, z, w, h, d, color, angle_x, angle_y):
        corners = [(x-w, y, z-d), (x+w, y, z-d), (x+w, y, z+d), (x-w, y, z+d), (x-w, y+h, z-d), (x+w, y+h, z-d), (x+w, y+h, z+d), (x-w, y+h, z+d)]
        self.draw_polygon(surface, [corners[4], corners[5], corners[6], corners[7]], color, angle_x, angle_y)
        self.draw_polygon(surface, [corners[0], corners[1], corners[5], corners[4]], [int(c*0.8) for c in color], angle_x, angle_y)
        self.draw_polygon(surface, [corners[1], corners[2], corners[6], corners[5]], [int(c*0.9) for c in color], angle_x, angle_y)

def visualize_lattice_pygame(path, shape="Cylinder", size=5*nm, aspect=1.0, slope=90*deg, detail=0.0,
                            a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg, **kwargs):
    if 'radius' in kwargs: size = kwargs['radius']
    if 'height' in kwargs: aspect = kwargs['height'] / size if size != 0 else 1.0
    H = size * aspect
    WIDTH, HEIGHT_WINDOW = 800, 800
    ANG_X, ANG_Y = 35.26, 45.0
    COLOR_BG, COLOR_LATTICE, COLOR_PARTICLE, COLOR_SUBSTRATE, COLOR_TEXT = (30, 30, 45), (80, 80, 100), (200, 180, 100), (60, 60, 70), (255, 255, 255)
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT_WINDOW))
    renderer = ShapeRenderer(WIDTH, HEIGHT_WINDOW)
    font = pygame.font.SysFont('Arial', 16)
    v1, v2 = (a * math.cos(xi), a * math.sin(xi)), (b * math.cos(xi + alpha), b * math.sin(xi + alpha))
    screen.fill(COLOR_BG)
    N = 4
    objects_to_draw = []
    objects_to_draw.append(('substrate', 0, 0, 0, (N+2)*a, 20, (N+2)*b))
    for i in range(-N, N + 1):
        objects_to_draw.append(('line', (-N*v1[0]+i*v2[0],0,-N*v1[1]+i*v2[1]), (N*v1[0]+i*v2[0],0,N*v1[1]+i*v2[1])))
        objects_to_draw.append(('line', (i*v1[0]-N*v2[0],0,i*v1[1]-N*v2[1]), (i*v1[0]+N*v2[0],0,i*v1[1]+N*v2[1])))
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            objects_to_draw.append(('particle', i*v1[0]+j*v2[0], i*v1[1]+j*v2[1]))
    def get_z(obj):
        if obj[0] == 'substrate': return 1e9
        if obj[0] == 'particle': px, pz, bias = obj[1], obj[2], -100
        else: px, pz, bias = (obj[1][0] + obj[2][0]) / 2, (obj[1][2] + obj[2][2]) / 2, 100
        _, _, z = renderer.project(px, 0, pz, ANG_X, ANG_Y)
        return z + bias
    objects_to_draw.sort(key=get_z, reverse=True)
    for obj in objects_to_draw:
        if obj[0] == 'substrate': renderer.draw_box(screen, 0, -20, 0, (N+1)*a, 20, (N+1)*b, COLOR_SUBSTRATE, ANG_X, ANG_Y)
        elif obj[0] == 'line':
            p1, p2 = renderer.project(*obj[1], ANG_X, ANG_Y), renderer.project(*obj[2], ANG_X, ANG_Y)
            pygame.draw.line(screen, COLOR_LATTICE, (p1[0], p1[1]), (p2[0], p2[1]), 1)
        elif obj[0] == 'particle': renderer.render_shape(screen, shape, obj[1], obj[2], size, H, slope, detail, COLOR_PARTICLE, ANG_X, ANG_Y)
    info = [f"Shape: {shape}", f"Size: {size:.1f}nm, Aspect: {aspect:.1f}", f"Lattice a: {a:.1f}nm"]
    for idx, text in enumerate(info): screen.blit(font.render(text, True, COLOR_TEXT), (10, 10 + idx * 20))
    pygame.image.save(screen, str(path))
    pygame.quit()
