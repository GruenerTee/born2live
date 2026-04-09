import os
import math
from born_lattice_pygame_viz import visualize_lattice_pygame

nm = 1.0
deg = math.pi / 180.0

SHAPES = [
    "Sphere", "Spheroid", "SphericalSegment", "SpheroidalSegment",
    "Cylinder", "Box", "Prism3", "Prism6", "Cone", 
    "Pyramid2", "Pyramid3", "Pyramid4", "Pyramid6", 
    "Bipyramid4", "CantellatedCube", "Dodecahedron", "Icosahedron",
    "PlatonicOctahedron", "PlatonicTetrahedron", "HemiEllipsoid",
    "HorizontalCylinder"
]

def generate_gallery():
    os.makedirs("viz_gallery", exist_ok=True)
    print("Generating Visualization Gallery...")
    
    for shape in SHAPES:
        path = f"viz_gallery/{shape}.png"
        print(f"  Rendering {shape}...")
        try:
            visualize_lattice_pygame(
                path, 
                shape=shape, 
                size=10*nm, 
                aspect=3.0, 
                slope=70*deg, 
                detail=0.3,
                a=50*nm, 
                b=50*nm
            )
        except Exception as e:
            print(f"  [ERROR] {shape}: {e}")

if __name__ == "__main__":
    generate_gallery()
