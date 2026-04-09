#!/usr/bin/env python3
import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bornagain as ba
from bornagain import ba_plot as bp
from physics_models import LatticeModel, SimulationEngine
from born_lattice_pygame_viz import visualize_lattice_pygame
from viz.movie_maker import MovieMaker

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

def animate_growth_with_sim():
    base_dir = "growth_with_sim"
    os.makedirs(base_dir, exist_ok=True)
    
    a = 40*nm
    n_frames = 15 # Reduced slightly for faster execution
    sizes = np.linspace(2*nm, a/2 * 0.85, n_frames)
    
    print(f"Starting Growth + Simulation Animation for {len(SHAPES)} shapes...")
    mm = MovieMaker(frame_rate=8, height=800)
    
    for shape in SHAPES:
        shape_dir = os.path.join(base_dir, shape)
        sim_dir = os.path.join(shape_dir, "sim")
        viz_dir = os.path.join(shape_dir, "viz")
        os.makedirs(sim_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"  Processing {shape}...")
        for i, size in enumerate(tqdm(sizes, desc=f"Calculating {shape}", leave=False)):
            # 1. Run BornAgain Simulation
            model = LatticeModel(shape=shape, size=size, aspect=1.5, slope=75*deg, detail=0.3, a=a, b=a)
            sample = model.create_sample()
            result = SimulationEngine.run_scattering_2d(sample, n_pixels=200)
            
            # 2. Save Simulation Plot
            sim_path = os.path.join(sim_dir, f"frame-{i:03d}.png")
            plt.figure(figsize=(8, 8))
            bp.plot_datafield(result)
            plt.title(f"Scattering: {shape} (size={size:.1f}nm)")
            plt.tight_layout()
            plt.savefig(sim_path, dpi=100)
            plt.close()
            
            # 3. Save 3D Visualization
            viz_path = os.path.join(viz_dir, f"frame-{i:03d}.png")
            visualize_lattice_pygame(
                viz_path,
                shape=shape, size=size, aspect=1.5, slope=75*deg, detail=0.3, a=a, b=a
            )
        
        # 4. Create Side-by-Side Video
        # MovieMaker.create_side_by_side usually expects one directory with two types of prefixes
        # but let's check its implementation or just combine them here.
        video_path = os.path.join(base_dir, f"sim_growth_{shape}.mp4")
        
        # Combine frames manually into a temp directory for MovieMaker/FFmpeg
        combined_dir = os.path.join(shape_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        
        for i in range(n_frames):
            from PIL import Image
            img_sim = Image.open(os.path.join(sim_dir, f"frame-{i:03d}.png"))
            img_viz = Image.open(os.path.join(viz_dir, f"frame-{i:03d}.png"))
            
            # Create a side-by-side image
            dst = Image.new('RGB', (img_sim.width + img_viz.width, img_sim.height))
            dst.paste(img_sim, (0, 0))
            dst.paste(img_viz, (img_sim.width, 0))
            dst.save(os.path.join(combined_dir, f"frame-{i:03d}.png"))

        # Finalize Video
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", f"{combined_dir}/frame-%03d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [OK] Created {video_path}")

if __name__ == "__main__":
    animate_growth_with_sim()
