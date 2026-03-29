#!/usr/bin/env python3
import bornagain as ba
from bornagain import deg, nm, angstrom
import numpy as np
import os
import datetime
import multiprocessing
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from bornagain import ba_plot as bp
from creat_sample_test import make_particle_lattice_sample
from born_lattice_pygame_viz import visualize_lattice_pygame

def get_simulation(sample):
    n = 500
    scan = ba.AlphaScan(n, 2*deg/n, 2*deg)
    scan.setWavelength(1.54*angstrom)
    return ba.SpecularSimulation(scan, sample)
def get_simulation2d(sample):
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    n = 200
    detector = ba.SphericalDetector(n, -2*deg, 2*deg, n, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation

def wrapper(args):
    simulate_and_create(*args)

def simulate_and_create(radius, height, ref ,a, i, scenario_name, description, path):
    plt = bp.plt
    plt.figure(figsize=(10, 6))

    # print(f"[{scenario_name}] Sim {i:03d}: {description}")
    sample = make_particle_lattice_sample(radius=radius, height=height, ref_ = ref,a=a, b=a)
    simulation = get_simulation2d(sample)
    result = simulation.simulate()
            
    bp.plot_datafield(result)
    plt.title(f"{scenario_name.replace('_', ' ').title()}\n{description}")
    
    # Save in scenario-specific folder
    out_dir = f"{path}{scenario_name}"
    # print (out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(f"{out_dir}/frame-{i:03d}.png", dpi=150)
    plt.close()
    visualize_lattice_pygame(f"{out_dir}/viz-{i:03d}.png", radius=radius, height=height, a=a, b=a)

if __name__ == '__main__':
    # Define baseline parameters
    time= datetime.datetime.now()
    base_radius = 5*nm
    base_height = 4*nm
    base_lattice = 10*nm
    n_frames = 10
    ref_values= [ 0.0006 ]  # np.linspace(0.0001, 0.001 , n_frames)
    scenarios = {
        "varying_radius": {
            "values": np.linspace(1*nm, 5*nm, n_frames),
            "param": "radius"
        },
        "varying_height": {
            "values": [5*nm], # np.linspace(1*nm, 10*nm, n_frames),
            "param": "height"
        },
        "varying_lattice": {
            "values":  np.linspace(10*nm, 150*nm, n_frames),
            "param": "a"
        }
    }
    for ref in ref_values:
        path='sim/'+ str(time.strftime("%Y-%m-%d__T%H:%MZ"))+'/'+str(ref)+'/'
    # Ensure sim directory exists
        if not os.path.exists(path):
            os.makedirs(path )

        all_tasks = []
        for s_name, config in scenarios.items():
            # Clean scenario directory
            os.system(f"rm -rf {path}{s_name} ; mkdir -p {path}{s_name}")
            
            for i, val in enumerate(config["values"]):
                #print (i, val) 
                # Setup params for this specific frame
                params = {"radius": base_radius, "height": base_height, "a": base_lattice}
                params[config["param"]] = val
                
                desc = f"r={params['radius']/nm:.1f}nm, h={params['height']/nm:.1f}nm, a={params['a']/nm:.4f}nm"
                all_tasks.append((params["radius"], params["height"],ref, params["a"], i, s_name, desc, path))

        print(f"Starting {len(all_tasks)} simulations across {len(scenarios)} scenarios...")
        with multiprocessing.Pool(processes=1) as p: 
            list(tqdm(p.imap(wrapper, all_tasks), total=len(all_tasks), desc="Simulations", leave=True))

        print("\nGenerating videos...")

        for s_name in tqdm(scenarios.keys(), desc="Video generation"):
            video_path = f"{path}film-{s_name}.mp4"
            # Combine simulation and pygame viz side-by-side, scaling them to the same height
            cmd = f"ffmpeg -r 8 -i {path}{s_name}/frame-%03d.png -r 8 -i {path}{s_name}/viz-%03d.png -filter_complex \"[0:v]scale=-1:800[v0];[1:v]scale=-1:800[v1];[v0][v1]hstack\" -vcodec mpeg4 -y {video_path} >/dev/null 2>&1"
            os.system(cmd)
            # print(f"Created: {video_path}")
