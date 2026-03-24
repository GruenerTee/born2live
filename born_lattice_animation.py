#!/usr/bin/env python3
import bornagain as ba
from bornagain import deg, nm, angstrom
import numpy as np
import os
import multiprocessing
import matplotlib
matplotlib.use('Agg')
from bornagain import ba_plot as bp
from creat_sample_test import make_particle_lattice_sample

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

def simulate_and_create(radius, height, a, i, scenario_name, description):
    plt = bp.plt
    plt.figure(figsize=(10, 6))

    print(f"[{scenario_name}] Sim {i:03d}: {description}")
    sample = make_particle_lattice_sample(radius=radius, height=height, a=a, b=a)
    simulation = get_simulation2d(sample)
    result = simulation.simulate()
            
    bp.plot_datafield(result)
    plt.title(f"{scenario_name.replace('_', ' ').title()}\n{description}")
    
    # Save in scenario-specific folder
    out_dir = f"./sim/{scenario_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(f"{out_dir}/frame-{i:03d}.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    # Define baseline parameters
    base_radius = 5*nm
    base_height = 4*nm
    base_lattice = 10*nm
    n_frames = 10

    scenarios = {
        "varying_radius": {
            "values": np.linspace(1*nm, 8*nm, n_frames),
            "param": "radius"
        },
        "varying_height": {
            "values":  np.linspace(1*nm, 10*nm, n_frames),
            "param": "height"
        },
        "varying_lattice": {
            "values":  np.linspace(1*nm, 150*nm, n_frames),
            "param": "a"
        }
    }

    # Ensure sim directory exists
    if not os.path.exists('sim'):
        os.makedirs('sim')

    all_tasks = []
    for s_name, config in scenarios.items():
        # Clean scenario directory
        os.system(f"rm -rf ./sim/{s_name} ; mkdir -p ./sim/{s_name}")
        
        for i, val in enumerate(config["values"]):
            print (i, val) 
            # Setup params for this specific frame
            params = {"radius": base_radius, "height": base_height, "a": base_lattice}
            params[config["param"]] = val
            
            desc = f"r={params['radius']/nm:.1f}nm, h={params['height']/nm:.1f}nm, a={params['a']/nm:.4f}nm"
            all_tasks.append((params["radius"], params["height"], params["a"], i, s_name, desc))

    print(f"Starting {len(all_tasks)} simulations across {len(scenarios)} scenarios...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        p.map(wrapper, all_tasks)

    print("\nGenerating videos...")
    for s_name in scenarios.keys():
        video_path = f"./sim/film-{s_name}.mp4"
        cmd = f"ffmpeg -f image2 -r 8 -i ./sim/{s_name}/frame-%03d.png -vcodec mpeg4 -y {video_path}"
        os.system(cmd)
        print(f"Created: {video_path}")
