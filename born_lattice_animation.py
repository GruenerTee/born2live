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

    # Dictionary of all available scenarios
    # A scenario can change one or more parameters over n_frames
    all_scenarios = {
        "varying_radius": {
            "radius": np.linspace(1*nm, 5*nm, n_frames),
        },
        "varying_height": {
            "height": np.linspace(1*nm, 10*nm, n_frames),
        },
        "varying_lattice": {
            "a": np.linspace(10*nm, 150*nm, n_frames),
        },
        "radius_and_height": {
            "radius": np.linspace(1*nm, 7*nm, n_frames),
            "height": np.linspace(1*nm, 15*nm, n_frames),
        },
        "radius_and_lattice": {
            "radius": np.linspace(1*nm, 8*nm, n_frames),
            "a": np.linspace(10*nm, 100*nm, n_frames),
        }
    }

    # --- CONTROL PANEL: Select which scenarios to run ---
    # List the keys from all_scenarios that you want to simulate
    active_scenario_names = ["radius_and_height"] # "varying_lattice"
    # ----------------------------------------------------

    for ref in ref_values:
        path='sim/'+ str(time.strftime("%Y-%m-%d__T%H:%MZ"))+'/'+str(ref)+'/'
    # Ensure sim directory exists
        if not os.path.exists(path):
            os.makedirs(path )

        all_tasks = []
        for s_name in active_scenario_names:
            config = all_scenarios[s_name]
            # Clean scenario directory
            os.system(f"rm -rf {path}{s_name} ; mkdir -p {path}{s_name}")
            
            # For each frame i, determine the parameters
            for i in range(n_frames):
                # Setup params for this specific frame starting with baselines
                params = {"radius": base_radius, "height": base_height, "a": base_lattice}
                
                # Update with scenario-specific values for this frame
                for param_name, values in config.items():
                    if len(values) == 1:
                        params[param_name] = values[0]
                    else:
                        params[param_name] = values[i]
                
                desc = f"r={params['radius']/nm:.1f}nm, h={params['height']/nm:.1f}nm, a={params['a']/nm:.4f}nm"
                all_tasks.append((params["radius"], params["height"], ref, params["a"], i, s_name, desc, path))

        print(f"Starting {len(all_tasks)} simulations across {len(active_scenario_names)} scenarios...")
        with multiprocessing.Pool(processes=1) as p: 
            list(tqdm(p.imap(wrapper, all_tasks), total=len(all_tasks), desc="Simulations", leave=True))

        print("\nGenerating videos...")

        for s_name in tqdm(active_scenario_names, desc="Video generation"):
            video_path = f"{path}film-{s_name}.mp4"
            # Combine simulation and pygame viz side-by-side, scaling them to the same height
            cmd = f"ffmpeg -r 8 -i {path}{s_name}/frame-%03d.png -r 8 -i {path}{s_name}/viz-%03d.png -filter_complex \"[0:v]scale=-1:800[v0];[1:v]scale=-1:800[v1];[v0][v1]hstack\" -vcodec mpeg4 -y {video_path} >/dev/null 2>&1"
            os.system(cmd)
            print(f"Created: {video_path}")
