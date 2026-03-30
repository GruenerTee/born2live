#!/usr/bin/env python3
import bornagain as ba
from bornagain import deg, nm, angstrom
import numpy as np
import os
import datetime
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from bornagain import ba_plot as bp
from bornagain.numpyutil import Arrayf64Converter as dac
from physics_models import LatticeModel, SimulationEngine
from born_lattice_pygame_viz import visualize_lattice_pygame
from ai_data_utils import export_sample, load_dataset
from viz.movie_maker import MovieMaker

def wrapper(args):
    simulate_and_create(*args)

def simulate_and_create(radius, height, ref, a, i, scenario_name, description, path):
    plt = bp.plt
    plt.figure(figsize=(10, 6))

    # Initialize the OOP Model
    model = LatticeModel(radius=radius, height=height, a=a, b=a, ref_idx=ref)
    sample = model.create_sample()
    
    # Run Simulation through Engine (using 200x200 resolution as before)
    result = SimulationEngine.run_scattering_2d(sample, n_pixels=200)
    
    # Export for AI training
    params = {
        "radius": float(radius/nm), 
        "height": float(height/nm), 
        "a": float(a/nm), 
        "ref": float(ref)
    }
    dataset_path = os.path.join(path, "simulation_dataset.npz")
    export_sample(dac.asNpArray(result.dataArray()), params, dataset_path)
            
    bp.plot_datafield(result)
    plt.title(f"{scenario_name.replace('_', ' ').title()}\n{description}")
    
    out_dir = f"{path}{scenario_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(f"{out_dir}/frame-{i:03d}.png", dpi=150)
    plt.close()
    
    # Pass parameters directly from model to visualization
    visualize_lattice_pygame(f"{out_dir}/viz-{i:03d}.png", 
                             radius=model.radius, height=model.height, 
                             a=model.a, b=model.b)

if __name__ == '__main__':
    # Define baseline parameters
    time = datetime.datetime.now()
    base_radius = 5*nm
    base_height = 4*nm
    base_lattice = 10*nm
    n_frames = 10
    ref_values = [0.0006]

    all_scenarios = {
        "varying_radius": {"radius": np.linspace(1*nm, 5*nm, n_frames)},
        "varying_height": {"height": np.linspace(1*nm, 10*nm, n_frames)},
        "varying_lattice": {"a": np.linspace(10*nm, 150*nm, n_frames)},
        "radius_and_height": {
            "radius": np.linspace(1*nm, 7*nm, n_frames),
            "height": np.linspace(1*nm, 15*nm, n_frames),
        }
    }
    
    # Simulation settings
    active_scenario_names = ["radius_and_height"]
    n_frames_to_run = 3 # Overriding n_frames for a quick test as before

    for ref in ref_values:
        path = 'sim/' + str(time.strftime("%Y-%m-%d__T%H:%MZ")) + '/' + str(ref) + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        all_tasks = []
        for s_name in active_scenario_names:
            config = all_scenarios[s_name]
            os.system(f"rm -rf {path}{s_name} ; mkdir -p {path}{s_name}")
            
            for i in range(n_frames_to_run):
                params = {"radius": base_radius, "height": base_height, "a": base_lattice}
                for param_name, values in config.items():
                    params[param_name] = values[i]
                
                desc = f"r={params['radius']/nm:.1f}nm, h={params['height']/nm:.1f}nm, a={params['a']/nm:.4f}nm"
                all_tasks.append((params["radius"], params["height"], ref, params["a"], i, s_name, desc, path))

        print(f"Starting {len(all_tasks)} simulations across {len(active_scenario_names)} scenarios...")
        for task in tqdm(all_tasks, desc="Simulations"):
            wrapper(task)

        print("\nGenerating videos...")
        mm = MovieMaker(frame_rate=8, height=800)
        for s_name in tqdm(active_scenario_names, desc="Video generation"):
            video_path = f"{path}film-{s_name}.mp4"
            mm.create_side_by_side(f"{path}{s_name}", video_path)
            print(f"Created: {video_path}")

        # Dataset verification
        print("\nVerifying AI Dataset...")
        dataset_path = os.path.join(path, "simulation_dataset.npz")
        if os.path.exists(dataset_path):
            target_keys = ["radius", "height", "a", "ref", "xi"]
            X, Y = load_dataset(dataset_path, target_keys)
            print(f"X (Images) shape: {X.shape}, Y (Params) shape: {Y.shape}")
