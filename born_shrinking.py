#!/usr/bin/env python3
"""
Example of specular reflectometry simulation with a shrinking number of layers.
The sample consists of alternating Ti and Ni layers, where the number of
repetitions (N) is varied from 10 down to 1.
"""
import bornagain as ba
import multiprocessing
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from tqdm import tqdm
from bornagain import ba_plot as bp, deg, angstrom, nm
from born_pygame_viz import draw_stack
from creat_sample_test import make_layer_stack

def get_simulation(sample):
    n = 500
    scan = ba.AlphaScan(n, 2*deg/n, 2*deg)
    scan.setWavelength(1.54*angstrom)
    return ba.SpecularSimulation(scan, sample)

# --- Animation Setup ---

def get_simulation2d(sample):
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    n = 200
    detector = ba.SphericalDetector(n, -2*deg, 2*deg, n, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation

def wrapper (args):
    simulate_and_creat(*args)

def simulate_and_creat(n_rep, factor, Top, i, path):
    plt = bp.plt
    plt.figure(figsize=(10, 6))

    # print(f"{i}th Simulation for n_repetitions = {n_rep}")
    sample = make_layer_stack(Top, factor, n_rep)
    simulation = get_simulation(sample)
    result = simulation.simulate()
            
    bp.plot_datafield(result)
    plt.title(f"Shrinking Stack\nRepetitions: {n_rep}, Top: {Top}, Factor: {factor:.2f}")
    
    frame_path = os.path.join(path, f"frame-{i:03d}.png")
    plt.savefig(frame_path, dpi=150)
    plt.close()
    
    # Save visualization using the updated draw_stack
    draw_stack(n_rep, Top, factor, i, path=path)

if __name__ == '__main__':
    # Setup directory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d__T%H:%MZ")
    path = f"sim/shrinking_{timestamp}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Build tasks
    tasks = []
    i = 0
    # Simulation properties for description/filename
    n_rep_range = range(1, -1, -1)
    top_layers = ['Ti', 'Ni']
    factors = np.arange(3, 0.1, -0.1)

    for n_rep in n_rep_range:
        for Top in top_layers:
            for factor in factors:
                tasks.append((n_rep, factor, Top, i, path))
                i += 1

    print(f"Starting {len(tasks)} simulations...")
    print(f"Results will be saved to: {path}")
    
    with multiprocessing.Pool(processes=4) as p:
        list(tqdm(p.imap(wrapper, tasks), total=len(tasks), desc="Simulations"))

    print("\nGenerating video...")
    video_path = f"{path}film-shrinking-combined.mp4"
    # Combine simulation and stack viz side-by-side
    cmd = (
        f"ffmpeg -r 10 -i {path}frame-%03d.png -r 10 -i {path}viz-%03d.png "
        f"-filter_complex \"[0:v]scale=-1:800[v0];[1:v]scale=-1:800[v1];[v0][v1]hstack\" "
        f"-vcodec mpeg4 -y {video_path} >/dev/null 2>&1"
    )
    os.system(cmd)
    print(f"Created: {video_path}")
