#!/usr/bin/env python3
import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import bornagain as ba
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac
from creat_sample_test import make_particle_lattice_sample

def run_simulation(params):
    """Worker function for a single simulation."""
    idx, a, b, radius, height, alpha, xi = params
    
    try:
        # Create sample
        sample = make_particle_lattice_sample(
            radius=radius*nm, 
            height=height*nm, 
            a=a*nm, 
            b=b*nm, 
            alpha=alpha*deg, 
            xi=xi*deg
        )
        
        # Setup simulation (Fixed resolution for consistency)
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        n_pixels = 100
        detector = ba.SphericalDetector(n_pixels, -2*deg, 2*deg, n_pixels, 0, 3*deg)
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        
        # Run Simulation
        result = simulation.simulate()
        
        # Convert to numpy (detached from Swig)
        data = dac.asNpArray(result.dataArray()).copy()
        
        # Return data and parameter dict
        p_dict = {
            "a": a, 
            "b": b, 
            "radius": radius, 
            "height": height, 
            "alpha": alpha, 
            "xi": xi
        }
        return data, p_dict
    except Exception as e:
        # Silently fail for individual errors in large batches
        return None

def main():
    # --- CONFIGURATION ---
    N_TOTAL = 10000
    CHUNK_SIZE = 500
    OUTPUT_DIR = "large_dataset"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Preparing to generate {N_TOTAL} samples in chunks of {CHUNK_SIZE}...")
    
    # 1. Pre-generate all random parameters
    all_tasks = []
    for i in range(N_TOTAL):
        # Lattice constants
        a = np.random.uniform(10, 100)
        b = np.random.uniform(10, 100)
        
        # Constraint: radius up to half of lattice constant (with 5% safety margin)
        max_r = min(a, b) / 2.0 * 0.95
        radius = np.random.uniform(1, max_r)
        
        height = np.random.uniform(1, 20)
        alpha = np.random.uniform(60, 120)
        xi = np.random.uniform(0, 90)
        
        all_tasks.append((i, a, b, radius, height, alpha, xi))
        
    # 2. Process in chunks to save memory and progress
    for chunk_start in range(0, N_TOTAL, CHUNK_SIZE):
        chunk_idx = chunk_start // CHUNK_SIZE
        chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:03d}.npz")
        
        if os.path.exists(chunk_file):
            print(f"Chunk {chunk_idx} already exists, skipping...")
            continue
            
        print(f"\nProcessing Chunk {chunk_idx + 1}/{(N_TOTAL // CHUNK_SIZE)}...")
        chunk_tasks = all_tasks[chunk_start : chunk_start + CHUNK_SIZE]
        
        chunk_X = []
        chunk_Y = []
        
        # Using parallel processing
        n_procs = 1  # multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(processes=n_procs) as pool:
            results = list(tqdm(pool.imap(run_simulation, chunk_tasks), total=len(chunk_tasks), desc=f"Chunk {chunk_idx}"))
            
        for res in results:
            if res is not None:
                data, p_dict = res
                chunk_X.append(data)
                chunk_Y.append(p_dict)
        
        # 3. Save Chunk
        if chunk_X:
            np.savez_compressed(chunk_file, X=np.array(chunk_X), Y_dicts=np.array(chunk_Y, dtype=object))
            print(f"Saved chunk to {chunk_file}")

    print("\nGeneration complete!")
    print(f"All data saved in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
