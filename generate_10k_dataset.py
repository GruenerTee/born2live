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
    idx, a, radius, height = params
    
    # Fixed parameters
    b = a
    alpha = 120.0
    xi = 0.0
    
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
        
        # Limit to 1 thread per process to allow efficient multiprocessing
        simulation.options().setNumberOfThreads(1)
        
        # Run Simulation
        result = simulation.simulate()
        
        # --- Peak Finding ---
        # Find top peaks in the scattering pattern
        # sigma=2, threshold=0.001
        found_peaks = ba.FindPeaks(result, 2, "nomarkov", 0.001)
        
        # Extract (x, y) coordinates and pad/truncate to exactly 10 peaks (20 values)
        peak_coords = []
        for p in found_peaks:
            peak_coords.extend([p[0], p[1]])
        
        # Ensure fixed size of 20 (10 peaks * 2 coords)
        if len(peak_coords) < 20:
            peak_coords.extend([0.0] * (20 - len(peak_coords)))
        else:
            peak_coords = peak_coords[:20]
            
        # Convert to numpy (detached from Swig)
        data = dac.asNpArray(result.dataArray()).copy()
        
        # Return data and parameter dict
        p_dict = {
            "a": a, 
            "radius": radius, 
            "height": height,
            "peaks": np.array(peak_coords, dtype=np.float32)
        }
        return data, p_dict
    except Exception as e:
        # Silently fail for individual errors in large batches
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate BornAgain scattering dataset.")
    parser.add_argument("--n_total", type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument("--chunk_size", type=int, default=500, help="Samples per file chunk")
    args = parser.parse_args()

    N_TOTAL = args.n_total
    CHUNK_SIZE = args.chunk_size
    OUTPUT_DIR = "large_dataset"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Preparing to generate {N_TOTAL} samples in chunks of {CHUNK_SIZE}...")
    
    # 1. Pre-generate all random parameters
    all_tasks = []
    for i in range(N_TOTAL):
        # Lattice constants (b=a)
        a = np.random.uniform(10, 100)
        
        # Constraint: radius up to half of lattice constant (with 5% safety margin)
        max_r = a / 2.0 * 0.95
        radius = np.random.uniform(1, max_r)
        
        height = np.random.uniform(1, 20)
        
        all_tasks.append((i, a, radius, height))
        
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
        n_procs = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(processes=max(1, n_procs)) as pool:
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
