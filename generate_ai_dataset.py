#!/usr/bin/env python3
import bornagain as ba
from bornagain import deg, nm, angstrom
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
from creat_sample_test import make_particle_lattice_sample

# As requested for data conversion
# Note: In recent BornAgain versions, results can often be converted directly, 
# but we'll use the logic provided to ensure compatibility.
def result_to_numpy(result):
    """Converts BornAgain simulation result to a numpy array."""
    # Datafield has a .array() method in modern BA, 
    # but we can also use the converter if preferred.
    return result.array()

def get_simulation2d(sample, n_pixels=128):
    """Standardized 2D simulation for AI training."""
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    # Using a fixed pixel size for the AI input
    detector = ba.SphericalDetector(n_pixels, -2*deg, 2*deg, n_pixels, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation

def simulate_task(params):
    """Task for a single simulation run."""
    radius, height, a, b, alpha, xi = params
    
    # Generate sample
    sample = make_particle_lattice_sample(
        radius=radius*nm, 
        height=height*nm, 
        a=a*nm, 
        b=b*nm, 
        alpha=alpha*deg, 
        xi=xi*deg
    )
    
    # Run simulation
    simulation = get_simulation2d(sample)
    result = simulation.simulate()
    
    # Convert to numpy
    data = result_to_numpy(result)
    
    # Return (Input_X, Output_Y)
    # X: 2D image data
    # Y: Physical parameters
    return data, np.array([radius, height, a, b, alpha, xi])

def generate_dataset(n_samples=50, output_file="bornagain_ai_dataset.npz", n_pixels=128):
    """
    Generates a dataset of 2D simulations and their corresponding physical parameters.
    
    Args:
        n_samples (int): Number of simulations to run.
        output_file (str): Path to save the compressed .npz file.
        n_pixels (int): Resolution of the 2D detector.
        
    Returns:
        tuple: (X, Y) where X is the image data and Y is the parameter array.
    """
    print(f"Generating dataset with {n_samples} samples...")
    
    # Define parameter ranges for sampling (Randomized for AI training)
    tasks = []
    for _ in range(n_samples):
        radius = np.random.uniform(1, 10)
        height = np.random.uniform(1, 15)
        a = np.random.uniform(10, 100)
        b = a # Keeping it square for this example, but can vary
        alpha = 120 # Fixed or np.random.uniform(60, 120)
        xi = 0      # Fixed or np.random.uniform(0, 90)
        tasks.append((radius, height, a, b, alpha, xi))

    # Run simulations in parallel
    X_data = []
    Y_params = []
    
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_task, tasks), total=n_samples, desc="Simulating Dataset"))
        
        for data, params in results:
            X_data.append(data)
            Y_params.append(params)

    # Convert to final numpy arrays
    X = np.array(X_data)
    Y = np.array(Y_params)
    
    # Save the dataset
    if output_file:
        print(f"Saving dataset to {output_file}...")
        np.savez_compressed(output_file, X=X, Y=Y, parameter_names=["radius", "height", "a", "b", "alpha", "xi"])
    
    return X, Y

if __name__ == '__main__':
    # Default execution when run as a script
    generate_dataset(n_samples=50, output_file="bornagain_ai_dataset.npz")
    print("Done. Dataset ready for training.")
