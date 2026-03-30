#!/usr/bin/env python3
import os
import torch
import numpy as np
from tqdm import tqdm
from ai_data_utils import export_sample, load_dataset
from train_ai_model import train_model, ScatteringCNN, BornAgainDataset
from creat_sample_test import make_particle_lattice_sample
import bornagain as ba
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac

def generate_mini_dataset(file_path, n_samples=10):
    """Generates a few random samples for testing the pipeline."""
    if os.path.exists(file_path):
        os.remove(file_path)
        
    print(f"Generating {n_samples} samples for mini-dataset...")
    for i in range(n_samples):
        # Random parameters
        r = np.random.uniform(1, 5)
        h = np.random.uniform(1, 5)
        a = np.random.uniform(10, 30)
        ref = 0.0006
        
        sample = make_particle_lattice_sample(radius=r*nm, height=h*nm, a=a*nm, b=a*nm)
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        result = simulation.simulate()
        
        # Save to dataset
        params = {"radius": r, "height": h, "a": a, "ref": ref}
        export_sample(dac.asNpArray(result.dataArray()), params, file_path)

def run_test():
    dataset_path = "mini_test_dataset.npz"
    target_keys = ["radius", "height", "a"]
    
    # 1. Generate data
    generate_mini_dataset(dataset_path, n_samples=10)
    
    # 2. Train model (Quickly)
    print("\nTraining model on mini-dataset (5 epochs)...")
    model = train_model(dataset_path, target_keys, epochs=5, batch_size=2)
    
    if model is None:
        return

    # 3. Test Inference
    print("\nTesting Inference on a NEW simulation...")
    test_r, test_h, test_a = 3.5, 2.5, 25.0
    print(f"Target Parameters: radius={test_r}, height={test_h}, a={test_a}")
    
    sample = make_particle_lattice_sample(radius=test_r*nm, height=test_h*nm, a=test_a*nm, b=test_a*nm)
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    result = simulation.simulate()
    
    # Pre-process image for model
    img_data = dac.asNpArray(result.dataArray())
    img_tensor = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0) # (Batch, Channel, H, W)
    if img_tensor.max() > 0:
        img_tensor /= img_tensor.max()
        
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor).numpy()[0]
    
    targets = {"radius": test_r, "height": test_h, "a": test_a}
    
    print("\n--- RESULTS ---")
    for i, key in enumerate(target_keys):
        print(f"{key.capitalize()}: Target={targets[key]:.2f}, Predicted={prediction[i]:.2f}")
    print("----------------")
    print("Note: With only 10 samples and 5 epochs, accuracy will be low, but the pipeline is verified!")

if __name__ == "__main__":
    run_test()
