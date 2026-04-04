#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from ai_data_utils import export_sample, load_dataset
from train_ai_model import train_model
from ai.model import ScatteringCNN
from ai.trainer import BornAgainDataset, REV_SHAPE_MAP
from creat_sample_test import make_particle_lattice_sample
import bornagain as ba
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac
from viz.sample_bridge import visualize_ba_sample

def generate_mini_dataset(file_path, n_samples=20):
    """Generates a random samples for testing with shapes."""
    if os.path.exists(file_path):
        os.remove(file_path)
        
    print(f"Generating {n_samples} samples for mini-dataset with random shapes...")
    shapes = ["cylinder", "sphere", "box", "cone"]
    
    for i in range(n_samples):
        r = np.random.uniform(1, 5)
        h = np.random.uniform(1, 5)
        a = np.random.uniform(10, 30)
        shape = np.random.choice(shapes)
        ref = 0.0006
        
        sample = make_particle_lattice_sample(radius=r*nm, height=h*nm, a=a*nm, b=a*nm, shape=shape)
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        result = simulation.simulate()
        
        # IMPORTANT: Include shape in the params dictionary
        params = {"radius": r, "height": h, "a": a, "ref": ref, "shape": shape}
        export_sample(dac.asNpArray(result.dataArray()), params, file_path)

def run_test(verbose=False):
    dataset_path = "mini_test_dataset.npz"
    target_keys = ["radius", "height", "a"]
    
    # 1. Generate data
    generate_mini_dataset(dataset_path, n_samples=20)
    
    # 2. Train model (Quickly)
    print("\nTraining Multi-Task model (Regression + Shape Classification)...")
    model = train_model(dataset_path, target_keys, epochs=10, batch_size=4, verbose=verbose)
    
    if model is None:
        return

    # 3. Test Inference
    print("\nTesting Inference on a NEW simulation...")
    test_r, test_h, test_a = 4.0, 3.0, 22.0
    test_shape = np.random.choice(["cylinder", "sphere", "box", "cone"])
    print(f"Target Parameters: radius={test_r}, height={test_h}, a={test_a}, shape={test_shape}")
    
    sample = make_particle_lattice_sample(radius=test_r*nm, height=test_h*nm, a=test_a*nm, b=test_a*nm, shape=test_shape)
    visualize_ba_sample(sample, "test_sample_viz.png")
    
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    result = simulation.simulate()
    
    img_data = dac.asNpArray(result.dataArray())
    img_tensor = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0)
    if img_tensor.max() > 0:
        img_tensor /= img_tensor.max()
        
    # Predict
    model.eval()
    with torch.no_grad():
        pred_reg, pred_cls = model(img_tensor)
        
    prediction_reg = pred_reg.numpy()[0]
    # Get shape class with highest probability
    predicted_class_idx = torch.argmax(pred_cls, dim=1).item()
    predicted_shape = REV_SHAPE_MAP.get(predicted_class_idx, "unknown")
    
    targets = {"radius": test_r, "height": test_h, "a": test_a}
    
    print("\n--- RESULTS ---")
    print(f"Sample visualization saved to: test_sample_viz.png")
    print(f"SHAPE: Target={test_shape}, Predicted={predicted_shape}")
    for i, key in enumerate(target_keys):
        print(f"{key.capitalize()}: Target={targets[key]:.2f}, Predicted={prediction_reg[i]:.2f}")
    print("----------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AI with Shape Recognition.")
    parser.add_argument("--verbose", action="store_true", help="Show dataflow.")
    args = parser.parse_args()
    run_test(verbose=args.verbose)
