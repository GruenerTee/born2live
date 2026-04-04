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

def extract_peaks_as_tensor(result_ba, n_peaks=10):
    """Utility to extract peaks and return a normalized tensor."""
    found_peaks = ba.FindPeaks(result_ba, 2, "nomarkov", 0.001)
    peak_coords = []
    for p in found_peaks:
        peak_coords.extend([p[0], p[1]])
    
    # Pad or truncate to fixed size
    target_len = n_peaks * 2
    if len(peak_coords) < target_len:
        peak_coords.extend([0.0] * (target_len - len(peak_coords)))
    else:
        peak_coords = peak_coords[:target_len]
        
    # Create tensor and normalize (same as trainer)
    tensor = torch.FloatTensor(peak_coords).unsqueeze(0) / 3.0
    return tensor, np.array(peak_coords, dtype=np.float32)

def calculate_r2(target_img, predicted_img):
    """Calculates the R^2 score between two 2D images."""
    y_true = target_img.flatten()
    y_pred = predicted_img.flatten()
    
    # Log-scale comparison is usually more meaningful for scattering
    y_true = np.log1p(y_true)
    y_pred = np.log1p(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0: return 0
    return 1 - (ss_res / ss_tot)

def generate_mini_dataset(file_path, n_samples=20):
    """Generates random samples for testing with shapes and peak features."""
    if os.path.exists(file_path):
        os.remove(file_path)
        
    print(f"Generating {n_samples} samples for mini-dataset with shapes and peak-finding...")
    shapes = ["cylinder", "sphere", "box", "cone"]
    
    for i in tqdm(range(n_samples)):
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
        
        _, peaks_flat = extract_peaks_as_tensor(result)

        params = {
            "radius": r, "height": h, "a": a, "ref": ref, "shape": shape,
            "peaks": peaks_flat
        }
        export_sample(dac.asNpArray(result.dataArray()), params, file_path)

def calculate_metrics(r, h, a, shape):
    """Calculates physical ratios and consistency metrics."""
    # Aspect Ratio: h / 2r
    aspect_ratio = h / (2 * r) if r > 0 else 0
    # Filling Factor: pi * r^2 / a^2 (Square lattice approximation)
    filling_factor = (np.pi * r**2) / (a**2) if a > 0 else 0
    # Overlap Ratio: a / 2r (Must be > 1 to be physically valid)
    overlap_ratio = a / (2 * r) if r > 0 else 0
    
    # Sphere consistency: |h - 2r| (Should be 0 for a perfect sphere)
    consistency = abs(h - 2*r) if shape == "sphere" else None
    
    return {
        "aspect_ratio": aspect_ratio,
        "filling_factor": filling_factor,
        "overlap_ratio": overlap_ratio,
        "consistency": consistency
    }

def run_test(verbose=False, n_tests=3):
    dataset_path = "mini_test_dataset.npz"
    target_keys = ["radius", "height", "a"]
    
    # 1. Generate data
    generate_mini_dataset(dataset_path, n_samples=100)
    
    # 2. Train model (Quickly)
    print("\nTraining Multi-Task Hybrid model (CNN + Peak Features)...")
    model = train_model(dataset_path, target_keys, epochs=100, batch_size=4, verbose=verbose)
    
    if model is None:
        return

    # We need the stats from the dataset to de-normalize predictions
    full_ds = BornAgainDataset(dataset_path, target_keys)
    reg_mean = full_ds.Y_reg_mean.numpy()
    reg_std = full_ds.Y_reg_std.numpy()

    model.eval()
    print(f"\n--- Running {n_tests} Comparison Tests with Validation Simulations ---")

    for test_idx in range(n_tests):
        # 3. Setup Test Parameters (Target)
        test_r = np.random.uniform(1, 5)
        test_h = np.random.uniform(1, 5)
        test_a = np.random.uniform(10, 30)
        test_shape = np.random.choice(["cylinder", "sphere", "box", "cone"])
        if test_shape == "sphere":
            test_h = 2 * test_r
            
        print(f"\n[Test {test_idx+1}] Target: {test_shape} (r={test_r:.2f}, h={test_h:.2f}, a={test_a:.2f})")
        
        # 4. Target Physical Simulation
        sample_target = make_particle_lattice_sample(radius=test_r*nm, height=test_h*nm, a=test_a*nm, b=test_a*nm, shape=test_shape)
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
        simulation_target = ba.ScatteringSimulation(beam, sample_target, detector)
        result_target = simulation_target.simulate()
        img_target = dac.asNpArray(result_target.dataArray())
        
        # 4b. Extract Peaks for Hybrid Input
        peak_tensor, _ = extract_peaks_as_tensor(result_target)

        # 5. AI Inference
        img_tensor = torch.FloatTensor(img_target).unsqueeze(0).unsqueeze(0)
        img_tensor = torch.log1p(img_tensor) # LOG SCALE
        if img_tensor.max() > 0:
            img_tensor /= img_tensor.max()
            
        with torch.no_grad():
            pred_reg, pred_cls = model(img_tensor, peak_tensor)
            
        # De-normalize predicted regression values
        p_reg_standardized = pred_reg.numpy()[0]
        p_reg_physical = (p_reg_standardized * reg_std) + reg_mean
        p_r, p_h, p_a = map(float, p_reg_physical)
        
        predicted_class_idx = torch.argmax(pred_cls, dim=1).item()
        predicted_shape = REV_SHAPE_MAP.get(predicted_class_idx, "unknown")
        
        # 6. Validation Physical Simulation (Using AI Predictions)
        print(f"Running Validation Simulation for AI Prediction...")
        try:
            sample_pred = make_particle_lattice_sample(
                radius=max(0.1, p_r)*nm, 
                height=max(0.1, p_h)*nm, 
                a=max(0.1, p_a)*nm, 
                b=max(0.1, p_a)*nm, 
                shape=predicted_shape
            )
            simulation_pred = ba.ScatteringSimulation(beam, sample_pred, detector)
            result_pred = simulation_pred.simulate()
            img_pred = dac.asNpArray(result_pred.dataArray())
            
            # 7. Calculate Image Match (R^2)
            r2_score = calculate_r2(img_target, img_pred)
        except Exception as e:
            print(f"(!) Validation Simulation failed: {e}")
            r2_score = 0.0
        
        # 8. Output Results
        target_m = calculate_metrics(test_r, test_h, test_a, test_shape)
        pred_m = calculate_metrics(p_r, p_h, p_a, predicted_shape)
        
        print(f"SHAPE    | Target: {test_shape:10s} | Predicted: {predicted_shape:10s}")
        print(f"{'-'*75}")
        print(f"{'Metric':15s} | {'Target':10s} | {'Predicted':10s} | {'Error %':10s}")
        print(f"{'-'*75}")
        
        for i, key in enumerate(target_keys):
            t_val = [test_r, test_h, test_a][i]
            p_val = [p_r, p_h, p_a][i]
            err = abs(t_val - p_val) / t_val * 100
            print(f"{key:15s} | {t_val:10.2f} | {p_val:10.2f} | {err:9.1f}%")
        
        print(f"{'-'*75}")
        print(f"IMAGE MATCH R^2 SCORE (Log-Scale): {r2_score*100:6.2f}%")
        print(f"{'-'*75}")
        
        if r2_score > 0.95:
            print(">>> EXCELLENT MATCH: The predicted structure reproduces the data perfectly.")
        elif r2_score > 0.80:
            print(">>> GOOD MATCH: The predicted structure captures the main features.")
        else:
            print(">>> POOR MATCH: The predicted structure differs significantly from target.")

    print("\nTesting Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AI with Validation Simulations.")
    parser.add_argument("--verbose", action="store_true", help="Show dataflow.")
    args = parser.parse_args()
    run_test(verbose=args.verbose)
