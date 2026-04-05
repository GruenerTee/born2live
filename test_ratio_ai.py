#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import bornagain as ba
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac

from ai_data_utils import load_dataset
from ai.ratio_models import ScatteringCNN_Ratios
from ai.trainer import REV_SHAPE_MAP
from creat_sample_test import make_particle_lattice_sample

def calculate_r2(target_img, predicted_img):
    """Calculates the R^2 score between two 2D images in log-scale."""
    y_true = np.log1p(target_img.flatten())
    y_pred = np.log1p(predicted_img.flatten())
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def run_ratio_test(model_path="trained_ratio_model.pth", n_tests=5):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # 1. Load Model and Stats
    print(f"Loading ratio-based model from {model_path}...")
    checkpoint = torch.load(model_path)
    
    # Initialize architecture
    model = ScatteringCNN_Ratios(num_ratios=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load normalization stats
    stats = checkpoint['stats']
    reg_mean = stats['reg_mean'].numpy()
    reg_std = stats['reg_std'].numpy()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n--- Running {n_tests} Ratio-based Validation Tests ---")

    for test_idx in range(n_tests):
        # 2. Setup Random Test Parameters
        test_a = np.random.uniform(20, 80)
        # Ensure radius < a/2
        test_r = np.random.uniform(2, test_a/2.1)
        test_h = np.random.uniform(2, 15)
        test_shape = np.random.choice(["cylinder", "sphere", "box", "cone"])
        
        if test_shape == "sphere":
            test_h = 2 * test_r

        print(f"\n[Test {test_idx+1}] Target: {test_shape} (a={test_a:.2f}, r={test_r:.2f}, h={test_h:.2f})")
        
        # 3. Target Simulation (Ground Truth)
        sample_target = make_particle_lattice_sample(radius=test_r*nm, height=test_h*nm, a=test_a*nm, b=test_a*nm, shape=test_shape)
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        detector = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)
        sim_target = ba.ScatteringSimulation(beam, sample_target, detector)
        img_target = dac.asNpArray(sim_target.simulate().dataArray())

        # 4. AI Inference
        img_tensor = torch.FloatTensor(img_target).unsqueeze(0).unsqueeze(0).to(device)
        img_tensor = torch.log1p(img_tensor)
        if img_tensor.max() > 0: img_tensor /= img_tensor.max()
            
        with torch.no_grad():
            pred_ratios, pred_cls = model(img_tensor)
            
        # 5. De-normalize and Convert Ratios to Physical Parameters
        # Predicted standardized: [a_std, ar_std, rh_std]
        p_std = pred_ratios.cpu().numpy()[0]
        # De-normalize: [a, a/r, r/h]
        p_ratios = (p_std * reg_std) + reg_mean
        
        pred_a = float(p_ratios[0])
        pred_a_r = float(p_ratios[1])
        pred_r_h = float(p_ratios[2])
        
        # Convert ratios back to r and h
        # r = a / (a/r)
        p_r = pred_a / pred_a_r if pred_a_r != 0 else 1.0
        # h = r / (r/h)
        p_h = p_r / pred_r_h if pred_r_h != 0 else 1.0
        
        predicted_class_idx = torch.argmax(pred_cls, dim=1).item()
        predicted_shape = REV_SHAPE_MAP.get(predicted_class_idx, "unknown")

        # 6. Validation Simulation (Using AI Predictions)
        print(f"Prediction: {predicted_shape} (a={pred_a:.2f}, r={p_r:.2f}, h={p_h:.2f})")
        try:
            sample_pred = make_particle_lattice_sample(
                radius=max(0.1, p_r)*nm, 
                height=max(0.1, p_h)*nm, 
                a=max(0.1, pred_a)*nm, 
                b=max(0.1, pred_a)*nm, 
                shape=predicted_shape
            )
            sim_pred = ba.ScatteringSimulation(beam, sample_pred, detector)
            img_pred = dac.asNpArray(sim_pred.simulate().dataArray())
            r2_score = calculate_r2(img_target, img_pred)
        except Exception as e:
            print(f"(!) Validation Simulation failed: {e}")
            r2_score = 0.0

        # 7. Output Results
        print(f"{'-'*60}")
        print(f"{'Variable':15s} | {'Target':10s} | {'Predicted':10s} | {'Error %'}")
        print(f"{'-'*60}")
        print(f"{'Lattice (a)':15s} | {test_a:10.2f} | {pred_a:10.2f} | {abs(test_a-pred_a)/test_a*100:7.1f}%")
        print(f"{'Radius (r)':15s} | {test_r:10.2f} | {p_r:10.2f} | {abs(test_r-p_r)/test_r*100:7.1f}%")
        print(f"{'Height (h)':15s} | {test_h:10.2f} | {p_h:10.2f} | {abs(test_h-p_h)/test_h*100:7.1f}%")
        print(f"{'-'*60}")
        print(f"IMAGE MATCH R^2 SCORE: {r2_score*100:6.2f}%")
        print(f"{'-'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ratio-based AI model.")
    parser.add_argument("--model", type=str, default="trained_ratio_model.pth", help="Path to model file.")
    parser.add_argument("--tests", type=int, default=5, help="Number of tests to run.")
    args = parser.parse_args()
    
    run_ratio_test(model_path=args.model, n_tests=args.tests)
