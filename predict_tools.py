#!/usr/bin/env python3
import torch
import numpy as np
import os
from ai.model import ScatteringCNN
from physics_models import LatticeModel, SimulationEngine
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac

class StructurePredictor:
    """Utility class for predicting physical parameters from scattering data."""
    def __init__(self, model_path="trained_10k_model.pth"):
        self.target_keys = ["radius", "height", "a", "b", "alpha", "xi"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = ScatteringCNN(num_outputs=len(self.target_keys)).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded AI model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Prediction will use random weights.")

    def _preprocess(self, img_data):
        """Prepare 2D numpy array for the CNN."""
        # Convert to tensor and add batch/channel dims: (1, 1, H, W)
        tensor = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0).to(self.device)
        # Normalize (0-1)
        if tensor.max() > 0:
            tensor /= tensor.max()
        return tensor

    def predict_from_params(self, radius, height, a):
        """Runs a simulation with given params and predicts the structure using AI."""
        print(f"\nTarget Simulation: r={radius:.2f}nm, h={height:.2f}nm, a={a:.2f}nm")
        
        # 1. Run physical simulation
        phys_model = LatticeModel(radius=radius*nm, height=height*nm, a=a*nm, b=a*nm)
        sample = phys_model.create_sample()
        result = SimulationEngine.run_scattering_2d(sample, n_pixels=100)
        
        # 2. Extract data and predict
        img_data = dac.asNpArray(result.dataArray())
        input_tensor = self._preprocess(img_data)
        
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()[0]
            
        print("--- AI Prediction Results ---")
        for i, key in enumerate(self.target_keys):
            print(f"{key:10s}: {prediction[i]:.4f}")
        return prediction

    def predict_from_csv(self, csv_path):
        """Reads a CSV file containing 2D scattering data and predicts the structure."""
        print(f"\nPredicting from CSV: {csv_path}")
        try:
            # Assumes the CSV is a 2D matrix (e.g., 100x100)
            img_data = np.loadtxt(csv_path, delimiter=',')
            
            # Ensure it matches the expected 100x100 resolution
            if img_data.shape != (100, 100):
                print(f"Reshaping/resizing data from {img_data.shape} to (100, 100)...")
                # Simple flatten/reshape if it's 10000 long or similar
                img_data = img_data.reshape(100, 100)

            input_tensor = self._preprocess(img_data)
            
            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().numpy()[0]
                
            print("--- AI Prediction Results ---")
            for i, key in enumerate(self.target_keys):
                print(f"{key:10s}: {prediction[i]:.4f}")
            return prediction
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None

if __name__ == "__main__":
    import sys
    predictor = StructurePredictor()
    
    # Simple CLI handling for quick tests
    if len(sys.argv) == 4:
        # Example: python3 predict_tools.py 5.0 10.0 20.0
        r, h, a = map(float, sys.argv[1:4])
        predictor.predict_from_params(r, h, a)
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        # Example: python3 predict_tools.py my_data.csv
        predictor.predict_from_csv(sys.argv[1])
    else:
        print("Usage:")
        print("  1. Predict from params: python3 predict_tools.py <radius> <height> <a>")
        print("  2. Predict from CSV:    python3 predict_tools.py <data.csv>")
