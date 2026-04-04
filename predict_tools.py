#!/usr/bin/env python3
import torch
import numpy as np
import os
import bornagain as ba
from ai.model import ScatteringCNN
from ai.trainer import REV_SHAPE_MAP
from physics_models import LatticeModel, SimulationEngine
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac

class StructurePredictor:
    """Utility class for predicting physical parameters from scattering data using a hybrid CNN+Peaks model."""
    def __init__(self, model_path="trained_10k_model.pth"):
        self.target_keys = ["a", "radius", "height"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = None
        
        # Load Model
        self.model = ScatteringCNN(num_reg_outputs=len(self.target_keys)).to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Support both new (dict) and old (state_dict only) checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.stats = checkpoint.get('stats')
                print(f"Loaded AI model and stats from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded AI model (legacy format) from {model_path}")
            self.model.eval()
        else:
            print(f"Warning: Model file {model_path} not found. Prediction will use random weights.")

    def _preprocess_img(self, img_data):
        """Prepare 2D numpy array for the CNN with log-scale normalization."""
        tensor = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor = torch.log1p(tensor) # Match trainer preprocessing
        if tensor.max() > 0:
            tensor /= tensor.max()
        return tensor

    def _de_normalize(self, prediction_vec):
        """Converts model's standardized output back to nanometers."""
        if self.stats and 'reg_mean' in self.stats:
            mean = self.stats['reg_mean'].numpy()
            std = self.stats['reg_std'].numpy()
            return (prediction_vec * std) + mean
        return prediction_vec

    def _extract_peaks(self, result_ba):
        """Extracts normalized peak coordinates from a BornAgain result."""
        found_peaks = ba.FindPeaks(result_ba, 2, "nomarkov", 0.001)
        peak_coords = []
        for p in found_peaks:
            peak_coords.extend([p[0], p[1]])
        
        # Pad/Truncate to 20 values (10 peaks)
        if len(peak_coords) < 20:
            peak_coords.extend([0.0] * (20 - len(peak_coords)))
        else:
            peak_coords = peak_coords[:20]
            
        return torch.FloatTensor(peak_coords).unsqueeze(0).to(self.device) / 3.0

    def predict_from_params(self, radius, height, a):
        """Runs a simulation with given params and predicts the structure using AI."""
        print(f"\nTarget Simulation: r={radius:.2f}nm, h={height:.2f}nm, a={a:.2f}nm")
        
        # 1. Run physical simulation
        phys_model = LatticeModel(radius=radius*nm, height=height*nm, a=a*nm, b=a*nm)
        sample = phys_model.create_sample()
        result = SimulationEngine.run_scattering_2d(sample, n_pixels=100)
        
        # 2. Extract inputs
        img_data = dac.asNpArray(result.dataArray())
        input_img = self._preprocess_img(img_data)
        input_peaks = self._extract_peaks(result)
        
        with torch.no_grad():
            pred_reg, pred_cls = self.model(input_img, input_peaks)
            prediction = pred_reg.cpu().numpy()[0]
            # De-normalize back to physical units
            prediction = self._de_normalize(prediction)
            
            predicted_class_idx = torch.argmax(pred_cls, dim=1).item()
            predicted_shape = REV_SHAPE_MAP.get(predicted_class_idx, "unknown")
            
        print("--- AI Prediction Results ---")
        print(f"Predicted Shape: {predicted_shape}")
        for i, key in enumerate(self.target_keys):
            print(f"{key:10s}: {prediction[i]:.4f}")
        return prediction

    def predict_from_csv(self, csv_path):
        """Reads a CSV file containing 2D scattering data and predicts the structure."""
        print(f"\nPredicting from CSV: {csv_path}")
        try:
            img_data = np.loadtxt(csv_path, delimiter=',')
            if img_data.shape != (100, 100):
                img_data = img_data.reshape(100, 100)

            input_img = self._preprocess_img(img_data)
            
            # For CSV data, we don't have a ba.Result object. 
            # In a real scenario, we would need to manually find peaks in the numpy array.
            # For now, we use a zero-fallback as handled by the model.
            input_peaks = torch.zeros(1, 20).to(self.device)
            
            with torch.no_grad():
                pred_reg, pred_cls = self.model(input_img, input_peaks)
                prediction = pred_reg.cpu().numpy()[0]
                prediction = self._de_normalize(prediction) # Physical units
                
                predicted_class_idx = torch.argmax(pred_cls, dim=1).item()
                predicted_shape = REV_SHAPE_MAP.get(predicted_class_idx, "unknown")
                
            print("--- AI Prediction Results ---")
            print(f"Predicted Shape: {predicted_shape}")
            for i, key in enumerate(self.target_keys):
                print(f"{key:10s}: {prediction[i]:.4f}")
            return prediction
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None

if __name__ == "__main__":
    import sys
    predictor = StructurePredictor()
    
    if len(sys.argv) == 4:
        r, h, a = map(float, sys.argv[1:4])
        predictor.predict_from_params(r, h, a)
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        predictor.predict_from_csv(sys.argv[1])
    else:
        print("Usage:")
        print("  1. Predict from params: python3 predict_tools.py <radius> <height> <a>")
        print("  2. Predict from CSV:    python3 predict_tools.py <data.csv>")
