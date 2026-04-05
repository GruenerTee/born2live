#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import bornagain as ba
from bornagain import nm, deg
from bornagain.numpyutil import Arrayf64Converter as dac

from ai.ratio_models import ScatteringCNN_Ratios
from ai.trainer import REV_SHAPE_MAP
from creat_sample_test import make_particle_lattice_sample
from ai_data_utils import load_dataset

SHAPE_MAP = {"cylinder": 0, "sphere": 1, "box": 2, "cone": 3}

class DualDataset(Dataset):
    """A dataset that returns targets for both Absolute and Ratio models."""
    def __init__(self, X, Y_abs, Y_cls):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.X = torch.log1p(self.X)
        if self.X.max() > 0: self.X /= self.X.max()
        
        self.Y_abs_raw = torch.FloatTensor(Y_abs) # [a, r, h]
        
        # Calculate Ratios [a, a/r, r/h]
        a = Y_abs[:, 0]
        r = np.where(Y_abs[:, 1] == 0, 1e-6, Y_abs[:, 1])
        h = np.where(Y_abs[:, 2] == 0, 1e-6, Y_abs[:, 2])
        self.Y_ratio_raw = torch.FloatTensor(np.stack([a, a/r, r/h], axis=1))
        
        # Normalization
        self.abs_mean, self.abs_std = self.Y_abs_raw.mean(0), self.Y_abs_raw.std(0) + 1e-6
        self.ratio_mean, self.ratio_std = self.Y_ratio_raw.mean(0), self.Y_ratio_raw.std(0) + 1e-6
        
        self.Y_abs = (self.Y_abs_raw - self.abs_mean) / self.abs_std
        self.Y_ratio = (self.Y_ratio_raw - self.ratio_mean) / self.ratio_std
        self.Y_cls = torch.LongTensor(Y_cls)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y_abs[idx], self.Y_ratio[idx], self.Y_cls[idx]

def calculate_r2(t, p):
    y_t = np.log1p(t.flatten())
    y_p = np.log1p(p.flatten())
    return 1 - (np.sum((y_t - y_p)**2) / np.sum((y_t - np.mean(y_t))**2))

def train_one_model(model, loader, val_loader, target_idx, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit_reg = nn.MSELoss()
    crit_cls = nn.CrossEntropyLoss()
    for e in range(epochs):
        model.train()
        for img, y_abs, y_ratio, y_cls in loader:
            img, y_cls = img.to(device), y_cls.to(device)
            target = y_abs.to(device) if target_idx == 1 else y_ratio.to(device)
            opt.zero_grad()
            out_r, out_c = model(img)
            loss = crit_reg(out_r, target) + crit_cls(out_c, y_cls)
            loss.backward()
            opt.step()
    return model

def run_comparison(n_samples=300, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Comparison: {n_samples} samples, {epochs} epochs on {device}")
    
    # 1. Load data from existing chunks if available, or generate a few
    dataset_dir = "large_dataset"
    import glob
    chunk_files = glob.glob(os.path.join(dataset_dir, "chunk_*.npz"))
    if not chunk_files:
        print("Please run generate_10k_dataset.py first to create the base data.")
        return

    X_all, Y_abs_all, Y_cls_all = [], [], []
    for f in chunk_files[:2]:
        X, Y = load_dataset(f, ["a", "radius", "height"])
        with np.load(f, allow_pickle=True) as loader:
            cls = [SHAPE_MAP.get(d.get('shape', 'cylinder'), 0) for d in loader['Y_dicts']]
        X_all.append(X); Y_abs_all.append(Y); Y_cls_all.extend(cls)
    
    ds = DualDataset(np.concatenate(X_all)[:n_samples], np.concatenate(Y_abs_all)[:n_samples], Y_cls_all[:n_samples])
    train_set, val_set = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    # 2. Train Both Models
    print("Training Model A (Absolute Parameters)...")
    model_abs = ScatteringCNN_Ratios().to(device)
    train_one_model(model_abs, train_loader, val_loader, 1, epochs, device)
    
    print("Training Model B (Ratio-based Parameters)...")
    model_ratio = ScatteringCNN_Ratios().to(device)
    train_one_model(model_ratio, train_loader, val_loader, 2, epochs, device)

    # 3. Side-by-Side Evaluation
    print("\n" + "="*85)
    print(f"{'Test':5s} | {'Param':10s} | {'Target':10s} | {'Model A (Abs)':15s} | {'Model B (Ratio)':15s}")
    print("="*85)

    model_abs.eval(); model_ratio.eval()
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    det = ba.SphericalDetector(100, -2*deg, 2*deg, 100, 0, 3*deg)

    for i in range(5):
        # Generate random unseen test case
        ta = np.random.uniform(30, 70); tr = np.random.uniform(2, ta/2.2); th = np.random.uniform(2, 12)
        ts = "cylinder"
        sample = make_particle_lattice_sample(radius=tr*nm, height=th*nm, a=ta*nm, b=ta*nm, shape=ts)
        img = dac.asNpArray(ba.ScatteringSimulation(beam, sample, det).simulate().dataArray())
        
        t_img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
        t_img = torch.log1p(t_img); t_img /= (t_img.max() if t_img.max() > 0 else 1)

        with torch.no_grad():
            out_abs, _ = model_abs(t_img)
            out_rat, _ = model_ratio(t_img)
            
        # De-normalize Abs
        p_abs = (out_abs.cpu().numpy()[0] * ds.abs_std.numpy()) + ds.abs_mean.numpy()
        pa_a, pa_r, pa_h = p_abs
        
        # De-normalize Ratio
        p_rat = (out_rat.cpu().numpy()[0] * ds.ratio_std.numpy()) + ds.ratio_mean.numpy()
        pb_a = p_rat[0]; pb_r = pb_a / p_rat[1]; pb_h = pb_r / p_rat[2]

        # Validation R2
        def get_r2(a, r, h):
            try:
                s = make_particle_lattice_sample(radius=max(0.1,r)*nm, height=max(0.1,h)*nm, a=max(1,a)*nm, b=max(1,a)*nm, shape=ts)
                im = dac.asNpArray(ba.ScatteringSimulation(beam, s, det).simulate().dataArray())
                return calculate_r2(img, im)
            except: return 0
        
        r2_a = get_r2(pa_a, pa_r, pa_h)
        r2_b = get_r2(pb_a, pb_r, pb_h)

        print(f"#{i+1:02d}  | Lattice a | {ta:10.2f} | {pa_a:15.2f} | {pb_a:15.2f}")
        print(f"     | Radius  r | {tr:10.2f} | {pa_r:15.2f} | {pb_r:15.2f}")
        print(f"     | Height  h | {th:10.2f} | {pa_h:15.2f} | {pb_h:15.2f}")
        print(f"     | R^2 Match | {'-':10s} | {r2_a*100:14.2f}% | {r2_b*100:14.2f}%")
        print("-" * 85)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Absolute vs Ratio-based AI models.")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to use for training/val")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    args = parser.parse_args()
    
    run_comparison(n_samples=args.samples, epochs=args.epochs)
