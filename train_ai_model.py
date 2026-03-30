#!/usr/bin/env python3
import glob
import os
from torch.utils.data import DataLoader, random_split
from ai.model import ScatteringCNN
from ai.trainer import ModelTrainer, BornAgainDataset

def main():
    # --- CONFIGURATION ---
    # Search for any simulation_dataset.npz in the sim/ directory
    npz_files = glob.glob("sim/**/simulation_dataset.npz", recursive=True)
    
    if not npz_files:
        print("No simulation_dataset.npz found. Run a simulation first.")
        return

    dataset_path = npz_files[0] # Take the first one found
    target_keys = ["radius", "height", "a", "ref"]
    epochs = 10
    batch_size = 4
    
    print(f"Training on: {dataset_path}")
    
    # 1. Prepare Data
    dataset = BornAgainDataset(dataset_path, target_keys)
    if len(dataset) < 2:
        print("Not enough samples for training.")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # 2. Initialize Model and Trainer
    model = ScatteringCNN(num_outputs=len(target_keys))
    trainer = ModelTrainer(model, lr=0.001)
    
    # 3. Run Training
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 4. Save
    trainer.save("bornagain_ai_model_final.pth")

if __name__ == "__main__":
    main()
