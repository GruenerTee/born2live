#!/usr/bin/env python3
import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader, random_split
from ai.model import ScatteringCNN
from ai.trainer import ModelTrainer, BornAgainDataset

def train_model(dataset_path, target_keys, epochs=10, batch_size=4, lr=0.001, verbose=False):
    """
    Main training function that returns both model and trainer.
    """
    print(f"Loading dataset: {dataset_path}")
    
    # 1. Prepare Data
    dataset = BornAgainDataset(dataset_path, target_keys)
    if len(dataset) < 2:
        print("Not enough samples for training.")
        return None, None
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # 2. Initialize Model and Trainer
    model = ScatteringCNN(num_reg_outputs=len(target_keys), verbose=verbose)
    trainer = ModelTrainer(model, lr=lr)
    
    # 3. Run Training
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    return model, trainer

def main():
    parser = argparse.ArgumentParser(description="Train the BornAgain AI model.")
    parser.add_argument("--verbose", action="store_true", help="Show dataflow during training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    args = parser.parse_args()
    
    # ... (dataset search logic remains same) ...
    npz_files = glob.glob("sim/**/simulation_dataset.npz", recursive=True)
    if not npz_files:
        npz_files = glob.glob("*.npz")
    
    if not npz_files:
        print("No .npz dataset found. Run a simulation first.")
        return

    dataset_path = npz_files[0]
    target_keys = ["radius", "height", "a", "ref"]
    
    # Run Training
    model, trainer = train_model(
        dataset_path, 
        target_keys, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        verbose=args.verbose
    )
    
    # 4. Save
    if trainer:
        save_path = "bornagain_ai_model_final.pth"
        trainer.save(save_path)


if __name__ == "__main__":
    main()
