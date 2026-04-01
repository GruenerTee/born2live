#!/usr/bin/env python3
import glob
import os
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from ai.model import ScatteringCNN
from ai.trainer import ModelTrainer, BornAgainDataset

def train_large_scale():
    # 1. Configuration
    dataset_dir = "large_dataset"
    model_output = "trained_10k_model.pth"
    # These must match the keys saved in generate_10k_dataset.py
    target_keys = ["a", "radius", "height"]
    
    batch_size = 32
    epochs = 50
    learning_rate = 0.0005

    # 2. Find and Load all chunks
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.npz")))
    if not chunk_files:
        print(f"Error: No chunk files found in {dataset_dir}/. Please run generate_10k_dataset.py first.")
        return

    print(f"Found {len(chunk_files)} chunks. Loading datasets...")
    
    # Create a list of individual datasets
    individual_datasets = []
    for f in chunk_files:
        try:
            ds = BornAgainDataset(f, target_keys)
            individual_datasets.append(ds)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    if not individual_datasets:
        print("No valid data loaded. Aborting.")
        return

    # Combine into one large dataset
    full_dataset = ConcatDataset(individual_datasets)
    print(f"Total samples: {len(full_dataset)}")

    # 3. Split into Train/Val (90% / 10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

    # 4. Initialize Model and Trainer
    # Input channel is 1 (grayscale scattering image)
    model = ScatteringCNN(num_outputs=len(target_keys))
    trainer = ModelTrainer(model, lr=learning_rate)
    
    # 5. Execute Training
    print(f"\nStarting large-scale training for {epochs} epochs...")
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # 6. Save final weights
    trainer.save(model_output)
    print(f"\nSuccess! Final model saved as: {model_output}")

if __name__ == "__main__":
    train_large_scale()
