#!/usr/bin/env python3
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_random_samples(dataset_dir="large_dataset", n_samples=6):
    """
    Loads random samples from the chunked dataset and displays them 
    with their corresponding physical parameters.
    """
    # 1. Find all chunks
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.npz")))
    if not chunk_files:
        print(f"Error: No chunk files found in {dataset_dir}/. Run generate_10k_dataset.py first.")
        return

    print(f"Found {len(chunk_files)} chunks. Picking random samples...")

    # 2. Pick a few random samples
    # We'll pick random chunks and then random samples within those chunks
    samples_to_show = []
    for _ in range(n_samples):
        chunk_path = random.choice(chunk_files)
        with np.load(chunk_path, allow_pickle=True) as loader:
            X = loader['X']
            Y_dicts = loader['Y_dicts']
            idx = random.randint(0, len(X) - 1)
            samples_to_show.append((X[idx], Y_dicts[idx], chunk_path, idx))

    # 3. Create Visualization Plot
    rows = (n_samples + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, (img, p_dict, chunk_name, idx) in enumerate(samples_to_show):
        ax = axes[i]
        
        # Display scattering image (log scale often looks better for scattering)
        # Using a small offset to avoid log(0)
        im = ax.imshow(img + 1e-6, cmap='viridis', norm=plt.cm.colors.LogNorm())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Create a title with the parameters
        title_str = (
            f"Chunk: {os.path.basename(chunk_name)} [Idx {idx}]\n"
            f"r={p_dict['radius']:.1f}, h={p_dict['height']:.1f}, a={p_dict['a']:.1f}"
        )
        ax.set_title(title_str, fontsize=10)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_filename = "dataset_preview.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nVisualization saved as '{output_filename}'. Open this file to see your data!")
    # If running on a local GUI-enabled machine, you can also use:
    # plt.show()

if __name__ == "__main__":
    # You can change the directory or number of samples here
    visualize_random_samples(n_samples=6)
