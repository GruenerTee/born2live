#!/usr/bin/env python3
import numpy as np
import os
import time

def export_sample(data_2d, params_dict, file_path="ai_dataset.npz"):
    """
    Exports a 2D data array and its associated parameters to a compressed .npz file.
    Appends to existing data if the file already exists.
    Uses a simple .lock file to prevent race conditions in multiprocessing.
    """
    lock_path = file_path + ".lock"
    
    # Simple lock mechanism
    while os.path.exists(lock_path):
        time.sleep(0.1)
    
    try:
        # Create lock
        open(lock_path, 'w').close()
        
        if os.path.exists(file_path):
            with np.load(file_path, allow_pickle=True) as loader:
                X = list(loader['X'])
                Y_dicts = list(loader['Y_dicts'])
        else:
            X = []
            Y_dicts = []

        X.append(np.array(data_2d, copy=True))
        Y_dicts.append(params_dict)

        np.savez_compressed(file_path, X=np.array(X), Y_dicts=np.array(Y_dicts, dtype=object))
    finally:
        # Remove lock
        if os.path.exists(lock_path):
            os.remove(lock_path)

def load_dataset(file_path, target_keys):
    """
    Loads the dataset and converts parameter dictionaries into a padded 2D numpy array.
    
    Args:
        file_path (str): Path to the .npz file.
        target_keys (list): List of parameter names in the desired order.
        
    Returns:
        tuple: (X, Y) where X is the 2D data array (N, H, W) and 
               Y is the parameter array (N, len(target_keys)) padded with 0s for missing keys.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No dataset found at {file_path}")

    with np.load(file_path, allow_pickle=True) as loader:
        X = loader['X']
        Y_dicts = loader['Y_dicts']

    # Convert list of dicts to a 2D numpy array based on target_keys
    n_samples = len(Y_dicts)
    n_params = len(target_keys)
    Y_padded = np.zeros((n_samples, n_params))

    for i, d in enumerate(Y_dicts):
        for j, key in enumerate(target_keys):
            # Pad with 0 if key is missing, otherwise take the value
            Y_padded[i, j] = d.get(key, 0.0)

    return X, Y_padded

if __name__ == "__main__":
    # Example usage / Test
    test_data = np.random.rand(128, 128)
    test_params = {"radius": 5.0, "height": 10.0}
    
    # Export
    export_sample(test_data, test_params, "test_dataset.npz")
    
    # Load with padding for a missing parameter 'a'
    X, Y = load_dataset("test_dataset.npz", ["radius", "height", "a"])
    
    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded Y (parameters): \n{Y}")
    print(f"Note: 'a' was padded with 0.0 as it was missing in the exported dict.")
