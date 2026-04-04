import re
import bornagain as ba
import os
import sys

# Add parent directory to path so we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from born_lattice_pygame_viz import visualize_lattice_pygame

def clean_val(val):
    """Helper to clean numeric values from code strings."""
    return float(re.sub(r'[*nmdegangstrom \n\t]', '', val))

def extract_params(code_str):
    """
    Robustly parses the BornAgain Python code string to extract physical dimensions.
    Handles multi-line arguments and multiple shapes.
    """
    params = {'shape': 'unknown'}
    
    # 1. Identify Shape and Extract FF parameters
    if 'ba.Cylinder' in code_str:
        match = re.search(r'ba\.Cylinder\s*\(([^,]+),\s*([^)]+)\)', code_str)
        if match:
            params['radius'] = clean_val(match.group(1))
            params['height'] = clean_val(match.group(2))
            params['shape'] = 'cylinder'
            
    elif 'ba.Sphere' in code_str:
        match = re.search(r'ba\.Sphere\s*\(([^)]+)\)', code_str)
        if match:
            params['radius'] = clean_val(match.group(1))
            params['height'] = params['radius'] * 2
            params['shape'] = 'sphere'
            
    elif 'ba.Box' in code_str:
        match = re.search(r'ba\.Box\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)', code_str)
        if match:
            params['radius'] = clean_val(match.group(1)) / 2.0 # Half-width as proxy for radius
            params['height'] = clean_val(match.group(3))
            params['shape'] = 'box'
            
    elif 'ba.Cone' in code_str:
        match = re.search(r'ba\.Cone\s*\(([^,]+),\s*([^,]+),', code_str)
        if match:
            params['radius'] = clean_val(match.group(1))
            params['height'] = clean_val(match.group(2))
            params['shape'] = 'cone'

    # 2. Extract Lattice parameters (handles multi-line)
    lat_match = re.search(r'ba\.BasicLattice2D\s*\(\s*([^,]+),\s*([^,]+),', code_str, re.DOTALL)
    if lat_match:
        params['a'] = clean_val(lat_match.group(1))
        params['b'] = clean_val(lat_match.group(2))
    else:
        # Fallback for even more complex formatting
        a_match = re.search(r'ba\.BasicLattice2D\s*\(\s*([^,]+)', code_str)
        if a_match: params['a'] = clean_val(a_match.group(1))

    return params

def visualize_ba_sample(sample, output_path="sample_viz.png"):
    """
    The main bridge: ba.Sample -> Code String -> Pygame Image
    """
    print(f"Inspecting sample for visualization...")
    code = ba.sampleCode(sample)
    params = extract_params(code)
    
    # We need at least radius and lattice 'a' to draw something meaningful
    if 'radius' in params and 'a' in params:
        print(f"Detected {params['shape']} lattice. Parameters: {params}")
        
        # Use existing pygame viz engine
        visualize_lattice_pygame(
            output_path, 
            radius=params.get('radius', 5.0),
            height=params.get('height', 5.0),
            a=params.get('a', 20.0),
            b=params.get('b', params.get('a', 20.0)),
            alpha=120.0 * (3.14159/180.0), # Default if not found
            xi=0.0
        )
        print(f"Visualization saved to {output_path}")
    else:
        print("ERROR: Could not extract enough parameters.")
        print("Debug - Parsed params:", params)
        print("Debug - Generated Code snippet:", code[:200] + "...")

if __name__ == "__main__":
    # Test with Sphere
    from creat_sample_test import make_particle_lattice_sample
    from bornagain import nm
    s = make_particle_lattice_sample(shape="sphere", radius=10*nm)
    visualize_ba_sample(s, "test_sphere.png")
