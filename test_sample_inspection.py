import bornagain as ba
from creat_sample_test import make_particle_lattice_sample

sample = make_particle_lattice_sample()

print("--- treeToString() ---")
try:
    print(sample.treeToString())
except AttributeError:
    print("treeToString() not available")

print("\n--- toPythonCode() ---")
try:
    print(sample.toPythonCode())
except AttributeError:
    print("toPythonCode() not available")

print("\n--- Inspecting layers ---")
try:
    for i, layer in enumerate(sample.layers()):
        material = layer.material()
        thickness = layer.thickness()
        print(f"Layer {i}: {material.name()}, thickness={thickness}")
except AttributeError:
    print("layers() not available on Sample object")

# Check if it's a MultiLayer (older API) or Sample (newer API)
print(f"\nSample type: {type(sample)}")
