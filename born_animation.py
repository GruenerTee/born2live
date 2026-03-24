#!/usr/bin/env python3
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_sample(n_repetitions):
    """Creates a sample with a variable number of repetitions."""
    vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_Ti = ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    material_Ni = ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    material_Si = ba.MaterialBySLD("Si", 2.0704e-06, 0)

    top_layer = ba.Layer(vacuum)
    ti_layer = ba.Layer(material_Ti, 30*angstrom)
    ni_layer = ba.Layer(material_Ni, 70*angstrom)
    substrate = ba.Layer(material_Si)

    stack = ba.LayerStack(n_repetitions)
    stack.addLayer(ti_layer)
    stack.addLayer(ni_layer)

    sample = ba.Sample()
    sample.addLayer(top_layer)
    sample.addStack(stack)
    sample.addLayer(substrate)
    return sample

def get_simulation(sample):
    n = 500
    scan = ba.AlphaScan(n, 2*deg/n, 2*deg)
    scan.setWavelength(1.54*angstrom)
    return ba.SpecularSimulation(scan, sample)

# --- Animation Setup ---

fig, ax = plt.subplots(figsize=(10, 6))
# Create an empty line object that we will update in each frame
line, = ax.semilogy([], [], lw=2, color='blue')

# Setup axis limits and labels
ax.set_xlim(0, 2)
ax.set_ylim(1e-8, 1.2)
ax.set_xlabel('alpha_i (deg)')
ax.set_ylabel('Reflectivity')
ax.grid(True, which="both", ls="-", alpha=0.5)

def init():
    """Initialize the background of the animation."""
    line.set_data([], [])
    return line,

def update(n_rep):
    """
    Update function called for each frame.
    n_rep comes from the 'frames' parameter in FuncAnimation.
    """
    sample = get_sample(n_rep)
    simulation = get_simulation(sample)
    result = simulation.simulate()
    
    # Convert result to numpy arrays
    data = np.array(result)
    x = result.axis(0).binCenters()
    
    # Update the line data
    line.set_data(x, data)
    
    # Update title to show current state
    ax.set_title(f'Specular Reflectometry: N = {n_rep} repetitions')
    
    return line,

# Create the animation: cycling from 10 down to 1
# interval is delay between frames in milliseconds
ani = FuncAnimation(fig, update, frames=range(10, 0, -1),
                    init_func=init, blit=True, interval=500, repeat=True)

# To save the film (requires ffmpeg or imagemagick installed):
# ani.save('shrinking_layers.mp4', writer='ffmpeg', fps=2)
# print("Animation saved as shrinking_layers.mp4")

plt.show()
