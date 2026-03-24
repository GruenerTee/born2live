#!/usr/bin/env python3
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_sample(thickness_scale):
    """
    Creates a sample where layer thicknesses are scaled.
    thickness_scale: 1.0 is full thickness, 0.0 is zero thickness.
    """
    # Materials
    vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_Ti = ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    material_Ni = ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    material_Si = ba.MaterialBySLD("Si", 2.0704e-06, 0)

    # Layers with scaled thickness
    ti_thick = 30 * angstrom * thickness_scale
    ni_thick = 70 * angstrom * thickness_scale
    
    top_layer = ba.Layer(vacuum)
    ti_layer = ba.Layer(material_Ti, ti_thick)
    ni_layer = ba.Layer(material_Ni, ni_thick)
    substrate = ba.Layer(material_Si)

    # Fixed number of repetitions
    n_repetitions = 10
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
line, = ax.semilogy([], [], lw=2, color='red')

ax.set_xlim(0, 2)
ax.set_ylim(1e-8, 1.2)
ax.set_xlabel('alpha_i (deg)')
ax.set_ylabel('Reflectivity')
ax.grid(True, which="both", ls="-", alpha=0.5)

def init():
    line.set_data([], [])
    return line,

def update(scale):
    """Update function for thickness scaling."""
    sample = get_sample(scale)
    simulation = get_simulation(sample)
    result = simulation.simulate()
    
    data = np.array(result)
    x = result.axis(0).binCenters()
    
    line.set_data(x, data)
    
    # Calculate current thicknesses for the title
    ti_now = 30 * scale
    ni_now = 70 * scale
    ax.set_title(f'Shrinking Layers: Ti={ti_now:.1f}Å, Ni={ni_now:.1f}Å (Scale: {scale:.2f})')
    
    return line,

# Animate scale from 1.0 down to 0.0 in 20 steps
scales = np.linspace(1.0, 0.0, 30)

ani = FuncAnimation(fig, update, frames=scales,
                    init_func=init, blit=True, interval=100, repeat=True)

# To save the film:
# ani.save('shrinking_thickness.mp4', writer='ffmpeg', fps=10)

plt.show()
