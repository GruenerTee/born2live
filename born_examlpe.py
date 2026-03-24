#!/usr/bin/env python3
"""
Basic example of specular reflectometry simulation.
The sample consists of 20 alternating Ti and Ni layers.
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom


def get_sample():
    # Materials
    vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_Ti = ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    material_Ni = ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    material_Si = ba.MaterialBySLD("Si", 2.0704e-06, 0)

    # Layers
    top_layer = ba.Layer(vacuum)
    ti_layer = ba.Layer(material_Ti, 30*angstrom)
    ni_layer = ba.Layer(material_Ni, 70*angstrom)
    substrate = ba.Layer(material_Si)

    # Periodic stack
    n_repetitions = 10
    stack = ba.LayerStack(n_repetitions)
    stack.addLayer(ti_layer)
    stack.addLayer(ni_layer)

    # Sample
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


if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_datafield(result)
    
    print (bp.export())

    bp.plt.show()
