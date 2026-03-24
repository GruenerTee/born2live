#!/usr/bin/env python3
"""
Cylinder form factor in Born approximation
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, nm


def get_sample():
    """
    A sample with cylinders in a homogeneous environment ("Vacuum"),
    implying a simulation in plain Born approximation.
    """

    # Materials
    material_particle = ba.RefractiveMaterial("Particle", 0.0006, 2e-08)
    vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)

    # Form factors
    ff = ba.Cylinder(5*nm, 5*nm)

    # Particles
    particle = ba.Particle(material_particle, ff)

    # Particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle)
    layout.setTotalParticleSurfaceDensity(0.01)

    # Layers
    layer = ba.Layer(vacuum)
    layer.addLayout(layout)

    # Sample
    sample = ba.Sample()
    sample.addLayer(layer)

    return sample


def get_simulation(sample):
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    n = 200
    detector = ba.SphericalDetector(n, -2*deg, 2*deg, n, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation


if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_datafield(result)
    bp.plt.show()
