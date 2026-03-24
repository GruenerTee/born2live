#!/usr/bin/env python3
"""
Example of specular reflectometry simulation with a shrinking number of layers.
The sample consists of alternating Ti and Ni layers, where the number of
repetitions (N) is varied from 10 down to 1.
"""
import bornagain as ba
import threading
import multiprocessing
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot 
import numpy as np
import matplotlib.animation as animation
from bornagain import ba_plot as bp, deg, angstrom, nm
from born_pygame_viz import draw_stack
from bornagain import sample_tools
from creat_sample_test import make_layer_stack


def get_sample(n_repetitions, shrunk_factor, top_layer):

    vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_Ti = ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    material_Ni = ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    material_Si = ba.MaterialBySLD("Si", 2.0704e-06, 0)

    # Layers
    top_layer = ba.Layer(vacuum)
    ti_layer = ba.Layer(material_Ti, 30*angstrom)
    ni_layer = ba.Layer(material_Ni, 70*angstrom)
    substrate = ba.Layer(material_Si)

    if top_layer == 'Ti':
        stack1 = ba.LayerStack(1)
        
        shrunk_ti = ba.Layer(material_Ti, shrunk_factor*30*angstrom)        
        stack1.addLayer(shrunk_ti)
        stack1.addLayer(ni_layer)
    else:
        stack1 = ba.LayerStack(1)
        shrunk_ni = ba.Layer(material_Ni, shrunk_factor*70*angstrom)        
        stack1.addLayer(shrunk_ni)


    # Periodic stack
    if n_repetitions > 0:
       stack = ba.LayerStack(n_repetitions)
       stack.addLayer(ti_layer)
       stack.addLayer(ni_layer)

    
    # Sample
    sample = ba.Sample()
    
    sample.addLayer(top_layer)
    
    sample.addStack(stack1)
    if n_repetitions > 0:
        sample.addStack(stack)
    sample.addLayer(substrate)
    
    #print (sample_tools.materialProfile(sample))S
    return sample


def get_simulation(sample):
    n = 500
    scan = ba.AlphaScan(n, 2*deg/n, 2*deg)
    scan.setWavelength(1.54*angstrom)
    return ba.SpecularSimulation(scan, sample)

def get_simulation2d(sample):
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    n = 200
    detector = ba.SphericalDetector(n, -2*deg, 2*deg, n, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation

def wrapper (args):
    simulate_and_creat(*args)

def simulate_and_creat(n_rep, factor, Top, i):
    plt = bp.plt
    plt.figure(figsize=(10, 6))

    print(f"{i}th Simulation for n_repetitions = {n_rep}")
    #sample = get_sample(n_rep, factor, Top)
    sample = make_layer_stack(Top, factor,n_rep)

    #simulation = get_simulation(sample)
    simulation = get_simulation2d(sample)


    result = simulation.simulate()
            
            # Extract data for plotting
            
    bp.plot_datafield(result)
    plt.savefig("./sim/sim-"+f"{i:02d}",dpi=300 )
    plt.close()
    draw_stack(n_rep, Top,  factor, i)

if __name__ == '__main__':
    # Use matplotlib directly via bornagain's plotter to overlay multiple scans

    #clean env
    os.system('rm -r sim/ ; mkdir sim')
    


    # Iterate over a shrinking number of repetitions
    i=0
    threads =[]
    tasks=[]
    for n_rep in range(1, -1, -1):
      for Top in ['Ti', 'Ni']:

            for factor in np.arange(3, 0.1, -0.1):
                i+=1
                tasks.append((n_rep, factor,Top, i))

                #t =multiprocessing.Process(target=simulate_and_creat, args=(n_rep, factor,Top, i))
                #threads.append(t)
    with multiprocessing.Pool(4) as p:
        p.map(wrapper, tasks)

    #*for t in threads:
    #    t.start()

    #for t in threads:
    #    t.join()

    os.system("ffmpeg -f image2 -r 10 -i ./sim/sim-%02d.png -vcodec mpeg4 -y ./sim/film-sim.mp4")
    os.system("ffmpeg -f image2 -r 10 -i ./sim/layer_stack_viz-%02d.png -vcodec mpeg4 -y ./film-draw.mp4")

    bp.plt.show()