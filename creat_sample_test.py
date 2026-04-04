import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom, nm

def make_layer_stack(top = "Ti",xfactor = 1 ,repitons=0,verbrose=False):
    periodic =False

    vacuum = ba.MaterialBySLD("Vacuum", 0, 0)
    material_Ti = ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    material_Ni = ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    material_Si = ba.MaterialBySLD("Si", 2.0704e-06, 0)

    # Layers
    top_layer = ba.Layer(vacuum)
    ti_layer = ba.Layer(material_Ti, 30*angstrom)
    ni_layer = ba.Layer(material_Ni, 70*angstrom)
    substrate = ba.Layer(material_Si)

    # Sample seting sample and bondari
    sample = ba.Sample()
    sample.addLayer(top_layer)
    
    if top == "Ti":
        if verbrose: print(top + " - " +str(xfactor*30))
        shrunk_layer = ba.Layer(material_Ti, xfactor*30*angstrom)
        sample.addLayer(shrunk_layer)
        sample.addLayer(ni_layer)
    elif top =='Ni':
        if verbrose: print(top + " - " +str(xfactor*30))
        shrunk_layer = ba.Layer(material_Ni, xfactor*30*angstrom)
        sample.addLayer(shrunk_layer)

    else:
        if verbrose: print('No top set')


    for layer in range (repitons):
        if verbrose: print("adding Ti-Ni-layer")
        sample.addLayer(ti_layer)
        sample.addLayer(ni_layer)
    # Periodic stack
    if periodic:
        stack = ba.LayerStack(5)
        stack.addLayer(ti_layer)
        stack.addLayer(ni_layer)
        sample.addStack(stack)    
    
   #bottom substart 
    sample.addLayer(substrate)

    return sample

def make_particle_lattice_sample(radius=5*nm, height=5*nm, ref_= 0.0006, a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg, shape="cylinder", verbrose=False):
    """
    Creates a sample with a 2D lattice of particles.
    Supported shapes: 'cylinder', 'sphere', 'box', 'cone'
    """
    if verbrose: 
        print(f"Creating {shape} lattice with:")
        print(f"R: {radius}, H: {height}, a: {a}, b: {b}, alpha: {alpha}")
    
    # Define materials
    material_Particle = ba.RefractiveMaterial("Particle", ref_ , 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6e-06, 2e-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    # Define form factors based on requested shape
    if shape.lower() == "cylinder":
        ff = ba.Cylinder(radius, height)
    elif shape.lower() == "sphere":
        ff = ba.Sphere(radius)
    elif shape.lower() == "box":
        # Using radius for half-width/length to keep scale consistent
        ff = ba.Box(radius*2, radius*2, height)
    elif shape.lower() == "cone":
        # Cone(radius, height, alpha)
        ff = ba.Cone(radius, height, 80*deg)
    else:
        print(f"Unknown shape '{shape}', defaulting to Cylinder")
        ff = ba.Cylinder(radius, height)

    # Define particles
    particle = ba.Particle(material_Particle, ff)

    # Define 2D lattices
    lattice = ba.BasicLattice2D(a, b, alpha, xi)

    # Define interference functions
    iff = ba.Interference2DParacrystal(lattice, 0*nm, 20000*nm, 20000*nm)
    iff.setIntegrationOverXi(True)
    iff_pdf_1  = ba.Profile2DCauchy(1*nm, 1*nm, 0*deg)
    iff_pdf_2  = ba.Profile2DCauchy(1*nm, 1*nm, 0*deg)
    iff.setProbabilityDistributions(iff_pdf_1, iff_pdf_2)

    # Define particle layouts
    layout = ba.ParticleLayout()
    layout.addParticle(particle)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(0.00288675134595)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_1.addLayout(layout)
    layer_2 = ba.Layer(material_Substrate)

    # Define sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)

    return sample