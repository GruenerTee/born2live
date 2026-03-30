import bornagain as ba
from bornagain import deg, nm, angstrom

class MaterialLibrary:
    """Centralized repository for material definitions."""
    @staticmethod
    def get_vacuum():
        return ba.MaterialBySLD("Vacuum", 0, 0)
    
    @staticmethod
    def get_ti():
        return ba.MaterialBySLD("Ti", -1.9493e-06, 0)
    
    @staticmethod
    def get_ni():
        return ba.MaterialBySLD("Ni", 9.4245e-06, 0)
    
    @staticmethod
    def get_si():
        return ba.MaterialBySLD("Si", 2.0704e-06, 0)
    
    @staticmethod
    def get_refractive(name, delta, beta):
        return ba.RefractiveMaterial(name, delta, beta)

class LatticeModel:
    """Handles 2D particle lattice samples."""
    def __init__(self, radius=5*nm, height=5*nm, a=20*nm, b=20*nm, 
                 alpha=120*deg, xi=0*deg, ref_idx=0.0006):
        self.radius = radius
        self.height = height
        self.a = a
        self.b = b
        self.alpha = alpha
        self.xi = xi
        self.ref_idx = ref_idx

    def create_sample(self):
        mat_particle = MaterialLibrary.get_refractive("Particle", self.ref_idx, 2e-08)
        mat_substrate = MaterialLibrary.get_refractive("Substrate", 6e-06, 2e-08)
        mat_vacuum = MaterialLibrary.get_refractive("Vacuum", 0.0, 0.0)

        ff = ba.Cylinder(self.radius, self.height)
        particle = ba.Particle(mat_particle, ff)

        lattice = ba.BasicLattice2D(self.a, self.b, self.alpha, self.xi)
        
        # Interference function setup
        iff = ba.Interference2DParacrystal(lattice, 0*nm, 20000*nm, 20000*nm)
        iff.setIntegrationOverXi(True)
        pdf = ba.Profile2DCauchy(1*nm, 1*nm, 0*deg)
        iff.setProbabilityDistributions(pdf, pdf)

        layout = ba.ParticleLayout()
        layout.addParticle(particle)
        layout.setInterference(iff)
        layout.setTotalParticleSurfaceDensity(0.00288675134595)

        layer_vac = ba.Layer(mat_vacuum)
        layer_vac.addLayout(layout)
        layer_sub = ba.Layer(mat_substrate)

        sample = ba.Sample()
        sample.addLayer(layer_vac)
        sample.addLayer(layer_sub)
        return sample

class StackModel:
    """Handles multi-layer thin film stacks."""
    def __init__(self, top_material="Ti", factor=1.0, repetitions=0):
        self.top_material = top_material
        self.factor = factor
        self.repetitions = repetitions

    def create_sample(self):
        mat_vac = MaterialLibrary.get_vacuum()
        mat_ti = MaterialLibrary.get_ti()
        mat_ni = MaterialLibrary.get_ni()
        mat_si = MaterialLibrary.get_si()

        sample = ba.Sample()
        sample.addLayer(ba.Layer(mat_vac))
        
        # Custom top layer logic
        if self.top_material == "Ti":
            sample.addLayer(ba.Layer(mat_ti, self.factor * 30 * angstrom))
            sample.addLayer(ba.Layer(mat_ni, 70 * angstrom))
        elif self.top_material == "Ni":
            sample.addLayer(ba.Layer(mat_ni, self.factor * 70 * angstrom))

        # Repeating units
        ti_layer = ba.Layer(mat_ti, 30 * angstrom)
        ni_layer = ba.Layer(mat_ni, 70 * angstrom)
        for _ in range(self.repetitions):
            sample.addLayer(ti_layer)
            sample.addLayer(ni_layer)
            
        sample.addLayer(ba.Layer(mat_si))
        return sample

class SimulationEngine:
    """Configures and executes BornAgain simulations."""
    @staticmethod
    def run_specular(sample, n_bins=500):
        scan = ba.AlphaScan(n_bins, 2*deg/n_bins, 2*deg)
        scan.setWavelength(1.54*angstrom)
        simulation = ba.SpecularSimulation(scan, sample)
        return simulation.simulate()

    @staticmethod
    def run_scattering_2d(sample, n_pixels=128):
        beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
        detector = ba.SphericalDetector(n_pixels, -2*deg, 2*deg, n_pixels, 0, 3*deg)
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        return simulation.simulate()
