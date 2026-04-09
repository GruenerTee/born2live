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

class ParticleFactory:
    """
    Standardized factory for all BornAgain hard particles.
    
    Master Variables for AI Training:
    1. size (S): Lateral dimension (Radius, Edge, or Length).
    2. aspect (η): Height ratio (H = S * aspect).
    3. slope (α): Face angle in degrees (default 90).
    4. detail (δ): Shape-specific modifier (truncation, width ratio, etc).
    """
    SHAPES = [
        "Sphere", "Spheroid", "SphericalSegment", "SpheroidalSegment",
        "Cylinder", "Box", "Prism3", "Prism6", "Cone", 
        "Pyramid2", "Pyramid3", "Pyramid4", "Pyramid6", 
        "Bipyramid4", "CantellatedCube", "Dodecahedron", "Icosahedron",
        "PlatonicOctahedron", "PlatonicTetrahedron", "HemiEllipsoid",
        "HorizontalCylinder"
    ]

    @staticmethod
    def get_particle(material, shape="Cylinder", size=5*nm, aspect=1.0, 
                     slope=90*deg, detail=0.0):
        H = size * aspect
        
        # Ensure slope is safe for pyramids (avoiding zero or negative base)
        # Most BA pyramids fail if H / tan(slope) > inradius of the base
        safe_slope = max(slope, 10*deg) if slope < 90*deg else 90*deg
        
        # Geometry safety: Cap height so pyramid doesn't "converge" below zero
        import math
        if safe_slope < 90*deg:
            # The inradius 'r' is the distance from center to nearest edge
            if shape in ["Pyramid3", "Prism3"]:
                r = size / (2 * math.sqrt(3))
            elif shape == "Pyramid2":
                W = size if detail <= 0 else size * detail
                r = min(size, W) / 2.0
            elif shape == "Pyramid6":
                r = size * math.sqrt(3) / 2.0
            else: # Pyramid4, Cone, etc.
                r = size / 2.0
            
            h_limit = r * math.tan(safe_slope)
            H = min(H, h_limit * 0.95) # 5% safety margin

        if shape == "Sphere":
            ff = ba.Sphere(size)
        elif shape == "Spheroid":
            ff = ba.Spheroid(size, H)
        elif shape == "SphericalSegment":
            # radius, top_cut, bottom_cut
            ff = ba.SphericalSegment(size, detail * size, 0)
        elif shape == "SpheroidalSegment":
            # radius_xy, radius_z, rm_top, rm_bottom
            ff = ba.SpheroidalSegment(size, H, detail * H, 0)
        elif shape == "Cylinder":
            ff = ba.Cylinder(size, H)
        elif shape == "Box":
            W = size if detail <= 0 else size * detail
            ff = ba.Box(size, W, H)
        elif shape == "Prism3":
            ff = ba.Prism3(size, H)
        elif shape == "Prism6":
            ff = ba.Prism6(size, H)
        elif shape == "Cone":
            ff = ba.Cone(size, H, slope)
        elif shape == "Pyramid2":
            W = size if detail <= 0 else size * detail
            # Pyramid2 constructor: length, width, height, alpha
            ff = ba.Pyramid2(size, W, H, safe_slope)
        elif shape == "Pyramid3":
            ff = ba.Pyramid3(size, H, safe_slope)
        elif shape == "Pyramid4":
            ff = ba.Pyramid4(size, H, safe_slope)
        elif shape == "Pyramid6":
            ff = ba.Pyramid6(size, H, safe_slope)
        elif shape == "Bipyramid4":
            ff = ba.Bipyramid4(size, H, detail if detail > 0 else 0.5, safe_slope)
        elif shape == "CantellatedCube":
            ff = ba.CantellatedCube(size, detail * size if detail > 0 else 0.2 * size)
        elif shape == "Dodecahedron":
            ff = ba.Dodecahedron(size)
        elif shape == "Icosahedron":
            ff = ba.Icosahedron(size)
        elif shape == "PlatonicOctahedron":
            ff = ba.PlatonicOctahedron(size)
        elif shape == "PlatonicTetrahedron":
            ff = ba.PlatonicTetrahedron(size)
        elif shape == "HemiEllipsoid":
            W = size if detail <= 0 else size * detail
            ff = ba.HemiEllipsoid(size, W, H)
        elif shape == "HorizontalCylinder":
            # radius, length, slice_bottom, slice_top
            # mapping size to radius, H to length, detail to slices
            ff = ba.HorizontalCylinder(size, H, -size, size)
        else:
            raise ValueError(f"Unknown shape: {shape}. Supported: {ParticleFactory.SHAPES}")
            
        return ba.Particle(material, ff)

class LatticeModel:
    """Handles 2D particle lattice samples with unified parameterization."""
    def __init__(self, shape="Cylinder", size=5*nm, aspect=1.0, slope=90*deg, 
                 detail=0.0, a=20*nm, b=20*nm, alpha=120*deg, xi=0*deg, 
                 ref_idx=0.0006, **kwargs):
        
        # Legacy support for radius/height arguments
        if 'radius' in kwargs: size = kwargs['radius']
        if 'height' in kwargs: aspect = kwargs['height'] / size if size != 0 else 1.0

        self.shape = shape
        self.size = size
        self.aspect = aspect
        self.slope = slope
        self.detail = detail
        self.a = a
        self.b = b
        self.alpha = alpha
        self.xi = xi
        self.ref_idx = ref_idx

    def create_sample(self):
        mat_particle = MaterialLibrary.get_refractive("Particle", self.ref_idx, 2e-08)
        mat_substrate = MaterialLibrary.get_refractive("Substrate", 6e-06, 2e-08)
        mat_vacuum = MaterialLibrary.get_refractive("Vacuum", 0.0, 0.0)

        particle = ParticleFactory.get_particle(
            mat_particle, self.shape, self.size, self.aspect, self.slope, self.detail
        )

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
