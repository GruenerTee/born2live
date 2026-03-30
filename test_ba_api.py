import bornagain as ba
import numpy as np
import pickle
from creat_sample_test import make_particle_lattice_sample
from bornagain import nm, deg

def test_api():
    sample = make_particle_lattice_sample(radius=5*nm, height=4*nm, a=20*nm, b=20*nm)
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    detector = ba.SphericalDetector(4, -2*deg, 2*deg, 4, 0, 3*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    result = simulation.simulate()
    
    print(f"Result type: {type(result)}")
    
    # Try .dataArray()
    try:
        da = result.dataArray()
        print(f"dataArray type: {type(da)}")
        arr = np.array(da)
        print(f"np.array(da) type: {type(arr)}")
        pickle.dumps(arr)
        print("np.array(da) is picklable")
    except Exception as e:
        print(f"dataArray check failed: {e}")

    # Try .plottableField()
    try:
        pf = result.plottableField()
        print(f"plottableField type: {type(pf)}")
        pickle.dumps(pf)
        print("plottableField is picklable")
    except Exception as e:
        print(f"plottableField check failed: {e}")

if __name__ == "__main__":
    test_api()
