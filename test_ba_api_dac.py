import bornagain as ba
from bornagain.numpyutil import Arrayf64Converter as dac
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
    da = result.dataArray()
    print(f"dataArray type: {type(da)}")

    # Try dac.asNpArray(da)
    try:
        arr_dac = dac.asNpArray(da)
        print(f"dac.asNpArray(da) type: {type(arr_dac)}")
        print(f"dac.asNpArray(da) shape: {arr_dac.shape}")
        pickle.dumps(arr_dac)
        print("dac.asNpArray(da) is picklable!")
    except Exception as e:
        print(f"dac.asNpArray(da) failed: {e}")

if __name__ == "__main__":
    test_api()
