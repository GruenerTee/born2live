import bornagain as ba
from bornagain import nm, deg
from physics_models import LatticeModel, ParticleFactory

def test_unified_iteration():
    print("Testing Unified Particle Iteration...")
    
    # Standard AI-ready parameters
    test_params = {
        "size": 10*nm,
        "aspect": 1.5,
        "slope": 75*deg,
        "detail": 0.3
    }

    for shape in ParticleFactory.SHAPES:
        try:
            model = LatticeModel(shape=shape, **test_params)
            sample = model.create_sample()
            print(f"  [OK] Created {shape}")
        except Exception as e:
            print(f"  [FAIL] {shape}: {e}")

def test_legacy_support():
    print("\nTesting Legacy Support (radius/height)...")
    try:
        # Legacy call using radius and height instead of size/aspect
        model = LatticeModel(radius=8*nm, height=12*nm)
        assert model.size == 8*nm
        assert model.aspect == 1.5
        sample = model.create_sample()
        print("  [OK] Legacy support working correctly")
    except Exception as e:
        print(f"  [FAIL] Legacy support: {e}")

if __name__ == "__main__":
    test_unified_iteration()
    test_legacy_support()
