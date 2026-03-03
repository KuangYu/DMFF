from dmff.mbar import (
    ReweightEstimator,
    Sample,
    SampleState,
    TargetState,
    OpenMMSampleState,
    buildTrajEnergyFunction,
    buildFrameEnergyFunction,
)
import dmff
import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import openmm as mm
import numpy as np
import numpy.testing as npt
try:
    import mdtraj as md
except ImportError as e:
    import warnings
    warnings.warn(f"mdtraj not found. Tests about MBAR would fail.")
try:
    from pymbar import MBAR
except ImportError as e:
    import warnings
    warnings.warn(f"pymbar not found. Tests about MBAR would fail.")
from dmff import Hamiltonian, NeighborListFreud
from tqdm import tqdm


class TestRW:
    @pytest.mark.parametrize(
        "pdb_file, prm_file, traj_file",
        [("tests/data/waterbox.pdb", "tests/data/water1.xml",
          "tests/data/w1_npt.dcd")])
    def test_rw_weight(self, pdb_file, prm_file, traj_file):
        pdb = app.PDBFile(pdb_file)
        H = Hamiltonian(prm_file)
        pot = H.createPotential(pdb.topology,
                        nonbondedMethod=app.PME,
                        nonbondedCutoff=0.9 * unit.nanometer)
        efunc = pot.getPotentialFunc()

        target_energy_function = buildFrameEnergyFunction(efunc,
                                                         pot.meta['cov_map'],
                                                         0.9)
        target_state = TargetState(300.0, target_energy_function, legacy=False)

        ref_state = OpenMMSampleState("ref",
                                       prm_file,
                                       pdb_file,
                                       temperature=300.0,
                                       pressure=1.0,
                                       legacy=False)
        traj = md.load(traj_file, top=pdb_file)[20::4]
        sample = Sample(traj, 'ref')
        rw = ReweightEstimator()
        rw.set_sample_and_state(sample, ref_state)
        
        params = H.getParameters().parameters
        params['NonbondedForce']['epsilon'] *= 1.05
        weights, kappa = rw.estimate_weight(target_state, params)

        wts_ref = np.array([1.76249840e-03, 3.48084472e-03, 6.35929095e-02, 4.92703480e-04,
                            2.10344084e-04, 6.67977579e-04, 1.26408870e-02, 2.74945042e-02,
                            1.24405102e-03, 2.03039164e-03, 8.76641819e-03, 8.04576942e-03,
                            7.27848160e-03, 5.61216728e-03, 1.74126627e-01, 9.48752213e-03,
                            7.87995566e-05, 3.47695895e-03, 4.39744929e-02, 1.13553087e-03,
                            5.91781464e-01, 2.35276913e-02, 6.66885665e-04, 9.78366294e-04,
                            7.44571363e-03])
        kappa_ref = 0.17884054587562492

        cos = np.dot(wts_ref, weights) / np.linalg.norm(wts_ref) / np.linalg.norm(weights)
        err_kappa = np.abs(kappa - kappa_ref)
        npt.assert_almost_equal(cos, 1.0, decimal=2)
        npt.assert_almost_equal(err_kappa, 0.0, decimal=3)
