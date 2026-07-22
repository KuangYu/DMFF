#!/usr/bin/env python
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import value_and_grad

import numpy as np
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dmff.api import Hamiltonian
from dmff.common.nblist import NeighborListFreud, NeighborListNNPOps


def sort_pairs(pairs):
    if pairs.shape[0] == 0:
        return pairs
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def split_real_and_padding(pairs, natoms):
    pairs = np.asarray(pairs)
    real_mask = np.logical_and(pairs[:, 0] < natoms, pairs[:, 1] < natoms)
    return pairs[real_mask], pairs[~real_mask]


def check_backend(tag, reference_pairs, test_pairs, natoms):
    ref_real, ref_padding = split_real_and_padding(reference_pairs, natoms)
    test_real, test_padding = split_real_and_padding(test_pairs, natoms)

    ref_real = sort_pairs(ref_real)
    test_real = sort_pairs(test_real)

    assert np.all(test_real[:, 0] < test_real[:, 1]), f"{tag}: found unordered real pairs"
    assert ref_real.shape == test_real.shape, f"{tag}: pair-count mismatch {ref_real.shape} vs {test_real.shape}"
    assert np.array_equal(ref_real, test_real), f"{tag}: real-pair mismatch"

    if test_padding.shape[0] > 0:
        assert np.all(test_padding[:, 0] == natoms), f"{tag}: invalid padding i index"
        assert np.all(test_padding[:, 1] == natoms), f"{tag}: invalid padding j index"
        assert np.all(test_padding[:, 2] == 0), f"{tag}: invalid padding covalent tag"

    return {
        "real_pairs": int(test_real.shape[0]),
        "padding_pairs": int(test_padding.shape[0]),
        "capacity": int(np.asarray(test_pairs).shape[0]),
    }


if __name__ == '__main__':
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions('residues.xml')
    pdb = app.PDBFile('waterbox_31ang.pdb')
    rc = 0.6

    pots = H.createPotential(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=rc * unit.nanometer,
        has_aux=True,
        ethresh=5e-4,
    )

    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    natoms = positions.shape[0]

    freud_nbl = NeighborListFreud(box, rc, pots.meta['cov_map'])
    nnpops_nbl = NeighborListNNPOps(box, rc, pots.meta['cov_map'])

    freud_pairs = freud_nbl.allocate(positions)
    nnpops_pairs = nnpops_nbl.allocate(positions)
    alloc_stats = check_backend('allocate', freud_pairs, nnpops_pairs, natoms)

    displacement = 1.0e-4 * jnp.sin(jnp.arange(positions.size, dtype=positions.dtype)).reshape(positions.shape)
    positions_updated = positions + displacement

    freud_nbl.update(positions_updated)
    nnpops_nbl.update(positions_updated)
    freud_pairs_updated = freud_nbl.pairs
    nnpops_pairs_updated = nnpops_nbl.pairs
    update_stats = check_backend('update', freud_pairs_updated, nnpops_pairs_updated, natoms)

    print('NeighborListNNPOps matches NeighborListFreud.')
    print('allocate:', alloc_stats)
    print('update  :', update_stats)
    print('first 10 NNPOps pairs:')
    print(np.asarray(nnpops_nbl.pairs[:10]))


    # check energies
    params = H.getParameters()
    pot_disp = pots.dmff_potentials['ADMPDispForce']
    print('Check energy consistency:')
    res_disp, F_disp = value_and_grad(pot_disp, has_aux=True)(positions, box, freud_nbl.pairs, params)
    print(res_disp)
    res_disp, F_disp = value_and_grad(pot_disp, has_aux=True)(positions, box, nnpops_nbl.pairs, params)
    print(res_disp)
