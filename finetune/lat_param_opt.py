from pathlib import Path
import shutil
import re
import bz2, pickle
import time
import numpy as np
from copy import deepcopy
from logzero import logger, logfile
import sys
sys.path.append('..')
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
from ase import Atoms
from openmm.app import PDBFile
from openmm import unit
from dmff.api import Hamiltonian
from base.inference.calculator import BaseCalculator

from finetune.ft import Sample, TargetState, ASENNPNPTSampleState, ReweightEstimator, A2NM, load_ckpt_from_xml, load_xml


DATA_ROOT = f'/mnt/bn/ai4s-hl/cliang/baseft-data'
LOG_ROOT = DATA_ROOT + '/log'
Path.mkdir(Path(LOG_ROOT), exist_ok=True)


reweighting_params = {
    'md_params': {
        'nsteps': 40000,
        'timestep': 2.0,
        'temperature': 300.0,
        'ttime': 20.0,
        'pfactor': 10000.0,
    },

    'sampling_params': {
        'interval': 100,
        'skip': 10000,
    },
  
    'training_params': {
        'iter_num': 500,
        'lr': 5e-5,
        'rerun_threshold': 0.9,
    }
}


_cutoff = 0.5


def load_evaluator(ckpt_path, config_path=None, cutoff=_cutoff / A2NM):
    # ckpt_path = ckpt_path.split('_')
    # ckpt_path = '_'.join([ckpt_path[0], ckpt_path[1], ckpt_path[2], ckpt_path[4]])
    if config_path is not None:
        config_path = Path(config_path)
        config_path = Path(*config_path.parts[1:])
        return BaseCalculator(ckpt_path=ckpt_path, config_path=str(config_path), cutoff=cutoff)
    return BaseCalculator(ckpt_path=ckpt_path, cutoff=cutoff)


def modify_pdb_residue_numbers(input_file, output_file):
    new_lines = []
    with open(input_file, 'r') as f_in:
        atom_count = 0
        for line in f_in:
            if line.startswith('ATOM  '):
                atom_count += 1
                new_line = (
                    line[:22]
                    + f'{atom_count:4d}'
                    + line[26:]
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line)
    with open(output_file, 'w') as f_out:
        f_out.writelines(new_lines)
    return


def read_init_struct(struct_name):
    from ase.build import make_supercell
    from pymatgen.core import SymmOp
    from pymatgen.core.structure import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    # https://matsci.org/t/putting-lattice-vectors-in-lower-triangle-form/47172
    structure = Structure.from_file(f'{DATA_ROOT}/POSCAR_{struct_name}')
    atoms = AseAtomsAdaptor.get_atoms(structure)
    cell = atoms.cell
    std = cell.standard_form()
    rotation_matrix = std[1]
    myop = SymmOp.from_rotation_and_translation(rotation_matrix)
    structure.apply_operation(myop)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    # structure.to('./POSCAR.out', fmt='poscar')

    P = np.array([[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]])
    supercell = make_supercell(atoms, P)
    return supercell


def read_init_pdb(struct_name):
    if not Path(f'{DATA_ROOT}/{struct_name}.pdb').exists():
        from ase.io import write
        atoms = read_init_struct(struct_name)
        write(f'{DATA_ROOT}/{struct_name}.pdb', atoms, format='proteindatabank')
        modify_pdb_residue_numbers(f'{DATA_ROOT}/{struct_name}.pdb', f'{DATA_ROOT}/{struct_name}.pdb')
    return PDBFile(f'{DATA_ROOT}/{struct_name}.pdb')


def md_name(nsteps, interval, skip, timestep, temperature, ttime, pfactor):
    name = f'{nsteps}-{interval}-{skip}-{timestep}-{temperature}-{ttime}-{pfactor}'
    name = name.replace('.', 'p')
    return name


def frame2atoms(traj, i):
    atom = Atoms(numbers=traj[i]["atomic_number"], cell=traj[i]["cell"], 
                 positions=traj[i]["atom_positions"], pbc=[1, 1, 1])
    return atom


def rerun_md_threshold(weight):
    """ following J. Chem. Theory Comput. 2011, 7, 1773-1782 """
    weight = weight / len(weight)
    kappa = jnp.exp(-1.0 * jnp.sum(weight * jnp.log(weight))) / len(weight)
    # """ Kish's effective sample size """
    # logger.info(f'use Kish effective sample size')
    # kappa = jnp.sum(weight) ** 2 / jnp.sum(weight ** 2) / len(weight)
    return kappa


def main(struct_name, tgt_value_list, loss_lmbda_list, prop_func_list, prop_name='lat_param', ffname='BASEForce'):
    
    init_atoms = read_init_struct(struct_name)
    nsteps, timestep, temperature, ttime, pfactor = \
        reweighting_params['md_params']['nsteps'], \
        reweighting_params['md_params']['timestep'], \
        reweighting_params['md_params']['temperature'], \
        reweighting_params['md_params']['ttime'], \
        reweighting_params['md_params']['pfactor']
    pressure = 0.0
    interval, skip = reweighting_params['sampling_params']['interval'], reweighting_params['sampling_params']['skip']
    target_name = '-'.join([f'{ll}x{t:.4g}' for ll, t in zip(loss_lmbda_list, tgt_value_list)])
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    task_name = f'{struct_name}_{prop_name}-{target_name}_{md_name(nsteps, interval, skip, timestep, temperature, ttime, pfactor)}'
    task_name = task_name.replace('.', 'p')
    task_name = f'{time_stamp}_{task_name}'
    logfile(f'{LOG_ROOT}/{task_name}.log')
    data_dir = f'{DATA_ROOT}/{task_name}'
    
    init_atoms_pdb = read_init_pdb(struct_name)
    init_xml_name = 'forcefield'
    init_xml = f'{DATA_ROOT}/{init_xml_name}.xml'
    ffinfo_temp = load_xml(init_xml)
    init_ckpt_path, _ = load_ckpt_from_xml(ffinfo_temp, ffname)
    init_ckpt_name = Path(init_ckpt_path).stem
    
    ref_name = 'ref'
    state_ref = ASENNPNPTSampleState(temperature=temperature, 
                                     name=ref_name, 
                                     init_xml=init_xml, 
                                     pressure=pressure,
                                     ffname=ffname, 
                                     e_eval_loader=load_evaluator, 
                                     cutoff=_cutoff)
    
    Path.mkdir(Path(data_dir))
    logger.info(f'{data_dir} is created')
    md_file_name = f'{DATA_ROOT}/{struct_name}_{md_name(nsteps, interval, skip, timestep, temperature, ttime, pfactor)}.pkl.bz2'
    init_atoms_copy = deepcopy(init_atoms)
    if not Path(md_file_name).exists():
        sample = state_ref.sample(init_atoms=init_atoms_copy, nsteps=nsteps, interval=interval, skip=skip, 
                                timestep=timestep, ttime=ttime, pfactor=pfactor, file_name=md_file_name)
        traj = sample.trajectory
    else:
        with bz2.open(md_file_name, "rb") as file:
            traj = pickle.load(file)
        for i in range(len(traj)):
            traj[i]['cell'] = jnp.array(traj[i]['cell']).astype(jnp.float32) * A2NM
            traj[i]['atom_positions'] = jnp.array(traj[i]['atom_positions']).astype(jnp.float32) * A2NM
        sample = Sample(traj, from_state=ref_name)
    starting_iter = 0

    rw = ReweightEstimator()
    rw.add_state(state_ref)
    rw.add_sample(sample)

    logger.info('Finish initialization')
    
    H = Hamiltonian(init_xml)
    pots = H.createPotential(init_atoms_pdb.topology, nonbondedCutoff=_cutoff * unit.nanometer)
    params = H.getParameters().parameters
    pot_func = pots.dmff_potentials[ffname]
    
    def energy_function(_frame, _parameters):
        return pot_func(jnp.array(_frame['atom_positions']), jnp.array(_frame['cell']), None, _parameters)

    state_tgt = TargetState(temperature=temperature, energy_function=energy_function, pressure=pressure)
    
    logger.info(f'Finish loading target state with target value: {target_name}')

    iter_num, lr, rerun_threshold = \
        reweighting_params['training_params']['iter_num'], \
        reweighting_params['training_params']['lr'], \
        reweighting_params['training_params']['rerun_threshold']

    logger.info(f'Training with constant lr={lr}')
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    def compute_prop(_traj):
        return [jnp.array(p_func(_traj)) for p_func in prop_func_list]
    
    def compute_aver_prop(_prop_array):
        return [jnp.mean(p_array) for p_array in _prop_array]
    
    def compute_weighted_aver_prop(_weight, _prop_array):
        return [jnp.mean(p_array * _weight) for p_array in _prop_array]
    
    def loss_fn(_weighted_prop, _tgt_value_list, _loss_lmbda_list):
        loss_total = [jnp.power(prop * loss_lmbda - target_value * loss_lmbda, 2) 
                      for prop, target_value, loss_lmbda in zip(_weighted_prop, _tgt_value_list, _loss_lmbda_list)]
        return jnp.sum(jnp.array(loss_total))
    
    def ckeck_status(_prop_array, _state_tgt, _tgt_value_list, _loss_lmbda_list, _params):
        weight = rw.estimate_weight(target_state=_state_tgt, params=_params)
        prop = compute_aver_prop(_prop_array)
        weighted_prop = compute_weighted_aver_prop(weight, _prop_array)
        loss_value = loss_fn(weighted_prop, _tgt_value_list, _loss_lmbda_list)
        kappa = rerun_md_threshold(weight)
        return prop, weighted_prop, loss_value, kappa
    
    def loss(_prop_array, _state_tgt, _tgt_value_list, _loss_lmbda_list, _params):
        weight = rw.estimate_weight(target_state=_state_tgt, params=_params)
        weighted_prop = compute_weighted_aver_prop(weight, _prop_array)
        loss_value = loss_fn(weighted_prop, _tgt_value_list, _loss_lmbda_list)
        return loss_value
    
    prop_array = compute_prop(traj)
    prop, weighted_prop, loss_value, kappa = ckeck_status(prop_array, state_tgt, tgt_value_list, loss_lmbda_list, params)
    loss_min = loss_value
    logger.info(f'iter {starting_iter}, pred prop {prop}, weighted prop {weighted_prop}, loss {loss_value}, loss_min {loss_min}, kappa {kappa}')
    
    for i in range(1 + starting_iter, iter_num + 1):
        grads = jax.grad(loss, argnums=(4))(prop_array, state_tgt, tgt_value_list, loss_lmbda_list, params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        prop, weighted_prop, loss_value, kappa = ckeck_status(prop_array, state_tgt, tgt_value_list, loss_lmbda_list, params)
        if loss_value < loss_min:
            loss_min = loss_value
            generator = H.getGenerators()[0]
            generator.write_to(params=params, state_dict_file=f'{data_dir}/{init_ckpt_name}_min.pt')
        logger.info(f'iter {i}, pred prop {prop}, weighted prop {weighted_prop}, loss {loss_value}, loss_min {loss_min}, kappa {kappa}')
        if kappa < rerun_threshold or jnp.isnan(kappa):
            logger.info(f"iter {i}, threshold {kappa}, rerun md")
            rw.remove_state(ref_name)
            ckpt_name = f'{data_dir}/{init_ckpt_name}_{i}.pt'
            xml_name = f'{data_dir}/{init_xml_name}_{i}.xml'
            state_ref.update_parameters(H=H, params=params, ckpt_name=ckpt_name, xml_name=xml_name)
            init_atoms_copy = deepcopy(init_atoms)
            sample = state_ref.sample(init_atoms=init_atoms_copy, nsteps=nsteps, interval=interval, skip=skip, 
                                      timestep=timestep, ttime=ttime, pfactor=pfactor, file_name=f'{data_dir}/md_{i}.pkl.bz2')
            logger.info(f'Finish rerunning md')
            rw.add_state(state_ref)
            rw.add_sample(sample)
            traj = sample.trajectory
            prop_array = compute_prop(traj)
            logger.info(f'iter {i}, after rerunning md:')
            prop, weighted_prop, loss_value, kappa = ckeck_status(prop_array, state_tgt, tgt_value_list, loss_lmbda_list, params)
            logger.info(f'iter {i}, pred prop {prop}, weighted prop {weighted_prop}, loss {loss_value}, loss_min {loss_min}, kappa {kappa}')
        else:
            logger.info(f"iter {i}, threshold {kappa}, no need to rerun md")
    generator = H.getGenerators()[0]
    generator.write_to(params=params, state_dict_file=f'{data_dir}/{init_ckpt_name}_latest.pt')
    return


if __name__ == '__main__':
    from pymatgen.core import Lattice
    from functools import partial

    def get_cell_params(traj, p_idx):
        return [np.array(Lattice(traj[i]['cell']).parameters[p_idx]) for i in range(len(traj))]

    cell_params_fn = [partial(get_cell_params, p_idx=p) for p in range(6)]
    main(struct_name='mp-757441-Li5Mn2Fe3_PO4_6', 
         tgt_value_list=[1.71, 1.724, 1.726, 61.72, 62.05, 61.97], 
         loss_lmbda_list=[10.0, 10.0, 10.0, 1.0, 1.0, 1.0], 
         prop_func_list=cell_params_fn)
