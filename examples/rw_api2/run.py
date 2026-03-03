#!/usr/bin/env python
from pathlib import Path
import numpy as np
import shutil
from logzero import logger, logfile
import sys
import subprocess

import torch
import jax
import jax.numpy as jnp
import optax
from openmm.app import PDBFile
from openmm import unit
import mdtraj as md
from dmff.api import Hamiltonian
from base.inference.calculator import BaseCalculator

from dmff.mbar import Sample, TargetState, ReweightEstimator, A2NM, EV2KJ, SampleState
from dmff.mbar import buildFrameEnergyFunction
from dmff.mbar import LmpsBaseSampleState


if __name__ == '__main__':

    struct_name = 'mp757441'
    rundir = 'out'
    init_xml = f'{rundir}/forcefield.xml'
    init_pdb = f'{rundir}/{struct_name}.pdb'
    traj_file = f'{rundir}/{struct_name}.dcd'


    """ File (logs/outputs) housekeeping """
    task_name = struct_name
    logfile(f'{rundir}/{task_name}.log')
    data_dir = f'{rundir}/{task_name}'
    Path.mkdir(Path(data_dir), exist_ok=True)
    logger.info(f'{data_dir} is created')
    sd_save_prefix = f'{data_dir}/state_dict'
    pt_save_prefix = f'{data_dir}/checkpoint'


    """ running parameters for MD and optimization """
    settings = {
            'md_params': {
                'nsteps': 40000,
                'timestep': 2.0,
            'temperature': 300.0,
            'ttime': 20.0,
            'ptime': 200.0
        },
        'sampling_params': {
            'interval': 100,
            'skip': 10000,
        },
        'training_params': {
            'iter_num': 500,
            'lr': 5e-5,
            'rerun_threshold': 0.9,
        },
        'ref_vals': np.array([1.71, 1.724, 1.726, 61.72, 62.05, 61.97]),
        'target_wts': np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        }

    cutoff = 0.5

    nsteps, timestep, temperature, ttime, pparam = \
        settings['md_params']['nsteps'], \
        settings['md_params']['timestep'], \
        settings['md_params']['temperature'], \
        settings['md_params']['ttime'], \
        settings['md_params']['ptime']
    pressure = 0.0
    interval, skip = settings['sampling_params']['interval'], settings['sampling_params']['skip']

    
    """ 1. Define DMFF Hamiltonian and Potential """
    pdb = PDBFile(init_pdb)
    H = Hamiltonian(init_xml)
    ckpt_path = H.ffinfo['Forces']['BASEForce']['meta']['ckpt']
    pots = H.createPotential(pdb.topology, nonbondedCutoff=cutoff * unit.nanometer)
    gen = H.getGenerators()[0]
    params = H.getParameters().parameters

    # Backup initial state_dict
    gen.write_to(params, f'{sd_save_prefix}_0.sd')
    logger.info(f'Init sd saved into {sd_save_prefix}_0.sd')

    """ 2. Setup SampleState and perform initial sampling """
    state_ref = LmpsBaseSampleState(
            temperature,
            'ref',
            ckpt_path,
            init_pdb,
            cutoff=cutoff)
    state_ref.update_parameters(params)

    # initial sampling
    if not Path(traj_file).exists():
        logger.info('Initial sampling ...')
        sample = state_ref.sample(nsteps, interval, skip, timestep, ttime, pparam, traj_file)
        logger.info('Finished initial sampling')
    else:
        traj = md.load_dcd(traj_file, top=init_pdb)
        sample = Sample(traj, from_state=state_ref.name)
    starting_iter = 0

    """ 3. Set up ReweightEstimator based on SampleState and Sample object """
    rw = ReweightEstimator()
    rw.set_sample_and_state(sample=sample, state=state_ref)
    logger.info('Finish initialization')
    
    """ 4. set up the energy function & TargetState based on it """
    # always expose the state_dict version to dmff
    pot_func = pots.dmff_potentials[state_ref.ffname]
    energy_function = buildFrameEnergyFunction(pot_func, builtin_nbl=True)
    state_tgt = TargetState(temperature=temperature, energy_function=energy_function, pressure=pressure, legacy=False)
    logger.info(f'Finish loading target state')

    """ 5. Define property and loss function """
    def property_fn(traj):
        return np.array([np.hstack([frame.unitcell_lengths, frame.unitcell_angles])[0] for frame in traj])
    
    def loss_fn(params, properties):
        weights, kappa = rw.estimate_weight(state_tgt, params)
        properties_averaged = jnp.dot(weights, properties)
        properties_diff = properties_averaged - settings['ref_vals']
        loss = jnp.sum((properties_diff * settings['target_wts'])**2)
        return loss, kappa

    """ 6. load optimization settings & set up the optimizer """
    iter_num, lr, rerun_threshold = \
        settings['training_params']['iter_num'], \
        settings['training_params']['lr'], \
        settings['training_params']['rerun_threshold']

    logger.info(f'Training with constant lr={lr}')
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    """ 7. check initial status & run optimization """
    properties = property_fn(rw.sample.trajectory)
    loss_min = 1e4

    for i in range(starting_iter, iter_num + 1):
        (loss_value, kappa), grads = jax.value_and_grad(loss_fn, argnums=(0), has_aux=True)(params, properties)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        # update parameters
        params = optax.apply_updates(params, updates)

        if loss_value < loss_min:
            """ save the model with minimum loss """
            loss_min = loss_value
            gen.write_to(params, f'{sd_save_prefix}_min.sd')
        
        gen.write_to(params, f'{sd_save_prefix}_{i}.sd')
        logger.info(f'Iter {i}: loss_value = {loss_value}, loss_min = {loss_value}')
      
        # check overlap after updates
        _, kappa = loss_fn(params, properties)
        if kappa < rerun_threshold or jnp.isnan(kappa):
            logger.info(f"* iter {i}, kappa={kappa}, threshold {kappa}, rerun md")
            rw.remove_sample_and_state()
            state_ref.update_parameters(params, ckpt_path=f'{pt_save_prefix}_{i}.pt')
            sample = state_ref.sample(nsteps, interval, skip, timestep, ttime, pparam, traj_file)
            logger.info(f'* Finish rerunning md')
            rw.set_sample_and_state(sample=sample, state=state_ref)
            properties = property_fn(sample.trajectory)
            logger.info(f'* Finish updating sample state')
            
    """ 8. save the final model """
    state_ref.update_parameters(params, ckpt_path=f'{pt_save_prefix}_final.pt')
    gen.write_to(params, f'{sd_save_prefix}_final.sd')
