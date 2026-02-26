from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import pickle, bz2
import collections
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase import units
import jax.numpy as jnp

from dmff.api.xmlio import XMLIO
from dmff.common.constants import EV2KJ


""" Suppose traj has been produced as a list of dict, with keys: 'cell', 'atom_positions', 'atomic_number' """


A2NM = 0.1


class SampleState:
    def __init__(self, temperature, name, pressure=0.0, mu_list=[]):
        self.temperature = temperature
        self.name = name
        self.beta = 1.0 / temperature / 8.314 * 1000.0
        self.pressure = pressure
        self.mu_list = mu_list
        return
    
    def calc_energy_frame(self, frame):
        raise NotImplementedError

    def calc_energy(self, trajectory):
        # return beta * u
        eners = []
        for frame in tqdm(trajectory):
            e = self.calc_energy_frame(frame)
            if self.pressure != 0:
                e += 0.06023 * self.pressure * compute_volume(frame['cell'])
            if len(self.mu_list) != 0 and num_atom_list is not None:
                elements_counting = pd.Series(frame['atomic_number']).value_counts(sort=False)  # pd.unique does not sort
                num_atom_list = elements_counting.to_list()
                e += sum(mu * num_atom for mu, num_atom in zip(self.mu_list, num_atom_list))
            eners.append(e * self.beta)
        return jnp.array(eners)

    # Newly added, resample to generate a Sample object
    # user-defined
    def sample(self, *args):
        """ Run MD here, generate a sample object
        args: parameters for MD (different for different MD packages)
        """
        raise NotImplementedError

    def update_parameters(self, *args):
        # update parameters of the state
        raise NotImplementedError


class ASENNPNPTSampleState(SampleState):
    def __init__(self, temperature, name, init_xml, ffname, e_eval_loader, cutoff=0.5, pressure=0.0, mu_list=[]):
        super().__init__(temperature=temperature, name=name, pressure=pressure, mu_list=mu_list)
        self.ffinfo_temp = load_xml(init_xml)
        self.ffname = ffname
        self.e_eval_loader = e_eval_loader
        self.init_ckpt_path, self.config_path = load_ckpt_from_xml(self.ffinfo_temp, ffname)
        self.cutoff = cutoff / A2NM  # A
        self.e_eval = self.e_eval_loader(self.init_ckpt_path, self.config_path, self.cutoff)
        return
    
    def calc_energy_frame(self, frame):
        atoms = Atoms(numbers=frame["atomic_number"], 
                      cell=frame["cell"] / A2NM, 
                      positions=frame["atom_positions"] / A2NM, 
                      pbc=[1, 1, 1])
        atoms.calc = self.e_eval
        energy = atoms.get_potential_energy()
        return energy * EV2KJ

    # Newly added, resample to generate a Sample object
    # user-defined
    def sample(self, init_atoms, nsteps, interval, skip, timestep, ttime, pfactor, file_name):
        init_atoms.calc = self.e_eval
        MaxwellBoltzmannDistribution(init_atoms, temperature_K=self.temperature)
        dyn = NPT(
            init_atoms,
            timestep=timestep*units.fs,
            temperature_K=self.temperature,
            externalstress=self.pressure*units.bar,
            ttime=ttime*units.fs,
            pfactor=pfactor*units.GPa*(units.fs**2),
            logfile=None,
            loginterval=1
        )
        obs = TrajectoryObserver(init_atoms)
        dyn.attach(obs, interval=interval)
        dyn.run(nsteps)
        obs.save(file_name, skip=int(skip / interval))
        with bz2.open(file_name, "rb") as file:
            obs = pickle.load(file)
        for i in range(len(obs)):
            obs[i]['cell'] = jnp.array(obs[i]['cell']).astype(jnp.float32) * A2NM
            obs[i]['atom_positions'] = jnp.array(obs[i]['atom_positions']).astype(jnp.float32) * A2NM
        return Sample(obs, from_state=self.name)

    def update_parameters(self, H, params, ckpt_name, xml_name):
        generator = H.getGenerators()[0]
        generator.write_to(params=params, state_dict_file=ckpt_name)
        write_ckpt_to_xml(new_xml_file=xml_name, ffinfo=self.ffinfo_temp, 
                          state_dict_file=ckpt_name, config=self.config_path, ffname=self.ffname)
        self.e_eval = self.e_eval_loader(ckpt_name, self.config_path, self.cutoff)
        return


class TrajectoryObserver(collections.abc.Sequence):
    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.cell = []
        self.atom_positions = []
        self.atomic_number = []
        return

    def __call__(self):
        self.cell.append(self.atoms.get_cell()[:])
        self.atom_positions.append(self.atoms.get_positions())
        self.atomic_number.append(self.atoms.get_atomic_numbers())
        return

    def __getitem__(self, item):
        return self.cell[item], self.atom_positions[item], self.atomic_number[item]
        
    def __len__(self):
        return len(self.cell)

    def save(self, filename, skip):
        traj = []
        for i in range(skip, len(self.cell)):
            traj.append({'cell': self.cell[i], 
                         'atom_positions': self.atom_positions[i], 
                         'atomic_number': self.atomic_number[i]})
        with bz2.open(filename, "wb") as file:
            pickle.dump(traj, file)
        return


class Sample:
    def __init__(self, trajectory, from_state):
        self.trajectory = trajectory
        self.from_state = from_state
        self.energy_data = {}
        return

    def generate_energy(self, state_list):
        for state in state_list:
            if state.name not in self.energy_data:
                self.energy_data[state.name] = np.array(
                    state.calc_energy(self.trajectory)
                )
        return
    

class TargetState:
    def __init__(self, temperature, energy_function, pressure=0.0, mu_list=[]):
        self.temperature = temperature
        self.energy_function = energy_function
        self.beta = 1.0 / temperature / 8.314 * 1000.0
        self.pressure = pressure
        self.mu_list = mu_list
        return

    def calc_energy(self, trajectory, parameters):
        eners = []
        for frame in tqdm(trajectory):
            e = self.energy_function(frame, parameters)
            if self.pressure != 0:
                e += 0.06023 * self.pressure * compute_volume(frame['cell'])
            if len(self.mu_list) != 0 and num_atom_list is not None:
                elements_counting = pd.Series(frame['atomic_number']).value_counts(sort=False)  # pd.unique does not sort
                num_atom_list = elements_counting.to_list()
                e += sum(mu * num_atom for mu, num_atom in zip(self.mu_list, num_atom_list))
            eners.append(e * self.beta)
        return jnp.array(eners)


class ReweightEstimator:
    def __init__(self, base_energies=None):
        self.samples = []
        self.states = []
        self.base_energies = base_energies
        return

    def add_sample(self, sample):
        self.samples.append(sample)
        return

    def add_state(self, state):
        self.states.append(state)
        return

    def remove_sample(self, name):
        init_num = len(self.samples)
        self.samples = [s for s in self.samples if s.from_state != name]
        final_num = len(self.samples)
        assert init_num > final_num
        return

    def remove_state(self, name):
        init_num = len(self.states)
        self.states = [s for s in self.states if s.name != name]
        final_num = len(self.states)
        assert init_num > final_num
        self.remove_sample(name)
        return
    
    def _collect_traj(self):
        self._full_samples = []
        for sample in self.samples:
            self._full_samples.extend(sample.trajectory)
        return
    
    def compute_energy_matrix(self):
        for sample in self.samples:
            sample.generate_energy(self.states)
        return

    def estimate_weight(self, target_state, params):
        self._collect_traj()
        self.compute_energy_matrix()
        uref, unew = [], []
        for sample in self.samples:
            for s_name, e_data in sample.energy_data.items():
                if self.base_energies is not None:
                    if s_name in self.base_energies:
                        for state in self.states:
                            if state.name == s_name:
                                beta = state.beta
                                break
                        e_data += beta * self.base_energies[s_name]
                uref.append(e_data)
        uref = jnp.concatenate(uref)
        unew = target_state.calc_energy(self._full_samples, params)
        deltaU = unew - uref
        deltaU = deltaU - deltaU.max()
        weight = jnp.exp(-deltaU)
        weight = weight / weight.mean()
        return weight


def compute_volume(cell):
    return jnp.abs(jnp.linalg.det(cell))


def load_xml(xml_file):
    xmlio = XMLIO()
    xmlio.loadXML(xml_file)
    ffinfo = xmlio.parseXML()
    return ffinfo


def load_ckpt_from_xml(ffinfo, ffname):
    ffinfo = ffinfo["Forces"][ffname]["meta"]
    return ffinfo["state_dict"], ffinfo["config"]


def write_ckpt_to_xml(new_xml_file, ffinfo, state_dict_file, config, ffname):
    xmlio = XMLIO()
    ffinfo["Forces"][ffname]["meta"]["state_dict"] = state_dict_file
    ffinfo["Forces"][ffname]["meta"]["config"] = config
    xmlio.writeXML(new_xml_file, ffinfo)
    return
