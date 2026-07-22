from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..utils import jit_condition
import numpy as np
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import pickle
import re
from functools import partial
from collections import OrderedDict
import copy

from ..sgnn.graph import MAX_VALENCE, TopGraph, from_pdb
from ..sgnn.gnn import MolGNNForce, prm_transform_f2i
from ..eann.eann import EANNForce, get_elem_indices
from ..api.topology import elem_to_index
from ..common.constants import EV2KJ

# load torch-related module
try:
    import torch
    import torch.nn as nn
    from ..torch_tools import t2j_pytree, j2t_pytree, wrap_torch_potential_kernel, t2j_extract_grad
    from torch2jax import t2j, j2t
except ImportError:
    pass

# load base-related module
try:
    from base.inference.calculator import get_parser
    import pymatgen.core.structure
except ImportError:
    pass


class SGNNGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):

        self.name = "SGNNForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        self.file = self.ffinfo["Forces"][self.name]["meta"]["file"]
        self.nn = int(self.ffinfo["Forces"][self.name]["meta"]["nn"])
        self.pdb = self.ffinfo["Forces"][self.name]["meta"]["pdb"]

        # load ML potential parameters
        with open(self.file, 'rb') as ifile:
            params = pickle.load(ifile)

        # convert to jnp array
        for k in params:
            params[k] = jnp.array(params[k])
            # set mask to all true
            paramset.addParameter(params[k], k, field=self.name, mask=jnp.ones(params[k].shape))

        # mask = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape), params)
        # paramset.addParameter(params, "params", field=self.name, mask=mask)
       

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        with open(self.file, 'wb') as ofile:
            pickle.dump(paramset[self.name], ofile)
        return

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        self.G = from_pdb(self.pdb)
        n_atoms = topdata.getNumAtoms()
        self.model = MolGNNForce(self.G, nn=self.nn)
        n_layers = self.model.n_layers
        def potential_fn(positions, box, pairs, params):
            # convert unit to angstrom
            positions = positions * 10
            box = box * 10
            prms = prm_transform_f2i(params[self.name], n_layers)
            return self.model.get_energy(positions, box, prms)

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["SGNNForce"] = SGNNGenerator

class EANNGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):

        self.name = "EANNForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        self.file = self.ffinfo["Forces"][self.name]["meta"]["file"]
        self.ngto = int(self.ffinfo["Forces"][self.name]["meta"]["ngto"])
        self.nipsin = int(self.ffinfo["Forces"][self.name]["meta"]["nipsin"])
        self.rc = float(self.ffinfo["Forces"][self.name]["meta"]["rc"]) * 10

        self.pdb = self.ffinfo["Forces"][self.name]["meta"]["pdb"]
        self.ommtopology = app.PDBFile(self.pdb).topology
        # load ML potential parameters
        with open(self.file, 'rb') as ifile:
            params = pickle.load(ifile)
        self.params = params
        # convert to jnp array
        for k in params:
            params[k] = jnp.array(params[k])
            # set mask to all true
            paramset.addParameter(params[k], k, field=self.name, mask=jnp.ones(params[k].shape))

        # mask = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape), params)
        # paramset.addParameter(params, "params", field=self.name, mask=mask)
       

    def getName(self) -> str:
        return self.name

    def overwrite(self, params):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        with open(self.file, 'wb') as ofile:
            pickle.dump(paramset[self.name], ofile)
        return

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        n_atoms = topdata.getNumAtoms()
        n_elem, elem_indices = get_elem_indices(self.ommtopology)
        self.model = EANNForce(n_elem, elem_indices, n_gto=self.ngto, nipsin=self.nipsin, rc=self.rc)
        n_layers = self.model.n_layers
        
        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True
        
        def potential_fn(positions, box, pairs, params, aux=None):
            # convert unit to angstrom
            positions = positions * 10
            box = box * 10
            if has_aux:
                return self.model.get_energy(positions, box, pairs, params[self.name]), aux
            else:
                return self.model.get_energy(positions, box, pairs, params[self.name])

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["EANNForce"] = EANNGenerator


class BASEGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet, dtype=None):
        self.name = "BASEForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None
        ffmeta = self.ffinfo["Forces"][self.name]["meta"]
        self.ckpt_file = None
        self.state_dict_file = None
        self.config_file = None
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        if "ckpt" in ffmeta:
            self.ckpt_file = ffmeta["ckpt"]
        if "state_dict" in ffmeta:
            self.state_dict_file = ffmeta["state_dict"]
            self.config_file = ffmeta["config"]

        self.model = self._initialize_model()
        
        # now model is fully loaded, start to register parameters
        named_parameters = self.model.named_parameters()
        self.params_t = OrderedDict()
        for name, param in named_parameters:
            self.params_t[name] = param
        self.params = t2j_pytree(self.params_t)
        for k in self.params:
            # set mask to all true
            paramset.addParameter(self.params[k], k, field=self.name, mask=jnp.ones(self.params[k].shape))

        self.params_noopt = OrderedDict()
        state_dict = self.model.state_dict()
        for k in state_dict:
            if k not in self.params_t:
                self.params_noopt[k] = state_dict[k]

        return


    def _initialize_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        if self.device == "cuda" and self.ckpt_file is not None:
            model = torch.jit.load(self.ckpt_file, map_location=self.device)
            self.state_dict_file = re.sub('.pt$', '_sd.pt', self.ckpt_file)
        else:
            # checkpoint = torch.load(self.state_dict_file, map_location=self.device)
            state_dict = torch.load(self.state_dict_file, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            args = get_parser(self.config_file)
            act_fn_map = {'ELU': nn.ELU(), 'CELU': nn.CELU(), 'GELU': nn.GELU(), 'SiLU': nn.SiLU(), 'Mish': nn.Mish(), 'Softplus': nn.Softplus()}
            # model configurations from config file
            nn_params = {
                    'dim': args.emb_dim,
                    'num_rbf': args.num_rbf,
                    'rcut': args.rcut,
                    'act_fn': act_fn_map[args.act_fn],
                    'energy_mlp_layers': args.energy_mlp_layers,
                    'onehot': args.onehot,
                    'cutoff_type': args.cutoff_type,
                    'repulsion_cut': args.repulsion_cut,
                    }
            gnn_params = {
                    'n_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'act_fn': act_fn_map[args.act_fn]
                    }
            model_name_upper = str(args.model_name).upper()

            if model_name_upper == 'ET':
                from base.models.base_et import BaseET
                model = BaseET(device = self.device,
                                dtype = self.dtype,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
            elif model_name_upper == 'DET':
                from base.models.base_det import BaseDET
                model = BaseDET(device = self.device,
                                dtype = self.dtype,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
            elif model_name_upper == 'VISNET':
                from base.models.base_visnet import BaseVisNet
                model = BaseVisNet(device = self.device,
                                dtype = self.dtype,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
            elif model_name_upper == 'TENSORNET':
                from base.models.base_tensornet import BaseTensorNet
                model = BaseTensorNet(device = self.device,
                                dtype = self.dtype,
                                nn_params = nn_params,
                                gnn_params = gnn_params)
            else:
                raise NotImplementedError('Supported model: ET, DET, VISNET')
            model.load_state_dict(state_dict)
        return model


    def getName(self) -> str:
        return self.name

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        self.topdata = topdata
        self.n_atoms = topdata.getNumAtoms()
        self.input = {}
        self.input['atom_types'] = []
        self.rc = nonbondedCutoff._value * 10 # in A
        for atom in topdata.atoms():
            element = atom.element.upper()
            self.input['atom_types'].append(elem_to_index[element])
        self.input['atom_types'] = torch.tensor(self.input['atom_types'])
        self.input['cumsum_atom'] = torch.tensor([0, self.n_atoms])

        # torch kernel
        def potential_torch_kernel(positions, box, pairs, params, filter_atoms=None, flambda=0.0):
            # Assuming all inputs are in torch tensor
            # Note here we build another pair list, the input variable pairs is only a placeholder
            if filter_atoms is None:
                filter_atoms = torch.tensor(np.ones(positions.shape[0]), dtype=torch.int16, device=self.device)
            else:
                filter_atoms = torch.tensor(filter_atoms, dtype=torch.int16, device=self.device)
            flambda = torch.tensor(flambda)
            crys_keys = ['lattice', 'cumsum_atom', 'cumsum_edge']
            atom_keys = ['pos', 'atom_types']
            edge_keys = ['center_index', 'neighbor_index', 'edge_shift', 'image']

            input = self.input
            # input in nm, convert to A
            input['lattice'] = box * 10
            input['pos'] = positions * 10
            structure = pymatgen.core.structure.Structure(input['lattice'].cpu(), 
                                                          self.input['atom_types'].cpu(), 
                                                          input['pos'].cpu(), 
                                                          0,
                                                          coords_are_cartesian=True)

            center_index, neighbor_index, edge_shift, image = self.get_edge(structure,
                                                                            input['pos'].to(self.device),
                                                                            input['lattice'].to(self.device),
                                                                            self.rc)
            input['center_index'] = center_index
            input['neighbor_index'] = neighbor_index
            
            # shifting edges, turn-off interactions between ghost atoms and other atoms
            # used for ghost atom simulations
            filter1 = filter_atoms[center_index]
            filter2 = filter_atoms[neighbor_index]
            filter_edges = filter1 * filter2
            scale_edge = (1-filter_edges) * 1.0e6 + filter_edges
            input['filter_atoms'] = filter_atoms
            input['flambda'] = flambda
            edge_shift = edge_shift * scale_edge[:, torch.newaxis]

            input['edge_shift'] = edge_shift
            # shift
            input['image'] = image
            n_edges = len(center_index)
            input['cumsum_edge'] = torch.tensor([0, n_edges])

            input['lattice'] = torch.stack([input['lattice']])

            for k in input.keys():
                input[k] = input[k].to(self.device)

            # load parameter to model
            if self.name in params.keys():
                state_dict = params[self.name]
            else:
                state_dict = params

            # build a model object for every invokation to avoid gradient accumulation
            model = self._initialize_model()
            model.load_state_dict(state_dict, strict=False)
            results = model.forward(input)
            return results, model

        # jax wrapper
        @partial(jax.custom_vjp, nondiff_argnums=(2,4,5))
        def potential_fn(position, box, pairs, params, filter_atoms=None, flambda=0.0):
            '''
            potential with ghost atoms
            positions: atom positions (the full dimensional array including ghost atoms)
            box: box size (3x3 matrix, lattice vector arranged in rows)
            pairs: atom pairs, simply use None for BASE
            filter_atoms: a numpy array using 1/0 to label real/ghost atoms
            flambda: a scaling factor used to control the magnitude of the ghost-atom interactions
                     flambda = 1.0 means full interaction, 0.0 means no interaction
                     Right now it is simply a placeholder, detailed swithing mechanism is not implemented
            '''
            position_t = j2t(position)
            box_t = j2t(box)
            params_t = j2t_pytree(params)
            result, model = potential_torch_kernel(position_t, box_t, None, params_t, 
                                                   filter_atoms=filter_atoms,
                                                   flambda=flambda)
            return t2j(result['pred_energy'][0]) * EV2KJ

        def potential_fwd(positions, box, pairs, params, filter_atoms=None, flambda=0.0):
            # gradient of positions and box will be computed internally and returned
            # by force an virial
            position_t = j2t(positions).detach()
            box_t = j2t(box).detach()
            position_t.requires_grad_(False)
            box_t.requires_grad_(False)
            params_t = j2t_pytree(params)
            result, model = potential_torch_kernel(position_t, box_t, None, params_t, 
                                                   filter_atoms=filter_atoms,
                                                   flambda=flambda)
            model.zero_grad()
            result['pred_energy'].backward()

            inputs = {'pos': positions,
                      'box': box,
                      'params': params
                    }
            energy = t2j(result['pred_energy'][0])
            dE_dp = jax.tree.map(lambda x: jnp.zeros(x.shape), inputs['params'])
            # read parameter gradient from the model
            for name, param in model.named_parameters():
                dE_dp[self.name][name] = t2j_extract_grad(param)
            return energy*EV2KJ, (t2j_pytree(result), inputs, dE_dp)

        def potential_bwd(pairs, filter_atoms, flambda, res, g):
            preds = res[0]
            inputs = res[1]
            dE_dp = res[2]
            # dE_dp = jax.tree.map(lambda x: jnp.zeros(x.shape), inputs['params'])
            # # read parameter gradient from the model
            # for name, param in model.named_parameters():
            #     dE_dp[self.name][name] = t2j_extract_grad(param)
            force = preds['pred_forces']
            # virial is -V\tau
            virial = preds['pred_virial'][0] # in eV
            pos = inputs['pos'] # in nm
            box = inputs['box'] # in nm
            box_inv = jnp.linalg.inv(box) # in nm-1
            # note force is in eV/A, but pos is in nm...
            dE_dB = box_inv.T@(pos.T@force*10 - virial)
            dE_dr = -force
            # unit conversion from eV/A to kJ/mol/nm
            return dE_dr*g*EV2KJ*10, dE_dB*g*EV2KJ, jax.tree.map(lambda x: x*g*EV2KJ, dE_dp)

        potential_fn.defvjp(potential_fwd, potential_bwd)

        return potential_fn

    def write_to(self, params, state_dict_file):
        if 'BASEForce' in params:
            self.params = params['BASEForce']
        else:
            self.params = params
        self.params_t = j2t_pytree(self.params)
        state_dict = copy.deepcopy(self.params_t)
        for k in self.params_noopt:
            state_dict[k] = self.params_noopt[k]
        torch.save(state_dict, state_dict_file)
        return

    def overwrite(self, params):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        self.write_to(params, self.state_dict_file)
        return

    def get_edge(self, structure, pos, lattice_tensor, cutoff):
        pos = pos.clone().detach()
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(r=cutoff, sites=structure.sites, numerical_tol=1e-8)
        image = torch.tensor(image, dtype=self.dtype, device=self.device)
        # edge_shift with periodicity correction
        edge_shift = pos[neighbor_index] - pos[center_index] + image @ lattice_tensor # cart coord

        center_index = torch.tensor(center_index, dtype=torch.long, device=self.device)
        neighbor_index = torch.tensor(neighbor_index, dtype=torch.long, device=self.device)
        return center_index, neighbor_index, edge_shift, image

    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["BASEForce"] = BASEGenerator



class CustomTorchGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet, dtype=None):
        """
        A custom torch model is specified by a full model checkpoint file
        The xml front end should be:

        ```xml
        <ForceField>
           <CustomTorchForce ckpt="model.pt" torch_script="True" dtype="float32"/>
        </ForceField>
        ```
        """

        self.name = "CustomTorchForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None
        ffmeta = self.ffinfo["Forces"][self.name]["meta"]
        self.ckpt_file = None
        self.state_dict_file = None
        self.config_file = None
        self.torch_script = False

        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        # precision
        if "dtype" in ffmeta["dtype"]:
            if '32' in ffmeta["dtype"]:
                self.dtype = torch.float32
            elif '64' in ffmeta["dtype"]:
                self.dtype = torch.float64
        self.ckpt_file = ffmeta["ckpt"]
        self.torch_script = (ffmeta["torch_script"] == 'True')

        if self.torch_script:
            self.load_method = torch.jit.load
            self.save_method = torch.jit.save
        else:
            self.load_method = partial(torch.load, weights_only=False)
            self.save_method = torch.save

        self.model = self._initialize_model()
        
        # now model is fully loaded, start to register parameters
        named_parameters = self.model.named_parameters()
        self.params_t = OrderedDict()
        for name, param in named_parameters:
            self.params_t[name] = param
        self.params = t2j_pytree(self.params_t)
        for k in self.params:
            # set mask to all true
            paramset.addParameter(self.params[k], k, field=self.name, mask=jnp.ones(self.params[k].shape))

        self.params_noopt = OrderedDict()
        state_dict = self.model.state_dict()
        for k in state_dict:
            if k not in self.params_t:
                self.params_noopt[k] = state_dict[k]
        return

    
    def _initialize_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.load_method(self.ckpt_file, map_location=self.device)
        # state dictionary file
        self.state_dict_file = re.sub('.([a-zA-Z0-9]+)$', '_sd.\g<1>', self.ckpt_file)
        return model

    def getName(self) -> str:
        return self.name

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        self.topdata = topdata
        self.n_atoms = topdata.getNumAtoms()

        # topo(primarily atom type) data
        self.atom_types = []
        for atom in topdata.atoms():
            element = atom.element.upper()
            self.atom_types.append(elem_to_index[element])
        self.atom_types = np.array(self.atom_types)

        # torch kernel
        def potential_torch_kernel(positions, box, pairs, params):

            # load parameter to model
            if self.name in params.keys():
                state_dict = params[self.name]
            else:
                state_dict = params

            # build a model object for every invokation to avoid gradient accumulation
            model = self._initialize_model()
            model.load_state_dict(state_dict, strict=False)
            results = model.forward(positions, box, self.atom_types)
            return results, model

        # jax wrapper
        @partial(jax.custom_vjp, nondiff_argnums=(2,))
        def potential_fn(position, box, pairs, params):
            position_t = j2t(position)
            box_t = j2t(box)
            params_t = j2t_pytree(params)
            results, model = potential_torch_kernel(position_t, box_t, None, params_t)
            return t2j(result['pred_energy'])

        def potential_fwd(positions, box, pairs, params):
            # gradient of positions and box will be computed internally and returned
            # by force an virial
            position_t = j2t(positions).detach()
            box_t = j2t(box).detach()
            position_t.requires_grad_(False)
            box_t.requires_grad_(False)
            params_t = j2t_pytree(params)
            result, model = potential_torch_kernel(position_t, box_t, None, params_t)
            model.zero_grad()
            result['pred_energy'].backward()

            inputs = {'pos': positions,
                      'box': box,
                      'params': params
                    }
            energy = t2j(result['pred_energy'])
            dE_dp = jax.tree.map(lambda x: jnp.zeros(x.shape), inputs['params'])
            # read parameter gradient from the model
            for name, param in model.named_parameters():
                dE_dp[self.name][name] = t2j_extract_grad(param)
            return energy, (t2j_pytree(result), inputs, dE_dp)

        def potential_bwd(pairs, res, g):
            preds = res[0]
            inputs = res[1]
            dE_dp = res[2]
            force = preds['pred_forces']
            # virial is in kJ/mol
            virial = preds['pred_virial']
            pos = inputs['pos'] # in nm
            box = inputs['box'] # in nm
            box_inv = jnp.linalg.inv(box) # in nm-1
            # force in kJ/mol/nm, positions in nm
            dE_dB = box_inv.T@(pos.T@force - virial)
            dE_dr = -force
            # unit conversion from eV/A to kJ/mol/nm
            return dE_dr*g, dE_dB, jax.tree.map(lambda x: x*g, dE_dp)

        potential_fn.defvjp(potential_fwd, potential_bwd)

        return potential_fn


    def write_to(self, params, ckpt_file, state_dict_file):
        if self.name in params:
            self.params = params[self.name]
        else:
            self.params = params
        self.params_t = j2t_pytree(self.params)
        state_dict = copy.deepcopy(self.params_t)
        for k in self.params_noopt:
            state_dict[k] = self.params_noopt[k]
        # save state dictiontary file
        torch.save(state_dict, state_dict_file)
        # save the full model checkpoint
        self.model.load_state_dict(state_dict)
        self.save_method(self.model, ckpt_file)
        return

    def overwrite(self, params):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        self.write_to(params, self.ckpt_file, self.state_dict_file)
        return


    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["CustomTorchForce"] = CustomTorchGenerator
