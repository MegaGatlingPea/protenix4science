import os
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "10.0"

import sys
sys.path.append('/home/wang_qing_han/protenix4science-Data_Condition_xujun')

import torch
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.data.featurizer import Featurizer
from biotite.structure import AtomArray
from protenix.data.parser import AddAtomArrayAnnot
import numpy as np
from protenix.data.constants import PROTEIN_REP_RESIDUE, DNA_REP_RESIDUE, RNA_REP_RESIDUE
from protenix.data.json_parser import add_reference_features
import biotite.structure as struc
import biotite.structure.io as strucio
from protenix.data.utils import int_to_letters
from protenix.data.ccd import get_component_atom_array

BB_TYPE_TO_ATOM = {
    'proteinbb': ["N", "CA", "C", "O"],
    'rnabb': ["C1'", "C2'", "C3'", "C4'", "C5'", "P", "OP1", "OP2", "O5'", "O4'", "O3'", "O2'"],
    'dnabb': ["C1'", "C2'", "C3'", "C4'", "C5'", "P", "OP1", "OP2", "O5'", "O4'", "O3'"],
    'ligandbb': ["-"]
}

def _build_bb_atom_array(num_residues, bb_type):
    chain = struc.AtomArray(0)
    chain.set_annotation("charge", np.array([]))
    chain.set_annotation("leaving_atom_flag", np.array([]))
    chain.set_annotation("alt_atom_id", np.array([]))
    chain.set_annotation("pdbx_component_atom_id", np.array([]))
    chain.set_annotation("mol_type", np.array([]))
    if bb_type == 'proteinbb':
        for res_id in range(num_residues):
            bb = get_component_atom_array(PROTEIN_REP_RESIDUE)
            mol_types = np.zeros(len(bb), dtype="U7")
            mol_types[:] = 'protein'
            bb.set_annotation('mol_type', mol_types)
            hetero = np.zeros(len(bb), dtype=bool)
            hetero[:] = False
            bb.hetero = hetero
            bb.res_id = np.array([res_id for _ in range(len(bb))])
            chain += bb
        #chain = add_reference_features(chain)
    
    elif bb_type == 'rnabb':
        for _ in range(num_residues):
            bb = get_component_atom_array(RNA_REP_RESIDUE)
            mol_types = np.zeros(len(bb), dtype="U7")
            mol_types[:] = 'rna'
            bb.set_annotation('mol_type', mol_types)
            hetero = np.zeros(len(bb), dtype=bool)
            hetero[:] = False
            bb.hetero = hetero
            bb.res_id = np.array([res_id for _ in range(len(bb))])
            chain += bb
        #chain = add_reference_features(chain)

    elif bb_type == 'dnabb':
        for res_id in range(num_residues):
            bb = get_component_atom_array(DNA_REP_RESIDUE)
            mol_types = np.zeros(len(bb), dtype="U7")
            mol_types[:] = 'dna'
            bb.set_annotation('mol_type', mol_types)
            hetero = np.zeros(len(bb), dtype=bool)
            hetero[:] = False
            bb.hetero = hetero
            bb.res_id = np.array([res_id for _ in range(len(bb))])
            chain += bb
        #chain = add_reference_features(chain)

    elif bb_type == 'ligandbb':
        for res_id in range(num_residues):
            bb = struc.AtomArray(0)
            bb.chain_id = np.array("", dtype='<U4')
            bb.res_id = np.array(0)
            bb.ins_code = np.array("", dtype='<U1')
            bb.res_name = np.array("-", dtype='<U5')
            bb.hetero = np.array(True, dtype=bool)
            bb.element = np.array("-", dtype='<U2')
            bb.charge = np.array(0)
            bb.leaving_atom_flag = np.array(0, dtype=bool)
            bb.alt_atom_id = np.array("-", dtype='<U4')
            bb.pdbx_component_atom_id = np.array("-", dtype='<U4')
            bb.mol_type = np.array('ligand', dtype=str)
            bb.res_id = np.array([res_id])
            chain += bb
        #chain.set_annotation("ref_pos", np.array([0, 0, 0]))
        #chain.set_annotation("ref_charge", np.array(0))
        #chain.set_annotation("ref_mask", np.array(1))

    chain.set_annotation("condition_atom_mask", np.zeros(len(chain), dtype=bool))
    chain.set_annotation("condition_token_mask", np.zeros(len(chain), dtype=bool))

    return chain

def _build_contig_atom_array(contig, contig_atom, pdb_file):
    stack = strucio.load_structure(pdb_file)
    charges = np.array([0] * len(stack))
    stack.set_annotation("charge", charges)
    chain_id = contig[0]
    res_begin = int(contig.split('-')[0][1:])
    res_end = int(contig.split('-')[1])
    contig_mask = (stack.chain_id == chain_id) & (np.isin(stack.res_id, np.arange(res_begin, res_end + 1)))
    contig_atom_array = stack[contig_mask]

    contig_atom_mask = contig_mask.copy()
    contig_token_mask = contig_mask.copy()
    for residue, contig_atom_str in contig_atom.items():
        contig_residue_id = residue[1:]
        contig_chain_id = residue[0]
        if int(contig_residue_id) in range(res_begin, res_end + 1) and contig_chain_id == chain_id:
            contig_atom_list = contig_atom_str.split(',')
            contig_atom_mask = contig_atom_mask & (np.isin(stack.atom_name, contig_atom_list))
            contig_token_mask = contig_token_mask & ~(stack.res_id == int(contig_residue_id))
    #contig_atom_array = add_reference_features(contig_atom_array)
    #contig_atom_array = contig_atom_array[contig_atom_mask]

    contig_atom_array.set_annotation("condition_atom_mask", contig_atom_mask[contig_mask])
    contig_atom_array.set_annotation("condition_token_mask", contig_token_mask[contig_mask])

    return contig_atom_array

def compose_bb_condition_atom_array(single_sample_dict):
    atom_array = struc.AtomArray(0)
    atom_array.set_annotation("charge", np.array([]))
    atom_array.set_annotation("leaving_atom_flag", np.array([]))
    atom_array.set_annotation("alt_atom_id", np.array([]))
    atom_array.set_annotation("pdbx_component_atom_id", np.array([]))
    atom_array.set_annotation("mol_type", np.array([]))
    atom_array.set_annotation("condition_atom_mask", np.array([]))
    atom_array.set_annotation("condition_token_mask", np.array([], dtype=bool))
    pdb_file = single_sample_dict["pdb"]
    fixed_atom = single_sample_dict["fixed_atom"]
    for fixed, fixed_type in zip(single_sample_dict["fixed"].split(','), single_sample_dict["fixed_type"].split(',')):
        if fixed_type == "proteinChain":
            fixed_atom_array = _build_contig_atom_array(fixed, fixed_atom, pdb_file)
            mol_types = ['protein'] * len(fixed_atom_array)
            fixed_atom_array.set_annotation('mol_type', mol_types)
            atom_array += fixed_atom_array
        elif fixed_type == "dnaChain":
            fixed_atom_array = _build_contig_atom_array(fixed, fixed_atom, pdb_file)
            mol_types = ['dna'] * len(fixed_atom_array)
            fixed_atom_array.set_annotation('mol_type', mol_types)
            atom_array += fixed_atom_array
        elif fixed_type == "rnaChain":
            fixed_atom_array = _build_contig_atom_array(fixed, fixed_atom, pdb_file)
            mol_types = ['rna'] * len(fixed_atom_array)
            fixed_atom_array.set_annotation('mol_type', mol_types)
            atom_array += fixed_atom_array
        elif fixed_type == "ligandChain":
            fixed_atom_array = _build_contig_atom_array(fixed, fixed_atom, pdb_file)
            mol_types = ['ligand'] * len(fixed_atom_array)
            fixed_atom_array.set_annotation('mol_type', mol_types)
            atom_array += fixed_atom_array
        else:
            num_residues = np.random.randint(int(fixed.split('-')[0]), int(fixed.split('-')[1]) + 1)
            fixed_atom_array = _build_bb_atom_array(num_residues, fixed_type)
            atom_array += fixed_atom_array

    return atom_array

def arrange_res_id(atom_array):
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    res_id = np.array(0)
    for start, stop in zip(starts[:-1], starts[1:]):
        atom_array.res_id[start:stop] = res_id
        res_id += 1
    return atom_array

class SampleDictToFeatures:
    def __init__(self, single_sample_dict):
        self.single_sample_dict = single_sample_dict
        self.atom_array = compose_bb_condition_atom_array(single_sample_dict)
        self.atom_array = arrange_res_id(self.atom_array)
        self.atom_array = add_reference_features(self.atom_array)
        self.entity_poly_type = self.get_entity_poly_type()

    def get_entity_poly_type(self) -> dict[str, str]:
        entity_type_mapping_dict = {
            "protein": "polypeptide(L)",
            "dna": "polydeoxyribonucleotide",
            "rna": "polyribonucleotide",
        }
        entity_poly_type = {}
        composed_mol_type = np.unique(self.atom_array.mol_type)
        for entity_id, entity_type in enumerate(composed_mol_type):
            if entity_type in entity_type_mapping_dict.keys():
                entity_poly_type[str(entity_id + 1)] = entity_type_mapping_dict[entity_type]
        return entity_poly_type
    
    def build_full_atom_array(self):
        atom_array = self.atom_array.copy()
        for entity_id, entity_type in enumerate(np.unique(self.atom_array.mol_type)):
            asym_id_str = int_to_letters(entity_id + 1)
            entity_mask = atom_array.mol_type == entity_type
            chain_id = atom_array.chain_id.copy()
            chain_id[entity_mask] = [asym_id_str] * entity_mask.sum()
            atom_array.set_annotation("label_asym_id", chain_id)
            atom_array.set_annotation("auth_asym_id", chain_id)
            atom_array.set_annotation("chain_id", chain_id)
            atom_array.set_annotation("label_seq_id", atom_array.res_id)
            atom_array.set_annotation("copy_id", [1] * len(atom_array))
            atom_array.set_annotation("label_entity_id", [entity_id + 1] * len(atom_array))
        return atom_array
    
    def add_atom_array_attributes(self, atom_array: AtomArray):
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
        atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
            atom_array, check_final_equiv=False
        )
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)
        return atom_array
    
    @staticmethod
    def mse_to_met(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI chapter 2.1
        MSE residues are converted to MET residues.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object after converted MSE to MET.
        """
        mse = atom_array.res_name == "MSE"
        se = mse & (atom_array.atom_name == "SE")
        atom_array.atom_name[se] = "SD"
        atom_array.element[se] = "S"
        atom_array.res_name[mse] = "MET"
        atom_array.hetero[mse] = False
        return atom_array
    
    def get_atom_array(self) -> AtomArray:
        atom_array = self.build_full_atom_array()
        atom_array = self.mse_to_met(atom_array)
        atom_array = self.add_atom_array_attributes(atom_array)
        return atom_array

    def get_feature_dict(self) -> tuple[dict[str, torch.Tensor], AtomArray, TokenArray]:
        """
        Generates a feature dictionary from the input sample dictionary.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A TokenArray object.
        """
        atom_array = self.get_atom_array()

        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        featurizer = Featurizer(token_array, atom_array, data_condition='all')
        feature_dict = featurizer.get_all_input_features()

        token_array_with_frame = featurizer.get_token_frame(
            token_array=token_array,
            atom_array=atom_array,
            ref_pos=feature_dict["ref_pos"],
            ref_mask=feature_dict["ref_mask"],
        )

        # [N_token]
        feature_dict["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()

        # [N_token, 3]
        feature_dict["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()
        return feature_dict, atom_array, token_array

if __name__ == "__main__":
    import json
    input_json_dict = json.load(open("/home/wang_qing_han/protenix4science-Data_Condition_xujun/examples/bb_generation.json"))
    single_sample_dict = input_json_dict[0]

    sample2feat = SampleDictToFeatures(
        single_sample_dict,
    )
    feature_dict, atom_array, token_array = sample2feat.get_feature_dict()
    print()