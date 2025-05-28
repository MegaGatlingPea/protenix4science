#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
from biotite.structure import AtomArray
from joblib import load

from protenix.data.tokenizer import TokenArray
from protenix.data.constants import STD_RESIDUES, PROTEIN_BACKBONE_ATOM_NAMES, NA_BACKBONE_ATOM_NAMES, BACKBONE_ATOM_NAMES

import pandas as pd

class MaskGenerator:
    """Class for generating masks for protein structures at different levels (entity, token, atom)."""
    
    O_INDEX = 3
    NHEAVYPROT = 14

    def __init__(self, token_array: TokenArray, atom_array: AtomArray):
        """
        Initialize the MaskGenerator.
        
        Args:
            token_array: TokenArray representing the protein tokens
            atom_array: AtomArray representing the protein atoms
        """
        self.token_array = token_array
        self.atom_array = atom_array
        
        
        # Identify standard tokens
        self.center_atoms_array = atom_array[token_array.get_annotation('centre_atom_index')]
        self.is_ligand_token = self.center_atoms_array.mol_type == 'ligand'
        self.is_std_token = (np.isin(self.center_atoms_array.res_name, list(STD_RESIDUES.keys()))) | (self.is_ligand_token)
        self.std_entity_ids = self.center_atoms_array[self.is_std_token].entity_id_int
        self.unique_std_entity_ids = np.unique(self.std_entity_ids)
    
    @staticmethod
    def choose_furthest_from_oxygen(res_atom_array: AtomArray, token_encoding, seed: Optional[int] = None) -> int:
        """
        Choose the furthest atom away from oxygen.
        
        Args:
            res_atom_array: AtomArray for the residue
            token_encoding: Token encoding information
            seed: Random seed for reproducibility
            
        Returns:
            Index of the chosen atom
        """
        if seed is not None:
            random.seed(seed)
            
        bond_feats = MaskGenerator.get_residue_bond_feats(res_atom_array, token_encoding)
        bond_graph = nx.from_numpy_array(bond_feats.numpy())
        at_dist = MaskGenerator.nodes_at_distance(bond_graph, MaskGenerator.O_INDEX)
        furthest = at_dist[-1]
        return random.choice(furthest)
    
    @staticmethod
    def get_residue_bond_feats(res_atom_array: AtomArray, token_encoding, include_H: bool = False) -> torch.Tensor:
        """
        Get ground truth bonds for each residue.
        
        Args:
            res_atom_array: AtomArray for the residue
            token_encoding: Token encoding information
            include_H: Whether to include hydrogen atoms
            
        Returns:
            2D tensor of bond information (bond_feats[i, j] represents the bond type between atom i and j)
        """
        n_atoms = len(token_encoding.atom_indices)
        bond_feats = torch.zeros((n_atoms, n_atoms))
        bonds = res_atom_array.bonds.as_array()
        
        for j, bond in enumerate(bonds):
            start_idx, end_idx, bond_type = bond[0], bond[1], bond[2]
            bond_feats[start_idx, end_idx] = bond_type
            bond_feats[end_idx, start_idx] = bond_type
            
        if not include_H:
            bond_feats = bond_feats[:MaskGenerator.NHEAVYPROT, :MaskGenerator.NHEAVYPROT]
            
        return bond_feats
    
    @staticmethod
    def nodes_at_distance(G: nx.Graph, start: int) -> List[List[int]]:
        """
        Generate a list of nodes at different distances from a specified starting node in a graph.
        
        Args:
            G: The input graph
            start: The starting node for distance calculation
            
        Returns:
            A list where at_dist[i] contains all nodes that are i edges away from the starting node
        """
        shortest_paths = nx.single_source_shortest_path_length(G, source=start)
        at_dist = [[] for _ in range(max(shortest_paths.values()) + 1)]
        
        for node, distance in shortest_paths.items():
            at_dist[distance].append(node)
            
        return at_dist
    
    def get_atom_names_within_n_bonds(self, cur_res_atom_array: AtomArray, token_encoding, 
                                       source_node: int, n_bonds: int) -> Set[int]:
        """
        Get atom names within n_bonds which will constitute the motif.
        
        Args:
            cur_res_atom_array: AtomArray for the current residue
            token_encoding: Token encoding information
            source_node: Source node index
            n_bonds: Number of bonds to consider
            
        Returns:
            Set of atom indices that are within n_bonds
        """
        bond_feats = self.get_residue_bond_feats(cur_res_atom_array, token_encoding)
        bond_graph = nx.from_numpy_array(bond_feats.numpy())
        paths = nx.single_source_shortest_path_length(bond_graph, source=source_node, cutoff=n_bonds)
        atoms_within_n_bonds = paths.keys()
        return list(atoms_within_n_bonds)
    
    def get_tip_atom_indices(self, token) -> np.ndarray:
        """
        Get tip atom indices for a token.
        
        Args:
            token: Token object
            
        Returns:
            Array of tip atom indices
        """
        cur_res_atom_array = self.atom_array[token.atom_indices]
        
        # Select seed atom
        if np.random.rand() < 0.8:
            # Choose atom furthest from backbone oxygen
            seed_atom = self.choose_furthest_from_oxygen(cur_res_atom_array, token)
        else:
            # Randomly choose an atom
            seed_atom = np.random.choice(range(len(token.atom_indices)), 1)[0]
            
        # Sample from geometric distribution to determine how many chemical bonds to extend
        n_bonds = np.random.geometric(p=0.5) - 1
        
        # Get atoms around the seed atom within n_bonds
        motif_atom_indices = self.get_atom_names_within_n_bonds(
            cur_res_atom_array, token, seed_atom, n_bonds)
            
        return np.array(token.atom_indices)[motif_atom_indices]
    

    def get_sym_tip_indices(self, token) -> np.ndarray:
        """
        Get symmetric tip atom indices for a token.
        
        Args:
            token: Token object
            
        Returns:
            Array of symmetric tip atom indices
        """

        tip_atom_indices = self.get_tip_atom_indices(token)

        sym_tip_atom_indices = np.array([])
        for atom in self.atom_array[tip_atom_indices]:
            atom_name = f'{atom.entity_id_int}_{atom.mol_atom_index}'
            atom_indices = np.asarray(self.sym_atom_name.get(atom_name, []))
            sym_tip_atom_indices = np.append(sym_tip_atom_indices, atom_indices)
        return sym_tip_atom_indices.astype(int)

    

    def get_sym_token_mask(self, asym_token_mask: np.ndarray) -> np.ndarray:
        """
        Get symmetry token mask based on asymmetry token mask.
        
        Args:
            asym_token_mask: Boolean array indicating which tokens are asymmetric
            
        Returns:
            Boolean array indicating which tokens are symmetric
        """
        sym_atom_name = {}
        for idx, rep_atom in enumerate(self.center_atoms_array):
            atom_name = f'{rep_atom.entity_mol_id}_{rep_atom.mol_atom_index}'
            id_lis = sym_atom_name.get(atom_name, [])
            id_lis.append(idx)
            sym_atom_name[atom_name] = id_lis

        sym_token_mask = copy.deepcopy(asym_token_mask)
        
        for rep_atom in self.center_atoms_array[~asym_token_mask]:  # 看哪些token没有被mask，如果其对称的token被mask，则也要mask
            atom_name = f'{rep_atom.entity_mol_id}_{rep_atom.mol_atom_index}'
            atom_indices = sym_atom_name.get(atom_name, [])
            atom_mask = asym_token_mask[atom_indices]
            if atom_mask.any():
                sym_token_mask[atom_indices] = True
        return sym_token_mask
    

    
    def generate_entity_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mask at entity level (mask entire chains/entities).
        
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask)
        """
        # Randomly select entities to mask
        masked_entity_ids = np.random.choice(
            self.unique_std_entity_ids, 
            size=np.random.randint(1, len(self.unique_std_entity_ids)), 
            replace=False
        )
        
        # Create token-level mask
        is_masked_token = np.isin(self.center_atoms_array.entity_id_int, masked_entity_ids) & self.is_std_token
        
        # Convert to atom-level mask
        return self.token_to_atom_mask(is_masked_token)
    
    def generate_token_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mask at token level (mask continuous segments of tokens).
        
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask)
        """
        is_masked_token = np.zeros(len(self.token_array), dtype=bool)
        std_indices = np.where(self.is_std_token)[0]
        left_length = len(std_indices)
        left_seq_num = masked_seq_num = np.random.randint(1, 5)  # Randomly select 1-4 continuous segments
        start_idx = 0
        
        for _ in range(masked_seq_num):
            avg_masked_seq_length = left_length // left_seq_num  # 201 // 2 = 100
            mask_length = np.random.randint(1, avg_masked_seq_length + 1)  # 1-100
            tmp_start_idx = np.random.randint(start_idx, start_idx + avg_masked_seq_length - mask_length + 1) if avg_masked_seq_length - mask_length > 0 else start_idx  # 排除 avg_masked_seq_length == mask_length 的极限情况 0
            is_masked_token[std_indices[tmp_start_idx:tmp_start_idx + mask_length]] = True
            
            left_length -= mask_length
            left_seq_num -= 1
            start_idx += mask_length
            
        return self.token_to_atom_mask(is_masked_token)
    
    def generate_atom_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mask at atom level (mask non-backbone atoms, keeping some tip atoms).
        
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask)
        """
        is_masked_token = self.is_std_token
        # Randomly select 20% of standard tokens to preserve tip atoms
        tipatom_condition_token = (np.random.rand(len(self.token_array)) < 0.2) & self.is_std_token
        # generate sym atom name
        self.sym_atom_name = {}
        for idx, rep_atom in enumerate(self.atom_array):
            atom_name = f'{rep_atom.entity_id_int}_{rep_atom.mol_atom_index}'
            id_lis = self.sym_atom_name.get(atom_name, [])
            id_lis.append(idx)
            self.sym_atom_name[atom_name] = id_lis
        # return token_to_atom_mask
        return self.token_to_atom_mask(is_masked_token, tipatom_condition_token)
    
    def generate_all_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mask for all standard tokens (keeping only backbone atoms).
        
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask)
        """
        is_masked_token = self.is_std_token
        return self.token_to_atom_mask(is_masked_token)
    
    def generate_mask(self, mask_method: str = '') -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate mask according to specified method.
        
        Args:
            mask_method: Method for generating masks ('entity', 'token', 'atom', 'all', or '' for random)
            
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask, used_mask_method)
        """
        # Validate mask method
        valid_methods = ['entity', 'token', 'atom', 'all', '']
        if mask_method not in valid_methods:
            raise ValueError(f"mask_method must be one of {', '.join(valid_methods)}")
            
        # If not specified, randomly select a method
        if mask_method == '':
            if len(self.unique_std_entity_ids) > 1:
                methods = ['entity', 'token', 'atom', 'all']
                probabilities = [0.3, 0.3, 0.3, 0.1]
            else:
                methods = ['token', 'atom', 'all']
                probabilities = [0.5, 0.3, 0.2]
                
            mask_method = np.random.choice(methods, p=probabilities)
        
        # print(f'Using mask method: {mask_method}')
        self.mask_method = mask_method

        # Generate mask based on selected method
        if mask_method == 'entity':
            is_condition_atom, is_condition_token = self.generate_entity_mask()
        elif mask_method == 'token':
            is_condition_atom, is_condition_token = self.generate_token_mask()
        elif mask_method == 'atom':
            is_condition_atom, is_condition_token = self.generate_atom_mask()
        elif mask_method == 'all':
            is_condition_atom, is_condition_token = self.generate_all_mask()



        return is_condition_atom, is_condition_token, mask_method
    

    def _reset_entity_sym_id_int(self, token_mask: np.ndarray):
        """
        Reset entity_id_int and sym_id_int for masked atoms.
        """
        self.atom_array.set_annotation('entity_id_int_label', copy.deepcopy(self.atom_array.get_annotation('entity_id_int')))
        self.atom_array.set_annotation('sym_id_int_label', copy.deepcopy(self.atom_array.get_annotation('sym_id_int')))
        entity_id_int = self.atom_array.entity_id_int
        sym_id_int = self.atom_array.sym_id_int
        asym_id_int_list = np.unique(self.atom_array.asym_id_int)
        new_entity_id_int = np.unique(entity_id_int).shape[0]
        old_sym_id_int = 0
        new_sym_id_int = 0
        old_entity_id_int = ''
        for asym_id_int in asym_id_int_list:
            asym_id_int_mask = self.atom_array.asym_id_int == asym_id_int
            if token_mask[asym_id_int_mask].any():
                new_entity_id_int += 1
                # print(f'asym_id_int: {asym_id_int} | entity_id from {entity_id_int[asym_id_int_mask][0]} to {new_entity_id_int} | sym_id_int from {sym_id_int[asym_id_int_mask][0]} to {new_sym_id_int}')
                entity_id_int[asym_id_int_mask] = new_entity_id_int
                sym_id_int[asym_id_int_mask] = new_sym_id_int
            else:
                if old_entity_id_int != entity_id_int[asym_id_int_mask][0]:
                    old_entity_id_int = entity_id_int[asym_id_int_mask][0]
                    old_sym_id_int = 0
                sym_id_int[asym_id_int_mask] = old_sym_id_int
                old_sym_id_int += 1
                # print(f'asym_id_int: {asym_id_int} | entity_id {entity_id_int[asym_id_int_mask][0]} | sym_id_int {sym_id_int[asym_id_int_mask][0]}')
        self.atom_array.set_annotation('entity_id_int', entity_id_int)
        self.atom_array.set_annotation('sym_id_int', sym_id_int)
        
    def _crop_side_chain_atoms(self, condition_atom_mask: np.ndarray):
        """
        Crop side chain atoms for masked atoms.
        """
        cropped_atom_indices = []
        totol_atom_num = 0
        for idx, token in enumerate(self.token_array):
            # 如果token被mask，并且不是配体，则移除side chain原子
            if self.is_masked_token[idx] and not self.is_ligand_token[idx]:
                token_mask = condition_atom_mask[token.atom_indices]
                token.atom_indices = np.array(token.atom_indices)[token_mask].tolist()
                token.atom_names = np.array(token.atom_names)[token_mask].tolist()
                
            cropped_atom_indices.extend(token.atom_indices)
            centre_idx_in_token_atoms = token.atom_indices.index(
                token.centre_atom_index
            )
            token_atom_num = len(token.atom_indices)
            token.atom_indices = list(
                range(totol_atom_num, totol_atom_num + token_atom_num)
            )
            token.centre_atom_index = token.atom_indices[centre_idx_in_token_atoms]
            totol_atom_num += token_atom_num

        cropped_atom_array = copy.deepcopy(self.atom_array[cropped_atom_indices])
        cropped_token_array = copy.deepcopy(self.token_array)
        return cropped_atom_array, cropped_token_array

    def token_to_atom_mask(self, is_masked_token: np.ndarray, 
                           tipatom_condition_token: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert token-level mask to atom-level mask.
        
        Args:
            is_masked_token: Boolean array indicating which tokens are masked
            tipatom_condition_token: Boolean array indicating which tokens should preserve tip atoms
            
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask)
        """
        if tipatom_condition_token is None:
            tipatom_condition_token = np.zeros(len(self.token_array), dtype=bool)
        
        # 获取对称的mask
        self.is_masked_token = self.get_sym_token_mask(is_masked_token)
        # Initialize atom-level masks
        is_condition_atom = np.zeros(len(self.atom_array), dtype=bool)  
        is_condition_token = np.zeros(len(self.atom_array), dtype=bool)
        is_protein_token = self.center_atoms_array.mol_type == 'protein'
        for idx, token in enumerate(self.token_array):
            if self.is_masked_token[idx]:
                # 针对蛋白和核酸，保留backbone原子
                if ~self.is_ligand_token[idx]:
                    # For masked tokens, keep backbone atoms
                    # 被mask的token，起码backbone原子是保留的
                    mask_atom_indices = np.array(token.atom_indices)[np.isin(token.atom_names, BACKBONE_ATOM_NAMES)]
                    # Add tip atoms if specified
                    if tipatom_condition_token[idx]:
                        tip_atom_indices = np.array([])
                        if is_protein_token[idx] and np.isin(PROTEIN_BACKBONE_ATOM_NAMES, token.atom_names).all():
                            try:
                                tip_atom_indices = self.get_sym_tip_indices(token)
                            except:
                                print(f"Error in get_tip_atom_indices for token {idx}")
                        
                        mask_atom_indices = np.append(mask_atom_indices, tip_atom_indices).astype(int)
                else:  # 配体分子，保留所有原子
                    mask_atom_indices = np.array(token.atom_indices)
                
                is_condition_atom[mask_atom_indices] = True
            else:
                # For unmasked tokens, keep all atoms
                # 没有被mask的token，包括非标准氨基酸和配体分子
                is_condition_token[token.atom_indices] = True
                is_condition_atom[token.atom_indices] = True
        
        # csv_file = '/home/xujunzhang/protenix4science/mask_ratio.csv'
        # if not os.path.exists(csv_file):
        #     pd.DataFrame({
        #         'mask_method': [],
        #         'asym_masked_token_ratio': [],
        #         'sym_masked_token_ratio': [],
        #         'sym_masked_atom_ratio': []
        #     }).to_csv(csv_file, index=False)
        # pd.DataFrame({
        #     'mask_method': [self.mask_method],
        #     'asym_masked_token_ratio': [is_masked_token.sum() / len(self.token_array)],
        #     'sym_masked_token_ratio': [self.is_masked_token.sum() / len(self.token_array)],
        #     'sym_masked_atom_ratio': [1 - (is_condition_atom.sum() / len(self.atom_array))]
        # }).to_csv(csv_file, index=False, mode='a', header=False)

        return is_condition_atom, is_condition_token


    def apply_mask(self, data_condition: str = 'all', mask_method: str = '') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Apply mask according to specified method.
        
        Returns:
            Tuple of (condition_atom_mask, condition_token_mask, is_masked_token, masked_asym_ids, used_mask_method)
        """
        # 创建mask
        condition_atom_mask, condition_token_mask, method_used = self.generate_mask(mask_method)
        # 设置属性
        self.atom_array.set_annotation('condition_atom_mask', condition_atom_mask)
        self.atom_array.set_annotation('condition_token_mask', condition_token_mask)

        if data_condition in ['all', 'data']:
            # 移除 side chain atoms
            return self._crop_side_chain_atoms(condition_atom_mask)
        else:
            return self.atom_array, self.token_array


# Example usage
if __name__ == "__main__":
    # Load data 
    token_array, atom_array, selected_token_indices = load('token_array.pkl')
    
    # Create mask generator
    mask_generator = MaskGenerator(token_array, atom_array)
    
    for _ in range(10):
        # Generate mask
        condition_atom_mask, condition_token_mask, method_used, masked_asym_ids = mask_generator.generate_mask(mask_method='all')
        
        print(f"Generated using method: {method_used}")
        print(f"Condition atom sum: {condition_atom_mask.sum()}")
        print(f"Condition token sum: {condition_token_mask.sum()}")


