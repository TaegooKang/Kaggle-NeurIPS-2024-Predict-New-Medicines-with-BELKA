import pandas as pd 
import numpy as np
import time
import os
import multiprocessing
import _pickle as  cPickle
import bz2
import indexed_bzip2 as ibz2

import torch_geometric

from typing import Any, Dict, List
from tqdm import tqdm

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'd '
        i += 1
    if hours > 0 and i <= 3:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 3:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 3:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 3:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f        

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))

x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def from_smiles(smiles: str, with_hydrogen: bool = True,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    x_atoms: List[int] = []
    x_features: List[List[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        x_atoms.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        features = (
            one_of_k_encoding(str(atom.GetChiralTag()), x_map['chirality'])
            + one_of_k_encoding(atom.GetTotalDegree(), x_map['degree'])
            + one_of_k_encoding(atom.GetFormalCharge(), x_map['formal_charge'])
            + one_of_k_encoding(atom.GetTotalNumHs(), x_map['num_hs'])
            + one_of_k_encoding(atom.GetNumRadicalElectrons(), x_map['num_radical_electrons'])
            + one_of_k_encoding(str(atom.GetHybridization()), x_map['hybridization'])
            + [atom.GetIsAromatic()]
            + [atom.IsInRing()]
        )
        features = np.packbits(features)
        x_features.append(features)
    
    
    x_atoms = np.expand_dims(np.array(x_atoms, dtype=np.uint8), axis=1)
    x_features = np.stack(x_features)
    
    x = np.concatenate([x_atoms, x_features], axis=1, dtype=np.uint8)
    
    edge_indices = []
    edge_types = []
    edge_attrs = []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_types.append(e_map['bond_type'].index(str(bond.GetBondType())))
        # edge_types.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e = (
            one_of_k_encoding(str(bond.GetStereo()), e_map['stereo'])
            + [bond.GetIsConjugated()]
            + [bond.IsInRing()]
        )
        e = np.packbits(e)

        edge_indices.append([i, j])
        # edge_indices.append([j, i])
        edge_attrs.append(e)
        # edge_attrs.append(e)
        
    edge_indices = np.array(edge_indices, dtype=np.uint8)
    edge_types = np.expand_dims(np.array(edge_types, dtype=np.uint8), axis=1)
    edge_attrs = np.stack(edge_attrs)
    edge_attrs = np.concatenate([edge_types, edge_attrs], axis=1, dtype=np.uint8)

    # if edge_index.numel() > 0:  # Sort indices.
    #     perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
    #     edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    
    return x, edge_indices, edge_attrs
    
    
def save_compressed_pickle(file, data):
    with bz2.BZ2File(file , 'w') as f:
        cPickle.dump(data, f)

def load_compressed_ibz2_pickle(file):
    with ibz2.open(file, parallelization=32) as f:
        data = cPickle.load(f)
    return data
    
    

if __name__ == '__main__':
        
    df_train = pd.read_parquet('./leash-BELKA/random_stratified_split/train.parquet')
    print('Read parquet done.')
    
    train_smiles = df_train['molecule_smiles'].values
    num_cpu = os.cpu_count()
    
    with multiprocessing.Pool(processes=num_cpu) as pool:
        train_graphs = list(tqdm(pool.imap(from_smiles, train_smiles), total=len(train_smiles)))
    
    s = time.time()
    save_compressed_pickle('./leash-BELKA/random_stratified_split/train_graphs.pickle.b2z', train_graphs)
    e = time.time()

    print(format_time(e-s))
  
    
    