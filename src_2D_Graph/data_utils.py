import torch
import numpy as np
import torch_geometric

from typing import Any, Dict, List

import _pickle as  cPickle
import indexed_bzip2 as ibz2

def load_compressed_ibz2_pickle(file):
    with ibz2.open(file, parallelization=32) as f:
        data = cPickle.load(f)
    return data

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

def from_smiles(smiles: str, with_hydrogen: bool = False,
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

    
    return x, edge_indices, edge_attrs


def drop_nodes(x, edge_index, edge_attr, aug_ratio):

    node_num, _ = x.shape
    edge_index = edge_index.T
    _, edge_num = edge_index.shape
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index_new = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        return x[idx_nondrop], np.array(edge_index_new), edge_attr[edge_mask]
        
    except:
        return x, edge_index.T, edge_attr
    
def make_batch_graph(graph, index=None, device='cpu', aug_prob=0.0, aug_ratio=0.0):
    if index is None:
        index = np.arange(len(graph)).tolist()
    
    xs = []
    edge_indices = []
    edge_attrs = []
    batch = []
    offset = 0
    for b, i in enumerate(index):
        x, edge_index, edge_attr = graph[i]
        if aug_prob > 0 and aug_ratio > 0:
            p = np.random.uniform(0, 1)
            if p < aug_prob:
                x, edge_index, edge_attr = drop_nodes(x, edge_index, edge_attr, aug_ratio)
        
        N = x.shape[0]
        xs.append(x)
        edge_attrs.append(np.concatenate([edge_attr, edge_attr], axis=0))
        edge_indices.append(np.concatenate([edge_index, edge_index[:,[1,0]]], axis=0, dtype=int) + offset)
        batch += N * [b]
        offset += N
    xs = torch.from_numpy(np.concatenate(xs)).to(device)
    edge_attrs = torch.from_numpy(np.concatenate(edge_attrs)).to(device)
    edge_indices = torch.from_numpy(np.concatenate(edge_indices).T).to(device)
    batch = torch.LongTensor(batch).to(device)
    
    return xs, edge_indices, edge_attrs, batch  
    
def graph_loader(queue, graphs, targets, batch_size, shuffle=False, drop_last=False, aug_prob=0.0, aug_ratio=0.0):
    g_index = [i for i in range(len(graphs))]
    if shuffle: np.random.shuffle(g_index)
    for _, index in enumerate(np.arange(0, len(g_index), batch_size)):
        index = g_index[index:index+batch_size]
        if len(index)!= batch_size: 
            if drop_last: continue #drop last

        batch = {
			'graphs': make_batch_graph(graphs, index, device='cpu', aug_prob=aug_prob, aug_ratio=aug_ratio),
			'targets': torch.tensor(targets[index], dtype=torch.float)
		}
        queue.put(batch)
    queue.put(None)
    
    
