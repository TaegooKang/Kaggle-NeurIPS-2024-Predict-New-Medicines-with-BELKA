import pandas as pd
import numpy as np 
import _pickle as  cPickle
import bz2
import indexed_bzip2 as ibz2
import multiprocessing
import time
import os

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem



def save_compressed_pickle(file, data):
    with bz2.BZ2File(file , 'w') as f:
        cPickle.dump(data, f)

def load_compressed_ibz2_pickle(file):
    with ibz2.open(file, parallelization=32) as f:
        data = cPickle.load(f)
    return data

def remove_dy(smiles):
    return smiles.replace("[Dy]", "")
    

def get_3d_coordinates_and_atoms(smiles, idx):
    
    # Remove [Dy] in smiles 
    smiles = remove_dy(smiles)
    
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    try:
    # Generate 3D coordinates
        mol = Chem.AddHs(mol)  # Add hydrogens for a more accurate 3D structure
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # Embed the molecule in 3D space
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=30)  # Optimize the geometry using UFF force field
        mol = Chem.RemoveHs(mol)
        
        # Get the 3D coordinates
        conformer = mol.GetConformer()
        pos = conformer.GetPositions()
        pos = np.array(pos, dtype=np.float32)
            
        
        # Get atoms and their counts
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()] 
        z = np.expand_dims(np.array(z, dtype=np.int8), 1) # n x 1
        #atom_counts = {atom: atoms.count(atom) for atom in set(atoms)}
        
        graph = np.concatenate([z, pos], 1)
        np.save(f'/data2/local_datasets/leash-BELKA/3d-graphs/train/graph{idx}.npy', graph)
        
        return 1
    
    except:
        return 0
    


def test():
    smiles = "CCC"
    get_3d_coordinates_and_atoms((smiles, 0))
    

if __name__ == '__main__':

    df_path = './leash-BELKA/random_stratified_split/train.parquet'
    df = pd.read_parquet(df_path)
    print('Read parquet done.')
    
    smiles = df['molecule_smiles'].values
    id = [x for x in range(len(smiles))]
    data = [(s, idx) for s, idx in zip(smiles, id)]
    
    num_cpu = os.cpu_count()
    start = time.time()
    with multiprocessing.Pool(processes=num_cpu) as pool:
        v = list(tqdm(pool.starmap(get_3d_coordinates_and_atoms, zip(smiles, id)), total=len(data)))
    end = time.time()
    
    print(np.sum(v) / len(smiles))
    
    print(f"Total processing time: {end-start:.2f}s")
    
    