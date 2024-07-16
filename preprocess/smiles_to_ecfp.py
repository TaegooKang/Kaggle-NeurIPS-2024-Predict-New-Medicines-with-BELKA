import os
import numpy as np
import pandas as pd
import multiprocessing

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem


"""
SMILE to ECFP vector
"""
def generate_ecfp(smiles, radius=2, bits=2048):
    molecule = Chem.MolFromSmiles(smiles)

    return np.packbits(np.array(list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))))


def test():
    s = 'CCO'
    
    ecfp = generate_ecfp(s)
    print(ecfp)


if __name__ == '__main__':
    smiles = pd.read_parquet('/data/datasets/leash-BELKA/origin/test.parquet')['molecule_smiles']
    num_cpu = os.cpu_count()
    with multiprocessing.Pool(processes=num_cpu) as pool:
        ecfp = list(tqdm(pool.imap(generate_ecfp, smiles), total=len(smiles)))

    ecfp = np.stack(ecfp)
    np.save(f'./leash-BELKA/origin/test_ecfp.npz', ecfp)