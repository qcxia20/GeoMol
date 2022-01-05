from rdkit import Chem

import os
import os.path as osp
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import PosixPath, Path
from tqdm import tqdm

# location of unzipped GEOM dataset


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False) # remove stereo in smi
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False) # remove stereo in confs
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids] # Only confs with same smi will be considered as clean_confs. The necessity is to filter out conformers that reacted during metadynamics simulation (i.e. not correspond to the initial SMIELS)


def correct_smiles(true_confs):

    conf_smis = []
    for c in true_confs:
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c))
        conf_smis.append(conf_smi)

    try:
        common_smi = mode(conf_smis) # mode is to find and return the most common one in a list
    except StatisticsError:
        print('StatisticsError', mode(conf_smis))
        return None  # these should be cleaned by hand

    if np.sum([common_smi == smi for smi in conf_smis]) == len(conf_smis):
        return mode(conf_smis)
    else:
        print('consensus', common_smi)  # these should probably also be investigated manually
        return common_smi
        # return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=PosixPath, help="path of test_smiles.csv, which is also the path for output test_smiles_corrected.csv")
    parser.add_argument("--dataset", type=str, default="drugs", help="datatype, [drugs, qm9]")
    args = parser.parse_args()

    # true_confs_dir = '/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/data/DRUGS/drugs/'
    true_confs_dir = Path(os.environ['HOME']) / 'git-repo/AI-CONF/datasets/GeoMol/data' / args.dataset.upper() / args.dataset
    # test_data = pd.read_csv('/pubhome/qcxia02/git-repo/AI-CONF/datasets/GeoMol/test/drugs-split0/1-3-16-8/test_smiles.csv')
    test_data = pd.read_csv(args.path / 'test_smiles.csv')

    corrected_smiles_dict = {}
    failed_smiles = []
    for i, data in test_data.iterrows():
    
        smi = data.smiles
    
        try:
            with open(osp.join(true_confs_dir, smi.replace('/', '_') + '.pickle'), "rb") as f: # since that /s are transformed into _ in GEOM dataset
                mol_dic = pickle.load(f)
        except FileNotFoundError:
            print(f'cannot find ground truth conformer file: {smi}')
            continue
    
        true_confs = [conf['rd_mol'] for conf in mol_dic['conformers']] # list of rdmol in pkl file
        true_confs = clean_confs(smi, true_confs) # list of rdmol of which has same smi as the one in pkl file
        if len(true_confs) == 0:
            corrected_smiles_dict[smi] = smi
            continue
    
        corrected_smi = correct_smiles(true_confs)
        corrected_smiles_dict[smi] = corrected_smi
    
        if corrected_smi is None:
            failed_smiles.append(smi)
            corrected_smiles_dict[smi] = smi
            print(f'failed: {smi}\n')
        print(smi, len(corrected_smiles_dict.values()))
    
    test_data['corrected_smiles'] = list(corrected_smiles_dict.values()) # add one column of corrected smiles
    test_data.to_csv('./test_smiles_corrected.csv', index=False)
    