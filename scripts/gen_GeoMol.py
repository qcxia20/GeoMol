# %%
import os
import os.path as osp
from pathlib import Path, PosixPath
import argparse
from networkx.algorithms.coloring.greedy_coloring import STRATEGIES
import numpy as np
import pandas as pd

import torch
import rdkit
from rdkit.Chem import AllChem
from rdkit import  Geometry

import pickle
import random
import copy
import time

# from model.model import GeoMol
from model.featurization import construct_loader
from generate_confs import generate_GeoMol_confs
from clean_smiles import clean_confs, correct_smiles
# %%
def is_allzero(tensor):
    return False if torch.any(tensor != 0) else True

def set_rdmol_positions(mol, pose):
    for i in range(len(pose)):
        # mol.GetConformer(0).SetAtomPosition(i, pose[i].tolist())
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(float(pose[i,0]), float(pose[i,1]), float(pose[i,2])))
    return mol

def generate_conformers(mol, num_confs, smi):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=num_confs, 
        maxAttempts=0,
        ignoreSmoothingFailures=True,
    )
    if mol.GetNumConformers() != num_confs:
        print('Warning: Failure cases occured at %s, generated: %d , expected: %d.' % (smi, mol.GetNumConformers(), num_confs, ))
        return [mol, smi]
    
    return [mol]

def extract_smi_mols_ref(test_loader, args):
    smiles, K, mols, posss = [], [], [], []
    smi_mols_dict = {}
    smi_conf_dict = {}
    testset = test_loader.dataset[:args.n_testset] if args.n_testset else test_loader.dataset
    for moldata in testset:
        mol = []
        molss = []
        smiles.append(moldata['name'])
        refnum = [is_allzero(moldata['pos'][0][i]) for i in range(args.n_true_confs)].count(False)
        K.append(refnum)
        mols.append(moldata['mol'])
        work_poss = [moldata['pos'][0][i] for i in range(args.n_true_confs) if not is_allzero(moldata['pos'][0][i])]
        posss.append(work_poss)
    
        for i, pose in enumerate(work_poss):
            mol = copy.deepcopy(moldata['mol']) # otherwise each mol will be changed to the last conf
            molss.append(set_rdmol_positions(mol, pose))
        
        smi_mols_dict[moldata['name']] = molss
    
    # To support the same as smi_mols_dict (<1000)
    smi_conf_dict = {
        'smiles': list(smi_mols_dict.keys()),
        'n_conformers': [len(i) for i in smi_mols_dict.values()]
    }
    """
    # save through data in GEOM dataset with redundant canonical SMILES
    smi_conf_dict = {
        'smiles': smiles,
        'n_conformers': [ i for i in K ] # 2K will be announced by generate_confs.py
        }
    """
    return smi_mols_dict, smi_conf_dict

def generate_rdkit_mols(smi_mols_dict):
    err_smis = []
    data_ref = smi_mols_dict
    smiles, confs = list(data_ref.keys()), list(data_ref.values())

    smi_rdkitmols_dict = {}
    for i in range(len(data_ref)):
        molss = []
        return_data = copy.deepcopy(confs[i])
        num_confs = len(return_data) * 2

        start_mol = return_data[0]
        mol_s = generate_conformers(start_mol, num_confs=num_confs, smi=smiles[i]) # The mol contains num_confs of confs
        if len(mol_s) != 1: # Error in generation
            mol, err_smi = mol_s
            err_smis.append(err_smi)
        else:
            mol = mol_s[0]

        num_pos_gen = mol.GetNumConformers()
        all_pos = []

        if num_pos_gen == 0:
            continue
        # assert num_confs == num_pos_gen
        for j in range(num_pos_gen):
            pose = mol.GetConformer(j).GetPositions()
            mol_new = copy.deepcopy(confs[i][0])
            molss.append(set_rdmol_positions(mol_new, pose))

        smi_rdkitmols_dict[smiles[i]] = molss

    return smi_rdkitmols_dict, err_smis

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=PosixPath, help="name of dataset")
    # parser.add_argument("--split_path", type=PosixPath, help="name of dataset")
    parser.add_argument("--split", type=str, default="split0", help="split No.")
    parser.add_argument("--dataset", type=str, default="drugs", help="[drugs,qm9]")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--n_true_confs", type=int, default=10)
    parser.add_argument("--n_testset", type=int, help="number of mols in test set for evaluation, use all if not announced")
    parser.add_argument("--rdkit", action="store_true", default = False, help="whether to generate rdkit mols at the same time")
    parser.add_argument("--geomol", action="store_true", default = False, help="whether to generate GeoMol mols at the same time")
    parser.add_argument('--trained_model_dir', type=str, help="absolute path of trained model used for prediction")
    parser.add_argument('--mmff', action='store_true', default=False)
    parser.add_argument('--datatype', type=str, default='test', help="['train', 'val','test']")
    parser.add_argument('--cutoff', type=float, default=0.7, help="simply filter molecules with clash")
    args = parser.parse_args()

    datetime = '-'.join(list(map(str, list(time.localtime())[1:5])))
    rootpath = Path(os.environ['HOME']) / "git-repo/AI-CONF/"
    split_path = rootpath / f"GeoMol/data/{args.dataset.upper()}/splits/{args.split}.npy"
    data_dir = rootpath / f"datasets/GeoMol/data/{args.dataset.upper()}/{args.dataset}"
    test_dir = rootpath / f"datasets/GeoMol/test/{args.dataset}-{args.split}/{datetime}"
    if not test_dir.exists():
        os.system(f"mkdir -p {test_dir}")

    # seed = 2021
    seed = 2022
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    n_true_confs = 20 if args.dataset=="drugs" else 10
    setattr(args, 'n_true_confs', n_true_confs)
    setattr(args, 'data_dir', data_dir)
    setattr(args, 'split_path', split_path)
    test_loader = construct_loader(args, modes=(args.datatype))
    
    # smis = [test_loader.dataset[i].name for i in range(1000)]
    # print(smis)
    # print(len(smis))
    # print(len(set(smis))) # Through simple test, we've found that 1000->978 comes from redundancy, which should be issue with canonical smiles

    smi_mols_dict, smi_conf_dict = extract_smi_mols_ref(test_loader, args)

    ###################### clean smi #####################
    test_data = pd.DataFrame(smi_conf_dict)
    true_confs_dir = data_dir
    corrected_smiles_dict = {}
    corrected_smi_mols_dict = {}
    failed_smiles = []
    len_true_confs = []
    for i, data in test_data.iterrows():
        smi = data.smiles
        try:
            with open(osp.join(true_confs_dir, smi.replace('/', '_') + '.pickle'), "rb") as f: # since that /s are transformed into _ in GEOM dataset
                mol_dic = pickle.load(f)
        except FileNotFoundError:
            print(f'cannot find ground truth conformer file: {smi}')
            continue
    
        true_confs = [conf['rd_mol'] for conf in mol_dic['conformers']] # list of rdmol in pkl file
        try:
            true_confs = clean_confs(smi, true_confs) # list of rdmol of which has same smi as the one in pkl file
        except rdkit.Chem.rdchem.KekulizeException:
            print('kekulize problem', smi)
            true_confs = []
        len_true_conf = len(true_confs) if len(true_confs) < args.n_true_confs else args.n_true_confs # keep < limited number
        len_true_confs.append(len_true_conf)
        if len(true_confs) == 0:
            corrected_smiles_dict[smi] = smi
            continue
    
        corrected_smi = correct_smiles(true_confs)
        corrected_smiles_dict[smi] = corrected_smi
    
        if corrected_smi is None:
            failed_smiles.append(smi)
            corrected_smiles_dict[smi] = smi
            print(f'failed: {smi}\n')
        # print(smi, len(corrected_smiles_dict.values()))
        corrected_smi_mols_dict[corrected_smi] = true_confs if len_true_conf < args.n_true_confs else true_confs[:args.n_true_confs]

    test_data['len_true_confs'] = len_true_confs
    test_data['corrected_smiles'] = list(corrected_smiles_dict.values()) # add one column of corrected smiles
    corrected_smi_conf_dict = pd.DataFrame({'smiles': test_data.corrected_smiles.values, 'n_conformers': test_data.len_true_confs.values})
    ######################################################
    ### dump ref mols, rdkit mols, and test_smiles.csv ###
    with open(test_dir / "test_ref.pickle", 'wb') as f:
        pickle.dump(smi_mols_dict, f)
    with open(test_dir / "test_ref_cleaned.pickle", 'wb') as f:
        pickle.dump(corrected_smi_mols_dict, f)

    df = pd.DataFrame(smi_conf_dict)
    df.to_csv(test_dir / "test_smiles.csv", index=False)
    test_data.to_csv(test_dir / 'test_smiles_corrected.csv', index=False)

    # """
    ################## RDKit generation ##################
    if args.rdkit:
        smi_rdkitmols_dict, err_smis = generate_rdkit_mols(smi_mols_dict)
        smi_rdkitmols_dict, err_smis = generate_rdkit_mols(smi_mols_dict)
        with open(test_dir / "test_rdkit.pickle", "wb") as fout:
            pickle.dump(smi_rdkitmols_dict, fout)
        with open(test_dir / "rdkit_err_smiles.txt", 'w') as f:
            f.write('\n'.join(err_smis))
    ######################################################
        # Cleaned
        smi_rdkitmols_dict, err_smis = generate_rdkit_mols(corrected_smi_mols_dict)
        smi_rdkitmols_dict, err_smis = generate_rdkit_mols(corrected_smi_mols_dict)
        with open(test_dir / "test_rdkit_cleaned.pickle", "wb") as fout:
            pickle.dump(smi_rdkitmols_dict, fout)
        with open(test_dir / "rdkit_err_smiles_cleaned.txt", 'w') as f:
            f.write('\n'.join(err_smis))
            
    ################# GeoMol generation ##################
    # """
    # """
    if args.geomol:
        testdata = pd.DataFrame(smi_conf_dict)
        conformer_dict, conformer_dict_cutoff, err_smis, err_smis_cutoff = generate_GeoMol_confs(args, testdata)
        with open(test_dir / "test_GeoMol.pickle", "wb") as fout:
            pickle.dump(conformer_dict, fout)
        with open(test_dir / f"test_GeoMol_c{args.cutoff}.pickle", "wb") as fout:
            pickle.dump(conformer_dict_cutoff, fout)        
        with open(test_dir / "GeoMol_err_smiles.txt", 'w') as f:
            f.write('\n'.join(err_smis))
        with open(test_dir / f"GeoMol_err_smiles_c{args.cutoff}.txt", 'w') as f:
            f.write('\n'.join(err_smis_cutoff))
    ######################################################
        testdata = pd.DataFrame(corrected_smi_conf_dict)
        conformer_dict, conformer_dict_cutoff, err_smis, err_smis_cutoff = generate_GeoMol_confs(args, testdata)
        with open(test_dir / "test_GeoMol_cleaned.pickle", "wb") as fout:
            pickle.dump(conformer_dict, fout)
        with open(test_dir / f"test_GeoMol_c{args.cutoff}_cleaned.pickle", "wb") as fout:
            pickle.dump(conformer_dict_cutoff, fout)        
        with open(test_dir / "GeoMol_err_smiles_cleaned.txt", 'w') as f:
            f.write('\n'.join(err_smis))
        with open(test_dir / f"GeoMol_err_smiles_c{args.cutoff}_cleaned.txt", 'w') as f:
            f.write('\n'.join(err_smis_cutoff))

    # """


    
