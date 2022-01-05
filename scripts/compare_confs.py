from pathlib import PosixPath
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import random
import argparse

def calc_performance_stats(true_confs, model_confs):
    
    threshold = np.arange(0, 2.5, .125)
    rmsd_list = []
    for tc in true_confs:
        for mc in model_confs:

            try:
                rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            except RuntimeError:
                return None
            rmsd_list.append(rmsd_val)

    rmsd_array = np.array(rmsd_list).reshape(len(true_confs), len(model_confs)) # reshape 1-D list into 2-D ndarray

    coverage_recall = np.sum(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0) / len(true_confs)
    amr_recall = rmsd_array.min(axis=1).mean()

    coverage_precision = np.sum(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1) / len(model_confs)
    amr_precision = rmsd_array.min(axis=0).mean()
    
    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpath", type=PosixPath, help="absolute path of test dir")
    parser.add_argument("--dataset", type=str, default="drugs", help="[drugs,qm9]")
    parser.add_argument("--ref", type=str, default="test_ref.pickle", help="name of ref pickle")
    parser.add_argument("--test", type=str, default="test_GeoMol.pickle", help="name of GeoMol pickle")
    parser.add_argument("--stats", type=str, default="stats.npy", help="name of output stats summary npy file")

    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    
    exp_dir = args.testpath
    suffix = ''
    with open(exp_dir / args.test, 'rb') as f:
        model_preds = pickle.load(f)
    
    dataset = args.dataset
    # test_data = pd.read_csv(exp_dir / 'test_smiles.csv')  # this should include the corrected smiles
    test_data = pd.read_csv(exp_dir / 'test_smiles_corrected.csv')  # this should include the corrected smiles

    with open(exp_dir / args.ref, 'rb') as f:
        true_mols = pickle.load(f)

    coverage_recall, amr_recall, coverage_precision, amr_precision = [], [], [], []
    test_smiles = []
    rdkit_smiles = test_data.smiles.values
    corrected_smiles = test_data.corrected_smiles.values
    model_smiles = [smi for smi in list(model_preds.keys())]
    threshold_ranges = np.arange(0, 2.5, .125)
    
    for smi, model_smi, corrected_smi in tqdm(zip(rdkit_smiles, model_smiles, corrected_smiles), total=len(rdkit_smiles)):
        
        try:
            model_confs = model_preds[model_smi]
        except KeyError:
            print(f'no model prediction available: {model_smi}')
            coverage_recall.append(threshold_ranges*0)
            amr_recall.append(np.nan)
            coverage_precision.append(threshold_ranges*0)
            amr_precision.append(np.nan)
            test_smiles.append(smi)
            continue
    
        # failure if model can't generate confs                                                                                                                                                         
        if len(model_confs) == 0:
            print(f'model failed: {smi}')
            coverage_recall.append(threshold_ranges*0)
            amr_recall.append(np.nan)
            coverage_precision.append(threshold_ranges*0)
            amr_precision.append(np.nan)
            test_smiles.append(smi)
            continue
        
        try:
            true_confs = true_mols[smi]
        except KeyError:
            print(f'cannot find ground truth conformer file: {smi}')
            continue
        
        # remove reacted conformers
        true_confs = clean_confs(corrected_smi, true_confs)
        if len(true_confs) == 0:
            print(f'poor ground truth conformers: {corrected_smi}')
            continue
            
        stats = calc_performance_stats(true_confs, model_confs)
        if not stats:
            print(f'failure calculating stats: {smi, model_smi}')
            continue
            
        cr, mr, cp, mp = stats
        coverage_recall.append(cr)
        amr_recall.append(mr)
        coverage_precision.append(cp)
        amr_precision.append(mp)
        test_smiles.append(smi)
    
    np.save(exp_dir/ args.stats, [coverage_recall, amr_recall, coverage_precision, amr_precision, test_smiles])
    
    coverage_recall_vals = [stat[10] for stat in coverage_recall] # 10 means 1.25
    coverage_precision_vals = [stat[10] for stat in coverage_precision] 
    
    print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals)*100:.2f}, Median = {np.median(coverage_recall_vals)*100:.2f}')
    print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
    print()
    print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals)*100:.2f}, Median = {np.median(coverage_precision_vals)*100:.2f}')
    print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')
    