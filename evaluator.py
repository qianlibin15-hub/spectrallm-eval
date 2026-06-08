import json
import re
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.metrics.pairwise import cosine_similarity
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem.MACCSkeys import GenMACCSKeys

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048) # ECFP4 fingerprint,向外扩展2个原子，编码维度2048

def mol_from_smiles(smiles):

    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

functional_groups = {
    "alcohol": "[OX2H]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "ketone": "[CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "carboxylic_acid": "C(=O)[OH]",
    "ester": "C(=O)O[#6]",
    "ether": "[OD2]([#6])[#6]",
    "amide": "C(=O)N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "benzene": "c1ccccc1",
    "halogen": "[F,Cl,Br,I]",
}

functional_group_patterns = {

    name: Chem.MolFromSmarts(smarts)
    for name, smarts in functional_groups.items()
}

def get_fg_set(mol):   # 功能组

    fg_set = set()   # 功能组集合

    for name, patt in functional_group_patterns.items():  # 遍历功能组

        if patt is not None and mol.HasSubstructMatch(patt):  # 匹配
            fg_set.add(name)

    return fg_set

   
def mces_similarity_from_mols(mol1, mol2, timeout=5):

    if mol1 is None or mol2 is None:
        return 0.0

    try:
        mcs = rdFMCS.FindMCS(                               # MCS搜索
            [mol1, mol2],
            timeout=timeout,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            ringMatchesRingOnly=True,
            completeRingsOnly=False
        )

        if mcs.canceled:
            # 说明timeout导致搜索未完成，结果可能偏小
            # 这里仍返回已有结果，但你要知道它是近似值
            pass

        common_bonds = mcs.numBonds
        max_bonds = max(mol1.GetNumBonds(),mol2.GetNumBonds())

        if max_bonds == 0:
            return 0.0
 
        # return common_bonds / max_bonds
        return max_bonds - common_bonds

    except Exception:
        return 0.0
    
def exact_match_from_mols(mol1, mol2):
    if mol1 is None or mol2 is None:
        return False

    s1 = Chem.MolToSmiles(mol1, canonical=True)
    s2 = Chem.MolToSmiles(mol2, canonical=True)

    return s1 == s2

def tanimoto_ecfp4_from_mols(mol1, mol2):
    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = morgan_generator.GetFingerprint(mol1)    # ECFP4指纹
    fp2 = morgan_generator.GetFingerprint(mol2)

    return TanimotoSimilarity(fp1, fp2)

def tanimoto_maccs_from_mols(mol1, mol2):   

    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = GenMACCSKeys(mol1)      # MACCS指纹
    fp2 = GenMACCSKeys(mol2)

    return TanimotoSimilarity(fp1, fp2)

def cosine_fp_from_mols(mol1, mol2):
    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = morgan_generator.GetFingerprint(mol1)
    fp2 = morgan_generator.GetFingerprint(mol2)

    arr1 = np.array(list(fp1))
    arr2 = np.array(list(fp2))

    return cosine_similarity( arr1.reshape(1, -1), arr2.reshape(1, -1))[0][0]  # 余弦相似度

def fg_similarity_from_mols(mol1, mol2):   # 基于功能基团集合的 Jaccard 相似度
    if mol1 is None or mol2 is None:
        return 0.0

    g1 = get_fg_set(mol1)
    g2 = get_fg_set(mol2)

    union = len(g1 | g2)  # 并集大小

    if union == 0:
        return 1.0

    return len(g1 & g2) / union

def fraggle_similarity_from_mols(mol1, mol2):   # Fraggle相似度
    if mol1 is None or mol2 is None:
        return 0.0

    try:
        result = FraggleSim.GetFraggleSimilarity(mol1, mol2)

        if isinstance(result, tuple):
            return float(result[0])

        return float(result)

    except Exception:
        return 0.0
    

def evaluate(data, return_detail=False):

    records = []
    errors = []
    invalid_cases = []
    fraggle_scores = []

    validity_flags = []

    exact = []
    tanimoto_scores = []
    maccs_scores = []
    cosine_scores = []
    fg_scores = []
    mces_scores = []

    approx_match = []
    acceptable_match = []

    for i, sample in enumerate(data):

        pred = sample["pred"]
        label = sample["label"]

        mol_pred = mol_from_smiles(pred)
        mol_label = mol_from_smiles(label)

        valid = mol_pred is not None
        validity_flags.append(valid)

        if not valid:
            errors.append({
                "index": i,
                "pred": pred,
                "label": label,
                "error_type": "invalid_smiles_pred"
            })
            invalid_cases.append(sample)
            continue

        t = tanimoto_ecfp4_from_mols(mol_pred, mol_label)
        mac = tanimoto_maccs_from_mols(mol_pred, mol_label)
        cos = cosine_fp_from_mols(mol_pred, mol_label)
        fg = fg_similarity_from_mols(mol_pred, mol_label)
        mces = mces_similarity_from_mols(mol_pred, mol_label, timeout=1)
        em = exact_match_from_mols(mol_pred, mol_label)
        fraggle = fraggle_similarity_from_mols(mol_pred,mol_label)

        tanimoto_scores.append(t)
        maccs_scores.append(mac)
        cosine_scores.append(cos)
        fg_scores.append(fg)
        mces_scores.append(mces)
        exact.append(em)

        approx_match.append(t >= 0.675)
        acceptable_match.append(t >= 0.4)
        fraggle_scores.append(fraggle)

        records.append({
            "index": i,
            "pred": pred,
            "label": label,
            "tanimoto": t,
            "maccs": mac,
            "cosine": cos,
            "fg": fg,
            "mces": mces,
            "exact": em,
            "fraggle": fraggle
        })

    def safe_mean(x):
        return np.mean(x) if len(x) > 0 else 0.0

    summary_df = {
        "validity": 100 * np.mean(validity_flags),
        "exact_match": 100 * safe_mean(exact),
        "approx_match": 100 * safe_mean(approx_match),
        "acceptable_match": 100 * safe_mean(acceptable_match),
        "tanimoto": safe_mean(tanimoto_scores),
        "tanimoto_maccs": safe_mean(maccs_scores),
        "cosine": safe_mean(cosine_scores),
        "functional_group": safe_mean(fg_scores),
        "mces": safe_mean(mces_scores),
        "valid_samples": len(records),
        "invalid_samples": len(invalid_cases),
        "fraggle": safe_mean(fraggle_scores),
    }

    df_records = pd.DataFrame(records)
    df_summary = pd.DataFrame([summary_df])
    detail = {
        "records": records,
        "invalid_cases": invalid_cases,
        "errors": errors
    }

    if return_detail:
        return df_records, df_summary, detail

    return df_records, df_summary, detail
    