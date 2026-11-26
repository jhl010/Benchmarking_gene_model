#!/usr/bin/env python3
"""Yang-style Ring_systems benchmark analysis for TargetDiff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import useful_rdkit_utils as uru
from useful_rdkit_utils.ring_systems import RingSystemLookup, get_min_ring_frequency

RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = 'TargetDiff'
OUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(REPO))
from Filtering_functions import filter_molecules

STANDARDIZER = Standardizer()
RING_FINDER = uru.RingSystemFinder()
RING_LOOKUPS = {
    'ChEMBL': RingSystemLookup(),
    'ZINC20': RingSystemLookup(ring_system_csv=str(REPO / 'data' / 'ZINC20_rings_ststem.csv')),
    'ZINC22': RingSystemLookup(ring_system_csv=str(REPO / 'data' / 'ZINC22_rings_system.csv')),
}
OTHER_FILTERS = ['Glaxo', 'Dundee', 'BMS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT']

def input_file() -> Path:
    candidates = sorted(SCRIPT_DIR.glob('combined_SMILES*generated.txt'))
    if not candidates:
        candidates = sorted(SCRIPT_DIR.glob('combined_smiles*generated.txt'))
    if not candidates:
        raise FileNotFoundError('No combined_SMILES*generated.txt file found in ' + str(SCRIPT_DIR))
    return candidates[0]


def canonical_smiles(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ''


def standardize_mol(mol: Chem.Mol):
    mol = STANDARDIZER.standardize(mol)
    mol = STANDARDIZER.fragment_parent(mol, skip_standardize=True)
    mol = STANDARDIZER.tautomer_parent(mol, skip_standardize=True)
    mol = STANDARDIZER.isotope_parent(mol, skip_standardize=True)
    mol = STANDARDIZER.charge_parent(mol, skip_standardize=True)
    return mol


def passes_ro5(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return bool(
        Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
        and Descriptors.MolWt(mol) <= 500
        and Descriptors.MolLogP(mol) <= 5
    )

def process_input(path: Path):
    rows = []
    invalid = 0
    with path.open(errors='replace') as f:
        for sample_id, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            mol = Chem.MolFromSmiles(parts[0])
            if mol is None:
                invalid += 1
                continue
            try:
                mol = standardize_mol(mol)
                mol = uru.get_largest_fragment(mol)
                smi = canonical_smiles(mol)
                inchi = Chem.MolToInchi(mol)
            except Exception:
                invalid += 1
                continue
            rows.append({
                'sample_id': sample_id,
                'raw_path': str(path),
                'smiles': smi,
                'inchi': inchi,
                'mol': mol,
            })
    return pd.DataFrame(rows), invalid

def ring_pass_summary(df: pd.DataFrame, label: str) -> dict:
    eligible = df[df['min_freq'] != -1]
    passed = int((df['min_freq'] > 100).sum())
    total = int(len(eligible))
    return {
        label + '_ring_pass_gt100': passed,
        label + '_ring_eligible': total,
        label + '_ring_pass_fraction': passed / total if total else np.nan,
    }


def apply_ring_lookup(df: pd.DataFrame, label: str) -> None:
    lookup = RING_LOOKUPS[label]
    result = df['mol'].apply(lookup.process_mol)
    df[['min_ring', 'min_freq']] = result.apply(uru.get_min_ring_frequency).tolist()

def main() -> None:
    inp = input_file()
    print('MODEL:', MODEL)
    print('INPUT:', inp)
    n_input = sum(1 for _ in inp.open(errors='replace'))
    df, invalid = process_input(inp)
    if df.empty:
        raise SystemExit('No valid molecules to analyse')

    apply_ring_lookup(df, 'ChEMBL')
    chembl_summary = ring_pass_summary(df, 'ChEMBL')
    ring0 = df[df['min_freq'] == 0]
    ring100 = df[df['min_freq'] > 100]

    df = df.drop_duplicates('inchi', keep='first', ignore_index=True)
    df[['smiles']].to_csv(OUT_DIR / 'unique_standardized_molecules.smi', index=False, header=False)
    ring0[['smiles']].to_csv(OUT_DIR / 'ring_appearance_0.smi', index=False, header=False)
    ring100[['smiles']].to_csv(OUT_DIR / 'ring_appearance_100more.smi', index=False, header=False)

    apply_ring_lookup(df, 'ZINC22')
    zinc22_summary = ring_pass_summary(df, 'ZINC22')
    apply_ring_lookup(df, 'ZINC20')
    zinc20_summary = ring_pass_summary(df, 'ZINC20')

    reos = uru.REOS()
    reos.set_active_rule_sets(['PAINS'])
    pains_df = reos.pandas_mols(df['mol'])
    filter_summary = {
        'PAINS_pass': int((pains_df['description'] == 'ok').sum()),
        'PAINS_pass_fraction': float((pains_df['description'] == 'ok').mean()),
    }
    for rule in OTHER_FILTERS:
        reos.set_active_rule_sets([rule])
        rule_df = reos.pandas_mols(df['mol'])
        ok = int((rule_df['description'] == 'ok').sum())
        filter_summary[rule + '_pass'] = ok
        filter_summary[rule + '_pass_fraction'] = ok / len(rule_df) if len(rule_df) else np.nan

    df['Reymond'] = df['mol'].apply(filter_molecules)
    df['Lipinski'] = df['smiles'].apply(passes_ro5)
    filter_summary['Reymond_pass'] = int(df['Reymond'].sum())
    filter_summary['Reymond_pass_fraction'] = float(df['Reymond'].mean())
    filter_summary['Lipinski_pass'] = int(df['Lipinski'].sum())
    filter_summary['Lipinski_pass_fraction'] = float(df['Lipinski'].mean())

    summary = {
        'model': MODEL,
        'input_lines': n_input,
        'invalid_or_failed': invalid,
        'valid_standardized': len(df),
        'unique_inchi': len(df),
    }
    summary.update(chembl_summary)
    summary.update(zinc22_summary)
    summary.update(zinc20_summary)
    summary.update(filter_summary)
    summary['has_ring_system'] = int(chembl_summary['ChEMBL_ring_eligible'])
    summary['has_ring_system_fraction'] = summary['has_ring_system'] / len(df) if len(df) else np.nan

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / 'ring_systems_summary.csv', index=False)
    print(summary_df.T.to_string())


if __name__ == '__main__':
    main()
