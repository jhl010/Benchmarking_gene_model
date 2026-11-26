#!/usr/bin/env python3
"""Yang-style Ring_systems analysis for FDA-approved small-molecule drugs (control dataset)."""
from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi
import useful_rdkit_utils as uru
from useful_rdkit_utils.ring_systems import RingSystemLookup

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).resolve().parent
REF_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REF_REPO))
from Filtering_functions import filter_molecules

STANDARDIZER = Standardizer()
OTHER_FILTERS = ['Glaxo', 'Dundee', 'BMS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT']


def standardize_mol(mol):
    try:
        mol = STANDARDIZER.standardize(mol)
        mol = STANDARDIZER.fragment_parent(mol, skip_standardize=True)
        mol = STANDARDIZER.tautomer_parent(mol, skip_standardize=True)
        mol = STANDARDIZER.isotope_parent(mol, skip_standardize=True)
        mol = STANDARDIZER.charge_parent(mol, skip_standardize=True)
        return mol
    except Exception:
        return None


def passes_rule_of_five(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return bool(
        Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
        and Descriptors.MolWt(mol) <= 500
        and Descriptors.MolLogP(mol) <= 5
    )


def main():
    raw_file = SCRIPT_DIR / 'drugbank_smdrugs_inchi.txt'
    standardized_smiles_list = []
    for line in raw_file.read_text().splitlines():
        inch = line.strip()
        if not inch:
            continue
        mol = inchi.MolFromInchi(inch)
        std = standardize_mol(mol)
        if std is not None:
            standardized_smiles_list.append(Chem.MolToSmiles(std))

    (SCRIPT_DIR / 'drugbank_smdrugs_smiles.txt').write_text('\n'.join(standardized_smiles_list) + '\n')
    print('standardized_smiles:', len(standardized_smiles_list))

    df = pd.DataFrame(standardized_smiles_list, columns=['SMILES'])
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['mol'] = df['mol'].apply(uru.get_largest_fragment)
    df = df[df['mol'].notna()].reset_index(drop=True)

    ring_system_finder = uru.RingSystemFinder()
    df['ring_sytems'] = df['mol'].apply(ring_system_finder.find_ring_systems)
    ring_system_list = list(itertools.chain.from_iterable(df['ring_sytems'].values))
    ring_count_df = pd.DataFrame(Counter(ring_system_list).items(), columns=['SMILES', 'Count'])
    ring_count_df.sort_values('Count', ascending=False, inplace=True)

    ring_system_lookup = RingSystemLookup()
    res = df['mol'].apply(ring_system_lookup.process_mol)
    df[['min_ring', 'min_freq']] = res.apply(uru.get_min_ring_frequency).tolist()

    df['inchi'] = df['mol'].apply(Chem.MolToInchi)
    df = df.drop_duplicates('inchi', keep='first', ignore_index=True)
    print('unique_inchi:', len(df))

    df_ring = df[df['min_freq'] != -1]
    filtered_df_ring_freq = df[df['min_freq'] > 100]
    print('ChEMBL_ring_eligible:', len(df_ring))
    print('ChEMBL_ring_pass_gt100:', len(filtered_df_ring_freq))

    reos = uru.REOS()
    reos.set_active_rule_sets(['PAINS'])
    pains = reos.pandas_mols(df['mol'])
    df_PAINS_filter = pd.concat([df, pains], axis=1)
    df_PAINS_filter = df_PAINS_filter[df_PAINS_filter['description'] == 'ok']
    print('PAINS_pass:', len(df_PAINS_filter))

    for rule in OTHER_FILTERS:
        reos.set_active_rule_sets([rule])
        new_df = reos.pandas_mols(df['mol'])
        print(rule, 'pass:', len(new_df[new_df['description'] == 'ok']), 'fraction:', len(new_df[new_df['description'] == 'ok']) / len(new_df))

    df['Reymond'] = df['mol'].apply(filter_molecules)
    df_Reymond_filter = df[df['Reymond'] == True]
    df['Lipinski'] = df['SMILES'].apply(passes_rule_of_five)
    df_Lipinski_filter = df[df['Lipinski'] == True]
    print('Reymond_pass:', len(df_Reymond_filter))
    print('Lipinski_pass:', len(df_Lipinski_filter))

    ring_system_lookup_zinc22 = RingSystemLookup(ring_system_csv=str(REF_REPO / 'data' / 'ZINC22_rings_system.csv'))
    res_zinc22 = df['mol'].apply(ring_system_lookup_zinc22.process_mol)
    df[['min_ring', 'min_freq']] = res_zinc22.apply(uru.get_min_ring_frequency).tolist()
    filtered_df_ring_freq = df[(df['min_freq'] > 100) & (df['min_freq'] != -1)]
    print('ZINC22_ring_pass_gt100:', len(filtered_df_ring_freq))
    filtered_df_ring_freq = df[(df['min_freq'] > 0) & (df['min_freq'] != -1)]
    filtered_df_ring_freq = df[(df['min_freq'] > 10) & (df['min_freq'] != -1)]

    ring_system_lookup_zinc20 = RingSystemLookup(ring_system_csv=str(REF_REPO / 'data' / 'ZINC20_rings_ststem.csv'))
    res_zinc20 = df['mol'].apply(ring_system_lookup_zinc20.process_mol)
    df[['min_ring', 'min_freq']] = res_zinc20.apply(uru.get_min_ring_frequency).tolist()
    filtered_df_ring_freq = df[(df['min_freq'] > 100) & (df['min_freq'] != -1)]
    print('ZINC20_ring_pass_gt100:', len(filtered_df_ring_freq))


if __name__ == '__main__':
    main()
