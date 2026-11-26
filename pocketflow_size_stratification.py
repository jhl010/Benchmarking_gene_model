#!/usr/bin/env python3
"""PocketFlow structural-prior metrics stratified by observed heavy-atom count.

This is an analysis unique to the local benchmark: it stratifies standardized
PocketFlow molecules into 15-20, 21-25, 26-30 and >30 heavy-atom groups and
recomputes, within each stratum, the ChEMBL ring-system metric (min_freq > 100)
and BM-scaffold coverage in the ZINC20/ZINC22 drug-like subsets.

The definitions follow the local PocketFlow sensitivity analysis
implemented in pocketflow_max_atom_sensitivity.py and are specific to this
local benchmark.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.inchi import MolToInchiKey
from statsmodels.stats.proportion import proportion_confint
from useful_rdkit_utils.ring_systems import RingSystemLookup, get_min_ring_frequency

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA = SCRIPT_DIR / 'data'
STRATA = ('15-20', '21-25', '26-30', '>30')

METRIC_LABELS = {
    'chembl_ring_min_freq_gt100': 'ChEMBL ring-system metric (min_freq > 100)',
    'zinc20_bm_coverage': 'BM scaffold coverage in ZINC20',
    'zinc22_bm_coverage': 'BM scaffold coverage in ZINC22',
}


def heavy_atom_stratum(atom_count: int) -> str | None:
    if atom_count < 15:
        return None
    if atom_count <= 20:
        return '15-20'
    if atom_count <= 25:
        return '21-25'
    if atom_count <= 30:
        return '26-30'
    return '>30'


def standardize(mol: Chem.Mol) -> Chem.Mol:
    mol = Standardizer().standardize(mol)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def bm_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)


def resolve_reference(name: str, override: str | None) -> Path:
    if override:
        return Path(override)
    local = LOCAL_DATA / f'standardized_BM_scaffolds_{name}.smi'
    if not local.exists():
        raise FileNotFoundError(
            f'Missing {name} BM scaffold reference: {local}. '
            'Download standardized_BM_scaffolds_ZINC20.smi and '
            'standardized_BM_scaffolds_ZINC22.smi from Hugging Face into the data folder.'
        )
    return local


def reference_hits(query_file: Path, reference_file: Path, output_file: Path) -> set[str]:
    with output_file.open('w') as output:
        completed = subprocess.run(
            ['grep', '-F', '-x', '-f', str(query_file), str(reference_file)],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr[-1000:])
    return {line.split()[0] for line in output_file.read_text(errors='replace').splitlines() if line.strip()}


def wilson(numerator: int, denominator: int) -> tuple[float, float, float]:
    if denominator == 0:
        return np.nan, np.nan, np.nan
    fraction = numerator / denominator
    lower, upper = proportion_confint(numerator, denominator, alpha=0.05, method='wilson')
    return fraction, lower, upper


def load_molecules(input_file: Path) -> pd.DataFrame:
    ring_lookup = RingSystemLookup()
    rows = []
    for line in input_file.read_text(errors='replace').splitlines():
        text = line.strip()
        if not text:
            continue
        smiles = text.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            mol = standardize(mol)
        except Exception:
            continue
        heavy_atoms = int(mol.GetNumAtoms())
        stratum = heavy_atom_stratum(heavy_atoms)
        if stratum is None:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        ring_systems = ring_lookup.process_smiles(canonical)
        _, minimum_frequency = get_min_ring_frequency(ring_systems)
        has_ring_system = minimum_frequency >= 0
        rows.append({
            'smiles': canonical,
            'inchikey': MolToInchiKey(mol),
            'heavy_atoms': heavy_atoms,
            'heavy_atom_stratum': stratum,
            'has_ring_system': has_ring_system,
            'chembl_ring_min_freq': minimum_frequency if has_ring_system else np.nan,
            'chembl_ring_min_freq_gt100': bool(has_ring_system and minimum_frequency > 100),
            'bm_scaffold': bm_scaffold(canonical),
        })
    return pd.DataFrame(rows)


def stratum_table(molecules: pd.DataFrame, hits: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        frame = molecules[molecules['heavy_atom_stratum'].eq(stratum)]
        ring_eligible = frame[frame['has_ring_system']]
        scaffold_eligible = frame[frame['bm_scaffold'].notna()]
        definitions = [
            ('chembl_ring_min_freq_gt100',
             int(ring_eligible['chembl_ring_min_freq_gt100'].sum()), len(ring_eligible)),
            ('zinc20_bm_coverage',
             int(scaffold_eligible['bm_scaffold'].isin(hits['ZINC20']).sum()), len(scaffold_eligible)),
            ('zinc22_bm_coverage',
             int(scaffold_eligible['bm_scaffold'].isin(hits['ZINC22']).sum()), len(scaffold_eligible)),
        ]
        for metric, numerator, denominator in definitions:
            fraction, ci_low, ci_high = wilson(numerator, denominator)
            rows.append({
                'heavy_atom_stratum': stratum,
                'metric': metric,
                'metric_label': METRIC_LABELS[metric],
                'numerator': numerator,
                'denominator': denominator,
                'fraction': fraction,
                'percent': 100 * fraction if pd.notna(fraction) else np.nan,
                'ci_low_percent': 100 * ci_low if pd.notna(ci_low) else np.nan,
                'ci_high_percent': 100 * ci_high if pd.notna(ci_high) else np.nan,
                'standardized_molecules_in_stratum': len(frame),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='PocketFlow heavy-atom stratified structural-prior analysis')
    parser.add_argument('--input', default=str(SCRIPT_DIR / 'PocketFlow' / 'combined_SMILES_PocketFlow_generated.txt'))
    parser.add_argument('--output-dir', default=str(SCRIPT_DIR / 'analysis_output' / 'pocketflow_size_stratification'))
    parser.add_argument('--zinc20', default=None, help='Override ZINC20 BM scaffold reference .smi')
    parser.add_argument('--zinc22', default=None, help='Override ZINC22 BM scaffold reference .smi')
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    molecules = load_molecules(input_file)
    if molecules.empty:
        raise SystemExit('No standardizable molecules in requested heavy-atom strata')

    query_scaffolds = sorted(set(molecules['bm_scaffold'].dropna()))
    query_file = output_dir / 'unique_bm_scaffold_queries.smi'
    query_file.write_text('\n'.join(query_scaffolds) + ('\n' if query_scaffolds else ''), encoding='utf-8')

    hits = {}
    for name, override in (('ZINC20', args.zinc20), ('ZINC22', args.zinc22)):
        reference_file = resolve_reference(name, override)
        matched_file = output_dir / f'{name}_matched_bm_scaffolds.smi'
        hits[name] = reference_hits(query_file, reference_file, matched_file)
        print(f'{name}: matched {len(hits[name]):,}/{len(query_scaffolds):,} queried unique BM scaffolds')

    table = stratum_table(molecules, hits)
    table.to_csv(output_dir / 'pocketflow_size_stratification_table.csv', index=False)

    size_summary = molecules.groupby('heavy_atom_stratum').agg(
        standardized_molecules=('smiles', 'size'),
        unique_inchikeys=('inchikey', 'nunique'),
        mean_heavy_atoms=('heavy_atoms', 'mean'),
    ).reindex(STRATA)
    size_summary.to_csv(output_dir / 'pocketflow_size_stratum_counts.csv')

    print('\nPocketFlow heavy-atom stratified structural-prior metrics')
    print(table.to_string(index=False))
    print('\nOutput directory:', output_dir)


if __name__ == '__main__':
    main()
