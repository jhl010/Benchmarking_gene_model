#!/usr/bin/env python3
"""Compute PocketFlow max_atom_num sensitivity metrics from repository files.

This script is the local counterpart to the previously run
analyze_pocketflow_max_atom_sensitivity.py. It reads the five
PocketFlow_max_atom_num_* directories and recomputes overall and
heavy-atom-stratified structural-prior metrics:

- ChEMBL ring-system metric (min_freq > 100)
- BM scaffold coverage in ZINC20 drug-like subset
- BM scaffold coverage in ZINC22 drug-like subset

Proportions are reported with two-sided Wilson 95% confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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

REPO = Path(__file__).resolve().parent
LOCAL_DATA = REPO / 'data'
SETTINGS = (25, 30, 35, 40, 45)
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


def wilson(numerator: int, denominator: int):
    if denominator == 0:
        return np.nan, np.nan, np.nan
    fraction = numerator / denominator
    lower, upper = proportion_confint(numerator, denominator, alpha=0.05, method='wilson')
    return fraction, lower, upper


def input_file_for_setting(setting: int) -> Path:
    folder = REPO / f'PocketFlow_max_atom_num_{setting}'
    candidates = sorted(folder.glob('combined_SMILES*generated.txt'))
    if not candidates:
        raise FileNotFoundError(f'No generated SMILES file in {folder}')
    return candidates[0]


def load_molecules(setting: int, ring_lookup) -> pd.DataFrame:
    rows = []
    path = input_file_for_setting(setting)
    for line_number, raw_line in enumerate(path.read_text(errors='replace').splitlines()):
        text = raw_line.strip()
        if not text:
            continue
        mol = Chem.MolFromSmiles(text.split()[0])
        if mol is None:
            continue
        try:
            mol = standardize(mol)
        except Exception:
            continue
        heavy_atoms = int(mol.GetNumAtoms())
        stratum = heavy_atom_stratum(heavy_atoms)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        ring_systems = ring_lookup.process_smiles(canonical)
        _, minimum_frequency = get_min_ring_frequency(ring_systems)
        has_ring_system = minimum_frequency >= 0
        rows.append({
            'max_atom_num': setting,
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


def resolve_reference(name: str, override: str | None) -> Path:
    if override:
        return Path(override)
    path = LOCAL_DATA / f'standardized_BM_scaffolds_{name}.smi'
    if not path.exists():
        raise FileNotFoundError(f'Missing {name} BM scaffold reference: {path}')
    return path


def reference_hits(query_file: Path, reference_file: Path, output_file: Path) -> set:
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
    hits = set()
    for line in output_file.read_text(errors='replace').splitlines():
        if line.strip():
            hits.add(line.strip().split()[0])
    return hits


def metric_rows(subset: pd.DataFrame, setting: int, stratum: str, hits: dict) -> list[dict]:
    ring_eligible = subset[subset['has_ring_system']]
    scaffold_eligible = subset[subset['bm_scaffold'].notna()]
    definitions = [
        ('chembl_ring_min_freq_gt100',
         int(ring_eligible['chembl_ring_min_freq_gt100'].sum()), len(ring_eligible)),
        ('zinc20_bm_coverage',
         int(scaffold_eligible['bm_scaffold'].isin(hits['ZINC20']).sum()), len(scaffold_eligible)),
        ('zinc22_bm_coverage',
         int(scaffold_eligible['bm_scaffold'].isin(hits['ZINC22']).sum()), len(scaffold_eligible)),
    ]
    rows = []
    for metric, numerator, denominator in definitions:
        fraction, ci_low, ci_high = wilson(numerator, denominator)
        rows.append({
            'max_atom_num': setting,
            'heavy_atom_stratum': stratum,
            'metric': metric,
            'metric_label': METRIC_LABELS[metric],
            'numerator': numerator,
            'denominator': denominator,
            'fraction': fraction,
            'percent': 100 * fraction if pd.notna(fraction) else np.nan,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_low_percent': 100 * ci_low if pd.notna(ci_low) else np.nan,
            'ci_high_percent': 100 * ci_high if pd.notna(ci_high) else np.nan,
            'standardized_molecules_in_stratum': len(subset),
        })
    return rows


def size_summary(subset: pd.DataFrame, setting: int) -> list[dict]:
    rows = []
    for stratum in STRATA:
        frame = subset[subset['heavy_atom_stratum'].eq(stratum)]
        rows.append({
            'max_atom_num': setting,
            'heavy_atom_stratum': stratum,
            'standardized_molecules': len(frame),
            'unique_inchikeys': int(frame['inchikey'].nunique()) if len(frame) else 0,
            'mean_heavy_atoms': float(frame['heavy_atoms'].mean()) if len(frame) else np.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute PocketFlow max_atom_num structural-prior metrics.')
    parser.add_argument('--output-dir', default=str(REPO / 'analysis_output' / 'pocketflow_max_atom_sensitivity'))
    parser.add_argument('--zinc20', default=None, help='Override ZINC20 BM scaffold reference .smi')
    parser.add_argument('--zinc22', default=None, help='Override ZINC22 BM scaffold reference .smi')
    args = parser.parse_args()

    ring_lookup = RingSystemLookup()
    all_molecules = []
    for setting in SETTINGS:
        molecules = load_molecules(setting, ring_lookup)
        print(f'max_atom_num={setting}: {len(molecules)} standardized molecules', file=sys.stderr)
        all_molecules.append(molecules)
    molecules = pd.concat(all_molecules, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_scaffolds = sorted(set(molecules['bm_scaffold'].dropna()))
    query_file = output_dir / 'unique_bm_scaffold_queries.smi'
    query_file.write_text(chr(10).join(query_scaffolds) + chr(10), encoding='utf-8')

    hits = {}
    for name, override in (('ZINC20', args.zinc20), ('ZINC22', args.zinc22)):
        reference_file = resolve_reference(name, override)
        matched_file = output_dir / f'{name}_matched_bm_scaffolds.smi'
        hits[name] = reference_hits(query_file, reference_file, matched_file)
        print(f'{name}: matched {len(hits[name])}/{len(query_scaffolds)} unique BM scaffolds', file=sys.stderr)

    overall_rows, stratified_rows, size_rows = [], [], []
    for setting in SETTINGS:
        subset = molecules[molecules['max_atom_num'].eq(setting)]
        overall_rows.extend(metric_rows(subset, setting, 'overall', hits))
        size_rows.extend(size_summary(subset, setting))
        for stratum in STRATA:
            strata_subset = subset[subset['heavy_atom_stratum'].eq(stratum)]
            stratified_rows.extend(metric_rows(strata_subset, setting, stratum, hits))

    overall = pd.DataFrame(overall_rows)
    stratified = pd.DataFrame(stratified_rows)
    size = pd.DataFrame(size_rows)
    overall.to_csv(output_dir / 'overall_structural_prior_metrics_wilson95.csv', index=False)
    stratified.to_csv(output_dir / 'stratified_structural_prior_metrics_wilson95.csv', index=False)
    size.to_csv(output_dir / 'observed_size_distribution_summary.csv', index=False)

    manifest = {
        'settings': list(SETTINGS),
        'source_format': 'repository combined_SMILES_*_generated.txt files',
        'standardization': 'MolVS Standardizer followed by RDKit hydrogen removal and sanitization',
        'ring_metric': 'ChEMBL ring-system metric (min_freq > 100)',
        'bm_matching': 'exact canonical isomeric BM scaffold match against full ZINC20/ZINC22 references',
        'interval': 'two-sided Wilson 95% confidence interval',
        'p_values_reported': False,
    }
    (output_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')
    readme_lines = [
        '# PocketFlow max_atom_num sensitivity analysis',
        '',
        'Generated by pocketflow_max_atom_sensitivity.py.',
        '',
        '- overall_structural_prior_metrics_wilson95.csv: overall metrics for Figure 4.',
        '- stratified_structural_prior_metrics_wilson95.csv: metrics within observed heavy-atom strata.',
        '- observed_size_distribution_summary.csv: standardized molecule and InChIKey counts per stratum.',
        '',
    ]
    (output_dir / 'README.md').write_text(chr(10).join(readme_lines), encoding='utf-8')
    print('Wrote sensitivity tables to', output_dir, file=sys.stderr)


if __name__ == '__main__':
    main()
