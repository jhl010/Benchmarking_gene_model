#!/usr/bin/env python3
"""Ertl-Schuffenhauer synthetic accessibility score.

Adapted from the benchmarking repository by Yang et al. The SA-score algorithm
is unchanged; this copy locates fpscores.pkl.gz next to itself so it can be
imported from any working directory.
"""
from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

_FRAGMENT_SCORES_FILE = Path(__file__).resolve().parent / 'fpscores.pkl.gz'
_fscores = None


def readFragmentScores(name='fpscores'):
    """Load the Ertl fragment contribution table.

    When name is 'fpscores', the bundled fpscores.pkl.gz next to this file is
    used. Otherwise name is treated as a basename and '.pkl.gz' is appended.
    """
    global _fscores
    if name == 'fpscores':
        path = _FRAGMENT_SCORES_FILE
    else:
        path = Path(name + '.pkl.gz')
    if not path.exists():
        raise FileNotFoundError(f'Fragment scores file not found: {path}')
    with gzip.open(path, 'rb') as fh:
        data = pickle.load(fh)
    out_dict = {}
    for row in data:
        for bit_id in row[1:]:
            out_dict[bit_id] = float(row[0])
    _fscores = out_dict


def numBridgeheadsAndSpiro(mol, ri=None):
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    return n_bridgehead, n_spiro


def calculateScore(mol):
    global _fscores
    if _fscores is None:
        readFragmentScores()

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bit_id, count in fps.items():
        nf += count
        score1 += _fscores.get(bit_id, -4) * count
    score1 /= nf

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridgeheads, n_spiro = numBridgeheadsAndSpiro(mol, ri)
    n_macrocycles = 0
    for ring in ri.AtomRings():
        if len(ring) > 8:
            n_macrocycles += 1

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = 0.0
    if n_macrocycles > 0:
        macrocycle_penalty = math.log10(2)

    score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    sascore = 11.0 - (sascore - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


def processMols(mols):
    print('smiles\tName\tsa_score')
    for mol in mols:
        if mol is None:
            continue
        score = calculateScore(mol)
        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        print(f'{smiles}\t{name}\t{score:.3f}')


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) < 2:
        raise SystemExit('usage: python sascorer.py <smiles_file>')
    t1 = time.time()
    readFragmentScores()
    t2 = time.time()
    suppl = Chem.SmilesMolSupplier(sys.argv[1])
    t3 = time.time()
    processMols(suppl)
    t4 = time.time()
    print(f'Reading took {t2 - t1:.2f} seconds. Calculating took {t4 - t3:.2f} seconds', file=sys.stderr)
