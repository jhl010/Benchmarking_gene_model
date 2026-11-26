#!/usr/bin/env python3
"""Batch BM-scaffold occurrence driver."""
from __future__ import annotations
import argparse, csv, gc, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CONTROL_DATASETS = [
    ('FDA', 'FDA_approve_sm_molecules/drugbank_smdrugs_smiles.txt'),
    ('CHEMBL', 'ChEMBL_clinical_smmolecules/ChEMBL_clinical_smmolecules.smi'),
    ('CrossDock', 'Crossdock_datasets/crossdock_ligand_smi.smi'),
]
MODEL_GLOB = 'combined_SMILES*generated.txt'
MAX_ATOM_PREFIX = 'PocketFlow_max_atom_num_'
FIELDNAMES = ['dataset','kind','input_file','benchmark','reference_file','matched_molecules','extractable_molecules','fraction','percent']
def discover_datasets(repo, include_max_atom):
    datasets=[]
    for label, rel in CONTROL_DATASETS:
        datasets.append({'dataset':label,'kind':'control','input_file':str(repo/rel),'source_rel':rel})
    for path in sorted(repo.glob('*/'+MODEL_GLOB)):
        folder=path.parent.name
        if not include_max_atom and folder.startswith(MAX_ATOM_PREFIX):
            continue
        kind='max_atom_sensitivity' if folder.startswith(MAX_ATOM_PREFIX) else 'model'
        datasets.append({'dataset':folder,'kind':kind,'input_file':str(path),'source_rel':str(path.relative_to(repo))})
    return datasets
def make_row(dataset, benchmark, reference_file, matched, extractable):
    fraction=(matched/extractable) if extractable else None
    return {'dataset':dataset['dataset'],'kind':dataset['kind'],'input_file':dataset['input_file'],'benchmark':benchmark,'reference_file':reference_file,'matched_molecules':matched,'extractable_molecules':extractable,'fraction':fraction,'percent':100.0*fraction if fraction is not None else None}
def run_zinc20(datasets, reference_file, jobs, rows):
    from BMScaffoldChecker import BMScaffoldChecker
    matcher=BMScaffoldChecker(str(reference_file), n_jobs=jobs)
    for dataset in datasets:
        path=Path(dataset['input_file'])
        if not path.exists():
            print('[warn] skipping missing input:', path, file=sys.stderr); continue
        try:
            _, matched, extractable=matcher.check_file(str(path))
        except Exception as exc:
            print('[warn] ZINC20 check failed for', dataset['dataset'], exc, file=sys.stderr); continue
        rows.append(make_row(dataset,'ZINC20',str(reference_file),matched,extractable))
        print('ZINC20', dataset['dataset'], f'{matched}/{extractable}', flush=True)
def run_zinc22(datasets, reference_file, cache_file, save_cache, rows):
    import BM_scaffold_checker_ZINC22 as zinc22
    scaffold_set=None
    if cache_file.exists():
        print('[cache] loading', cache_file, file=sys.stderr)
        scaffold_set=zinc22.load_set(str(cache_file))
    else:
        print('[build] reading', reference_file, file=sys.stderr)
        scaffold_set=zinc22.build_scaffold_set(str(reference_file))
        if save_cache and cache_file:
            print('[cache] saving', cache_file, file=sys.stderr)
            zinc22.dump_set(str(cache_file), scaffold_set)
    if scaffold_set is None:
        print('[warn] ZINC22 scaffold set could not be loaded; skipping ZINC22', file=sys.stderr); return
    for dataset in datasets:
        path=Path(dataset['input_file'])
        if not path.exists():
            print('[warn] skipping missing input:', path, file=sys.stderr); continue
        try:
            _, matched, extractable=zinc22.check_file(str(path), scaffold_set)
        except Exception as exc:
            print('[warn] ZINC22 check failed for', dataset['dataset'], exc, file=sys.stderr); continue
        rows.append(make_row(dataset,'ZINC22',str(reference_file),matched,extractable))
        print('ZINC22', dataset['dataset'], f'{matched}/{extractable}', flush=True)
def main():
    parser=argparse.ArgumentParser(description='Batch BM-scaffold coverage')
    parser.add_argument('--zinc20', default=str(REPO/'data'/'standardized_BM_scaffolds_ZINC20.smi'))
    parser.add_argument('--zinc22', default=str(REPO/'data'/'standardized_BM_scaffolds_ZINC22.smi'))
    parser.add_argument('--zinc22-cache', default=str(REPO/'data'/'standardized_BM_scaffolds_ZINC22.pkl.gz'))
    parser.add_argument('--output', default=str(REPO/'BM_scaffold_occurrence_results.csv'))
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--skip-zinc20', action='store_true')
    parser.add_argument('--skip-zinc22', action='store_true')
    parser.add_argument('--exclude-max-atom', action='store_true')
    parser.add_argument('--no-cache-save', action='store_true')
    args=parser.parse_args()
    datasets=discover_datasets(REPO, include_max_atom=not args.exclude_max_atom)
    if not datasets:
        raise SystemExit('No datasets discovered')
    print('Discovered', len(datasets), 'datasets', file=sys.stderr)
    rows=[]
    if not args.skip_zinc20:
        zinc20=Path(args.zinc20)
        if not zinc20.exists():
            print('[warn] ZINC20 reference not found:', zinc20, file=sys.stderr)
        else:
            run_zinc20(datasets,zinc20,args.jobs,rows)
    gc.collect()
    if not args.skip_zinc22:
        zinc22=Path(args.zinc22)
        if not zinc22.exists():
            print('[warn] ZINC22 reference not found:', zinc22, file=sys.stderr)
        else:
            run_zinc22(datasets,zinc22,Path(args.zinc22_cache),not args.no_cache_save,rows)
    if not rows:
        raise SystemExit('No coverage results were produced')
    output=Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader(); writer.writerows(rows)
    print('Wrote', len(rows), 'rows to', output, file=sys.stderr)
if __name__=='__main__':
    main()
