#!/usr/bin/env python3
# Consolidated figure-and-statistics script for the locally deployed seven-model benchmark.
#
# This is the local counterpart to Yang et al.’s Picture_drawing.ipynb.  It
# loads the finalized molecule-level and summary analysis tables produced by
# the local pipeline, runs the manuscript statistical comparisons, and draws
# the publication figures.
#
# The statistical method intentionally differs from Yang et al. (who used
# ANOVA/Tukey).  Here every one of the seven locally deployed generators is
# compared with the CrossDocked2020 control for each of nine molecular
# properties using a two-sided Kolmogorov-Smirnov (KS) test.  Effect sizes
# are quantified with Cliff’s delta, and 95% confidence intervals for the
# median difference and Cliff’s delta are estimated from 1,000 nonparametric
# bootstrap resamples of molecular records (seed 20261119).  KS p values are
# Benjamini-Hochberg adjusted across the 63 generator-property comparisons.
#
# Figures produced
# ----------------
# Figure 1   Structural-prior metrics (ChEMBL ring-system metric, ZINC20 and
#            ZINC22 BM-scaffold coverage, PAINS pass) for the three controls
#            and seven generators.
# Figure 2   Normalized radar profiles of nine molecular properties.
# Figure 3   Violin plots of the nine molecular properties (nine panels).
# Figure 4   PocketFlow max_atom_num sensitivity (25, 30, 35, 40, 45).
# Figure S1  Cliff’s-delta forest plot (generator vs CrossDocked2020) together
#            with Table S4.
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors

import sascorer

RDLogger.DisableLog('rdApp.*')


ANALYSIS = ROOT / "analysis_output"
OUT = ROOT / "analysis_output" / "Picture_drawing"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261119)
N_BOOT = 1000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Molecular-property computation
# ---------------------------------------------------------------------------
CONTROLS = [
    ('FDA', 'FDA_approve_sm_molecules/drugbank_smdrugs_smiles_standardized.txt'),
    ('CHEMBL', 'ChEMBL_clinical_smmolecules/ChEMBL_clinical_smmolecules_standardized.smi'),
    ('crossdock', 'Crossdock_datasets/crossdock_ligand_smi_standardized.smi'),
]

MODELS = [
    ('targetdiff', 'TargetDiff'),
    ('decompdiff', 'DecompDiff'),
    ('pocketflow', 'PocketFlow'),
    ('graphbp', 'GraphBP'),
    ('apo2mol', 'Apo2Mol'),
    ('diffsbdd', 'DiffSBDD'),
    ('autofragdiff', 'AutoFragDiff'),
]

PROPERTY_NAMES = {
    'MW': 'Molecular Weight',
    'AliR_C': 'Aliphatic Rings',
    'AroR_C': 'Aromatic Rings',
    'ChiA_C': 'Chiral Atom Count',
    'SA': 'SA Score',
    'NHOH_C': 'NH/OH Count',
    'HetA_C': 'Heteroatom Count',
    'RotB_C': 'Rotatable Bonds',
    'BriA_C': 'Bridgehead Atom Count',
}

RADAR_PROPERTIES = list(PROPERTY_NAMES)
RADAR_DENOMINATORS = {
    'MW': 500,
    'AliR_C': 4,
    'AroR_C': 3,
    'ChiA_C': 6,
    'SA': 6,
    'NHOH_C': 6,
    'HetA_C': 10,
    'RotB_C': 8,
    'BriA_C': 2,
}


def input_datasets():
    datasets = []
    for label, rel in CONTROLS:
        datasets.append((label, 'control', ROOT / rel))
    for label, folder in MODELS:
        candidates = sorted((ROOT / folder).glob('combined_SMILES*generated.txt'))
        if not candidates:
            raise FileNotFoundError(f'No combined_SMILES*generated.txt found in {ROOT / folder}')
        datasets.append((label, 'model', candidates[0]))
    return datasets


def property_row(label, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        chiral_atoms = int(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        return {
            'dataset': label,
            'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
            'inchikey': Chem.MolToInchiKey(mol),
            'MW': float(Descriptors.MolWt(mol)),
            'AliR_C': int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            'AroR_C': int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            'ChiA_C': chiral_atoms,
            'SA': float(sascorer.calculateScore(mol)),
            'NHOH_C': int(Lipinski.NumHDonors(mol)),
            'HetA_C': int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            'RotB_C': int(Lipinski.NumRotatableBonds(mol)),
            'BriA_C': int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            'QED': float(QED.qed(mol)),
        }
    except Exception:
        return None


def build_molecule_table(datasets):
    rows = []
    counts = {}
    for label, kind, path in datasets:
        if not path.exists():
            print('[warn] missing input:', path, file=sys.stderr)
            continue
        n = 0
        with path.open(errors='replace') as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                row = property_row(label, parts[0])
                if row is None:
                    continue
                row['kind'] = kind
                rows.append(row)
                n += 1
        counts[label] = n
        print(label, n, 'valid molecules', file=sys.stderr)
    return pd.DataFrame(rows), counts


def build_long_table(molecules):
    records = []
    for short_name, long_name in PROPERTY_NAMES.items():
        for _, row in molecules.iterrows():
            records.append({
                'model': row['dataset'],
                'label': row['dataset'],
                'property': long_name,
                'value': row[short_name],
            })
    return pd.DataFrame(records)


def build_radar_table(molecules):
    records = []
    for dataset, group in molecules.groupby('dataset'):
        for prop in RADAR_PROPERTIES:
            mean = float(pd.to_numeric(group[prop], errors='coerce').mean())
            denom = RADAR_DENOMINATORS[prop]
            records.append({
                'dataset': dataset,
                'property': prop,
                'normalized_mean': mean / denom,
                'normalization_denominator': denom,
                'recomputed': True,
            })
    return pd.DataFrame(records)


def build_qed_summary(molecules):
    rows = []
    for dataset, group in molecules.groupby('dataset'):
        values = pd.to_numeric(group['QED'], errors='coerce').dropna()
        rows.append({
            'dataset': dataset,
            'n_qed': len(values),
            'mean_qed': float(values.mean()),
            'median_qed': float(values.median()),
            'sd_qed': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'q1_qed': float(values.quantile(0.25)),
            'q3_qed': float(values.quantile(0.75)),
            'min_qed': float(values.min()),
            'max_qed': float(values.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared dataset definitions
# ---------------------------------------------------------------------------
F1_MODELS = [
    "FDA", "CHEMBL", "CrossDock",
    "TargetDiff", "DecompDiff", "PocketFlow", "GraphBP", "Apo2Mol",
    "DiffSBDD", "AutoFragDiff",
]
F3_ORDER = [
    "FDA", "CHEMBL", "crossdock",
    "targetdiff", "decompdiff", "pocketflow", "graphbp", "apo2mol",
    "diffsbdd", "autofragdiff",
]
GENERATOR_DISPLAY = {
    "targetdiff": "TargetDiff",
    "decompdiff": "DecompDiff",
    "pocketflow": "PocketFlow",
    "graphbp": "GraphBP",
    "apo2mol": "Apo2Mol",
    "diffsbdd": "DiffSBDD",
    "autofragdiff": "AutoFragDiff",
}
CONTROL_COLOR = "#1761a0"
GENERATED_COLOR = "#9ed2ea"
CONTROL_COLOR_DARK = "#1f4e79"
GENERATED_COLOR_LIGHT = "#8ecae6"


def save_figure(fig, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.png/.svg/.pdf", flush=True)


# ---------------------------------------------------------------------------
# Figure 1: structural-prior metrics
# ---------------------------------------------------------------------------
F1_LITERATURE_PERCENT = {
    "min_freq >100": [72.5, 72.8, 69.2],
    "BM scaffold ZINC20": [55.6, 47.0, 41.3],
    "BM scaffold ZINC22": [60.4, 51.6, 50.0],
    "PAINS pass": [96.6, 96.5, 96.3],
}
F1_GENERATED_MODELS = [m for m in F1_MODELS if m not in ("FDA", "CHEMBL", "CrossDock")]


def _read_ring_summary(model: str) -> pd.Series:
    path = ROOT / model / "ring_systems_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ring summary for {model}: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty ring summary for {model}: {path}")
    return frame.iloc[0]


def _read_bm_fraction(model: str, benchmark: str) -> float:
    path = ROOT / "BM_scaffold_occurrence_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing BM occurrence results: {path}")
    frame = pd.read_csv(path)
    hit = frame[(frame["dataset"].eq(model)) & (frame["benchmark"].eq(benchmark))]
    if hit.empty:
        raise ValueError(f"Missing BM result for {model} {benchmark}")
    return float(hit.iloc[0]["fraction"])


def load_figure1_data() -> pd.DataFrame:
    rows = []
    metrics = ["min_freq >100", "BM scaffold ZINC20", "BM scaffold ZINC22", "PAINS pass"]
    for metric in metrics:
        for idx, dataset in enumerate(F1_MODELS):
            if dataset in ("FDA", "CHEMBL", "CrossDock"):
                percent = F1_LITERATURE_PERCENT[metric][idx]
                source = "literature Figure 3"
            elif metric == "min_freq >100":
                summary = _read_ring_summary(dataset)
                percent = 100.0 * float(summary["ChEMBL_ring_pass_fraction"])
                source = "current corrected analysis"
            elif metric == "PAINS pass":
                summary = _read_ring_summary(dataset)
                percent = 100.0 * float(summary["PAINS_pass_fraction"])
                source = "current analysis"
            else:
                benchmark = "ZINC20" if metric.endswith("ZINC20") else "ZINC22"
                percent = 100.0 * _read_bm_fraction(dataset, benchmark)
                source = "current corrected analysis"
            rows.append({
                "model": dataset,
                "metric": metric,
                "percentage": percent,
                "fraction": percent / 100.0,
                "source": source,
            })
    data = pd.DataFrame(rows)
    data["fraction"] = pd.to_numeric(data["fraction"], errors="coerce")
    return data


def draw_figure1(data: pd.DataFrame) -> None:
    metrics = ["min_freq >100", "BM scaffold ZINC20", "BM scaffold ZINC22", "PAINS pass"]
    titles = [
        "ChEMBL ring-system metric (min_freq > 100)",
        "BM scaffolds in ZINC20",
        "BM scaffolds in ZINC22",
        "PAINS pass",
    ]
    pivot = data.pivot(index="metric", columns="model", values="fraction").reindex(
        index=metrics, columns=F1_MODELS
    )
    colors = [CONTROL_COLOR_DARK] * 3 + [GENERATED_COLOR_LIGHT] * 7

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, metric, title, letter in zip(axes.flat, metrics, titles, ["A", "B", "C", "D"]):
        values = pivot.loc[metric].to_numpy(dtype=float)
        x = np.arange(len(F1_MODELS))
        ax.bar(x, values, color=colors, width=0.78, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(F1_MODELS, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Fraction")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(i, value + 0.02, f"{value * 100:.1f}", ha="center", va="bottom", fontsize=7)
    save_figure(fig, "Figure1_structural_prior")


# ---------------------------------------------------------------------------
# Figure 2: normalized radar profiles
# ---------------------------------------------------------------------------
def draw_figure2(data: pd.DataFrame) -> None:
    props = ["MW", "AliR_C", "AroR_C", "ChiA_C", "SA", "NHOH_C", "HetA_C", "RotB_C", "BriA_C"]
    pivot = data.pivot(index="property", columns="dataset", values="normalized_mean").reindex(index=props)
    slots = [
        "FDA", "targetdiff", "decompdiff", "pocketflow",
        "CHEMBL", "graphbp", "apo2mol", None,
        "crossdock", "diffsbdd", "autofragdiff", None,
    ]
    control_names = {"FDA", "CHEMBL", "crossdock"}
    angles = np.linspace(0, 2 * np.pi, len(props), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(19.75, 16.67), facecolor="white")
    gs = fig.add_gridspec(3, 4, left=0.035, right=0.965, top=0.965, bottom=0.035,
                          wspace=0.42, hspace=0.50)
    for index, model in enumerate(slots):
        if model is None:
            continue
        row, col = divmod(index, 4)
        ax = fig.add_subplot(gs[row, col], polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        color = CONTROL_COLOR_DARK if model in control_names else GENERATED_COLOR_LIGHT
        values = pivot[model].astype(float).to_numpy()
        values_closed = np.r_[values, values[0]]
        ax.plot(angles_closed, values_closed, color=color, lw=2.0)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.32)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888888")
        ax.set_rlabel_position(45)
        ax.set_xticks(angles)
        ax.set_xticklabels(props, fontsize=8.5, fontweight="bold", color=color)
        ax.tick_params(axis="x", pad=17)
        ax.grid(color="#d8d8d8", alpha=0.65, lw=0.8)
        ax.spines["polar"].set_color("#444444")
        ax.spines["polar"].set_linewidth(1.0)
        title = GENERATOR_DISPLAY.get(model, model)
        ax.set_title(title, y=1.27, fontsize=14, fontweight="bold", color=color, pad=0)
    save_figure(fig, "Figure2_radar")


# ---------------------------------------------------------------------------
# Figure 3: nine-property violin plots
# ---------------------------------------------------------------------------
def draw_figure3(data: pd.DataFrame) -> None:
    panels = [
        ("Molecular Weight", 1000),
        ("Aliphatic Rings", 10),
        ("Aromatic Rings", 8),
        ("Chiral Atom Count", 14),
        ("SA Score", 10),
        ("NH/OH Count", 10),
        ("Heteroatom Count", 30),
        ("Rotatable Bonds", 20),
        ("Bridgehead Atom Count", 8),
    ]
    colors = [CONTROL_COLOR] * 3 + [GENERATED_COLOR] * 7
    sns.set_theme(style="white", context="paper", font_scale=1.1)

    fig, axes = plt.subplots(3, 3, figsize=(21, 18), constrained_layout=True)
    for ax, (prop, ymax), letter in zip(axes.flat, panels, list("ABCDEFGHI")):
        subset = data[data["property"].eq(prop)]
        sns.violinplot(
            data=subset, x="label", y="value", order=[m for m in F3_ORDER],
            palette=colors, inner="box", cut=0, density_norm="width",
            linewidth=1.6, saturation=1, ax=ax,
        )
        for body in ax.collections:
            body.set_alpha(0.8)
        ax.set_xlabel("")
        ax.set_ylabel(prop, fontsize=16, fontweight="bold")
        ax.set_title(f"({letter}) {prop}", loc="left", fontsize=12)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    save_figure(fig, "Figure3_property_violins")


# ---------------------------------------------------------------------------
# Statistical method: KS + Cliff’s delta + bootstrap 95% CI
# ---------------------------------------------------------------------------
STATS_PROPERTY_NAMES = {
    "Molecular Weight": "Molecular weight (Da)",
    "Aliphatic Rings": "Aliphatic-ring count",
    "Aromatic Rings": "Aromatic-ring count",
    "Chiral Atom Count": "Chiral-atom count",
    "SA Score": "SA score",
    "NH/OH Count": "NH/OH group count",
    "Heteroatom Count": "Heteroatom count",
    "Rotatable Bonds": "Rotatable-bond count",
    "Bridgehead Atom Count": "Bridgehead-atom count",
}
GENERATOR_KEYS = [
    "targetdiff", "decompdiff", "pocketflow", "graphbp", "apo2mol",
    "diffsbdd", "autofragdiff",
]


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    ys = np.sort(y)
    lower = np.searchsorted(ys, x, side="left").sum()
    equal = (np.searchsorted(ys, x, side="right") - np.searchsorted(ys, x, side="left")).sum()
    return 2.0 * (lower + 0.5 * equal) / (len(x) * len(y)) - 1.0


def bootstrap_difference(x: np.ndarray, y: np.ndarray):
    md = np.empty(N_BOOT)
    cd = np.empty(N_BOOT)
    nx, ny = len(x), len(y)
    for i in range(N_BOOT):
        xb = x[RNG.integers(nx, size=nx)]
        yb = y[RNG.integers(ny, size=ny)]
        md[i] = np.median(xb) - np.median(yb)
        cd[i] = cliff_delta(xb, yb)
    return np.quantile(md, [0.025, 0.975]), np.quantile(cd, [0.025, 0.975])


def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def compute_effect_sizes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel_prop, stats_prop in STATS_PROPERTY_NAMES.items():
        values = {}
        for key in ["crossdock", *GENERATOR_KEYS]:
            series = data[(data["property"].eq(panel_prop)) & (data["model"].eq(key))]["value"]
            values[key] = series.to_numpy(dtype=float)
        control = values["crossdock"]
        for key in GENERATOR_KEYS:
            x = values[key]
            median_ci, delta_ci = bootstrap_difference(x, control)
            ks = stats.ks_2samp(x, control, method="asymp")
            rows.append({
                "property": stats_prop,
                "generator": GENERATOR_DISPLAY[key],
                "control": "CrossDocked2020",
                "n_generator": len(x),
                "n_control": len(control),
                "median_generator": np.median(x),
                "median_control": np.median(control),
                "median_difference": np.median(x) - np.median(control),
                "median_difference_ci_low": median_ci[0],
                "median_difference_ci_high": median_ci[1],
                "cliffs_delta": cliff_delta(x, control),
                "cliffs_delta_ci_low": delta_ci[0],
                "cliffs_delta_ci_high": delta_ci[1],
                "ks_statistic": ks.statistic,
                "ks_p_value": ks.pvalue,
            })
    result = pd.DataFrame(rows)
    result["ks_bh_adjusted_p_value"] = benjamini_hochberg(result["ks_p_value"].to_numpy())
    return result


def draw_figure_s1(effect_sizes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.4), constrained_layout=True)
    for ax, prop in zip(axes.flat, list(STATS_PROPERTY_NAMES.values())):
        subset = effect_sizes[effect_sizes["property"].eq(prop)]
        y = np.arange(len(subset))[::-1]
        estimate = subset["cliffs_delta"].to_numpy()
        low = subset["cliffs_delta_ci_low"].to_numpy()
        high = subset["cliffs_delta_ci_high"].to_numpy()
        ax.axvline(0, color="#555555", lw=0.8, zorder=0)
        ax.errorbar(
            estimate, y, xerr=np.vstack([estimate - low, high - estimate]),
            fmt="o", color="#2d6a9f", ecolor="#2d6a9f", capsize=2.5,
            markersize=4.8, lw=1.1,
        )
        ax.set_yticks(y, subset["generator"].tolist(), fontsize=8)
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(prop, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
    fig.supxlabel(
        "Cliff’s delta (generator vs CrossDocked2020; positive = higher generator values)",
        fontsize=10,
    )
    save_figure(fig, "FigureS1_cliffs_delta")


# ---------------------------------------------------------------------------
# Figure 4: PocketFlow max_atom_num sensitivity
# ---------------------------------------------------------------------------
def load_figure4_data() -> pd.DataFrame:
    data = pd.read_csv(
        ANALYSIS / "pocketflow_max_atom_sensitivity" / "overall_structural_prior_metrics_wilson95.csv"
    )
    return data[data["heavy_atom_stratum"].eq("overall")].copy()


def draw_figure4(data: pd.DataFrame) -> None:
    metric_colors = {
        "chembl_ring_min_freq_gt100": "#0F4D92",
        "zinc20_bm_coverage": "#42949E",
        "zinc22_bm_coverage": "#9A4D8E",
    }
    metric_markers = {
        "chembl_ring_min_freq_gt100": "o",
        "zinc20_bm_coverage": "s",
        "zinc22_bm_coverage": "^",
    }
    metric_labels = {
        "chembl_ring_min_freq_gt100": "ChEMBL ring-system metric (min_freq > 100)",
        "zinc20_bm_coverage": "BM scaffold coverage in ZINC20",
        "zinc22_bm_coverage": "BM scaffold coverage in ZINC22",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    for metric in metric_colors:
        subset = data[data["metric"].eq(metric)].sort_values("max_atom_num")
        x = subset["max_atom_num"].to_numpy(dtype=float)
        y = subset["percent"].to_numpy(dtype=float)
        low = subset["ci_low_percent"].to_numpy(dtype=float)
        high = subset["ci_high_percent"].to_numpy(dtype=float)
        ax.errorbar(
            x, y, yerr=[y - low, high - y], label=metric_labels[metric],
            color=metric_colors[metric], marker=metric_markers[metric],
            markersize=6, capsize=3, lw=1.4,
        )
    ax.set_xlabel("PocketFlow max_atom_num", fontsize=11)
    ax.set_ylabel("Proportion of standardized molecules (%)", fontsize=11)
    ax.set_xticks([25, 30, 35, 40, 45])
    ax.set_ylim(50, 90)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 0))
    save_figure(fig, "Figure4_pocketflow_max_atom_num")


def main() -> None:
    fig1 = load_figure1_data()
    draw_figure1(fig1)

    molecules, _ = build_molecule_table(input_datasets())
    if molecules.empty:
        raise SystemExit("No valid molecules found")
    fig2 = build_radar_table(molecules)
    draw_figure2(fig2)
    fig3 = build_long_table(molecules)
    draw_figure3(fig3)

    effect_sizes = compute_effect_sizes(fig3)
    effect_sizes.to_csv(OUT / "TableS4_effect_sizes_CrossDocked2020.csv", index=False)
    draw_figure_s1(effect_sizes)

    fig4 = load_figure4_data()
    draw_figure4(fig4)

    readme = (
        "# Picture_drawing.py outputs\n\n"
        "This directory contains the consolidated figures and statistics produced by "
        "`Benchmarking_gene_model_scripts/Picture_drawing.py`, the local counterpart to "
        "Yang et al.’s `Picture_drawing.ipynb`.\n\n"
        "Figure 1: ChEMBL ring-system metric, ZINC20/ZINC22 BM-scaffold coverage and PAINS pass.\n"
        "Figure 2: normalized radar profiles of nine molecular properties.\n"
        "Figure 3: violin plots of the nine molecular properties.\n"
        "Figure 4: PocketFlow max_atom_num sensitivity.\n"
        "Figure S1 + TableS4: KS tests, Cliff’s delta and 1,000-resample bootstrap 95% CIs.\n"
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("all figures and statistics written to", OUT, flush=True)


if __name__ == "__main__":
    main()
