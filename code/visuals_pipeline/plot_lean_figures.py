import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Patch
import pycountry_convert as pc

# ----------------------------
# Global plotting style
# ----------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['axes.grid']       = False
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['axes.titlesize']  = 14
plt.rcParams['axes.labelsize']  = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ----------------------------
# Parameters
# ----------------------------
QUESTIONS     = ["Q8","Q11","Q17","Q65","Q69","Q70","Q152","Q153","Q154","Q155"]
WVS_PATH      = r"../../output/master_code_prep_output/wvs7_full_data.csv"
SAMPLE_N      = None
RANDOM_STATE  = 123
METRIC        = "js"
REGION_COLORS = {
    'Africa':'#d62728','Asia':'#2ca02c','Europe':'#1f77b4',
    'North America':'#ff7f0e','South America':'#9467bd',
    'Oceania':'#8c564b','Antarctica':'#e377c2',
    'Other':'#7f7f7f','Unknown':'#7f7f7f'
}

# ----------------------------
# Small helpers
# ----------------------------
def qcols(df, prefix):
    # Returns columns starting with the given prefix.
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]

def normalize_vec(v):
    # Normalizes a numeric vector to a non-negative distribution summing to one.
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, 0.0)
    v[v < 0] = 0.0
    s = v.sum()
    return v / s if s > 0 else v

def normalize_rows(df, cols):
    # Row-normalizes selected columns so each row sums to one.
    arr = df[cols].to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    arr[arr < 0] = 0.0
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    df[cols] = arr / row_sums
    return df

def js_divergence(p, q):
    # Computes Jensen–Shannon divergence between two discrete distributions.
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = np.nan_to_num(p, nan=1e-12); q = np.nan_to_num(q, nan=1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * (kl(p, m) + kl(q, m))

def lighten(color, amount=0.3):
    # Lightens a Matplotlib color by blending toward white.
    try:
        color = mc.cnames[color]
    except KeyError:
        pass
    rgb = np.array(mc.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(rgb + (white - rgb) * amount)

def add_region(df, iso3_col='B_COUNTRY_ALPHA'):
    # Adds a 'Region' column derived from ISO3 codes.
    def _region(iso3):
        try:
            iso2 = pc.country_alpha3_to_country_alpha2(iso3)
            cc = pc.country_alpha2_to_continent_code(iso2)
            return pc.convert_continent_code_to_continent_name(cc)
        except Exception:
            return "Unknown"
    out = df.copy()
    out['Region'] = out[iso3_col].apply(_region)
    return out

def rename_label_cols(df, target_qid):
    # Renames label-like columns (e.g., Q152_1) to the target prefix.
    ren = {}
    for c in df.columns:
        if re.match(r"^Q\d+_", str(c)):
            clean = re.sub(r"^Q\d+_", "", str(c))
            ren[c] = f"{target_qid}_{clean}"
    return df.rename(columns=ren)

# ----------------------------
# Core I/O and mapping
# ----------------------------
def prediction_paths(qid: str) -> tuple[str, str]:
    """
    Builds file-system paths for filtered and UNSG prediction CSVs.

    Parameters
    ----------
    qid : str
        Question identifier (e.g., 'Q8', 'Q155').

    Returns
    -------
    tuple[str, str]
        Paths to (filtered_csv, unsg_csv).

    Raises
    ------
    ValueError
        If the question identifier is not mapped.
    """
    q = qid.upper()
    if q in {"Q8","Q11","Q17","Q65","Q69","Q70"}:
        base_dir = f"../../output/question_pipeline_output/{q.lower()}_predictions"
        filtered = f"{q.lower()}_predictions_filtered.csv"
        unsg    = f"{q.lower()}_predictions_unsg.csv"
        return os.path.join(base_dir, filtered), os.path.join(base_dir, unsg)
    if q in {"Q152","Q153"}:
        base_dir = "../../output/question_pipeline_output/q152_q153_predictions"
        filtered = "q152_q153_predictions_filtered.csv"
        unsg    = "q152_q153_predictions_unsg.csv"
        return os.path.join(base_dir, filtered), os.path.join(base_dir, unsg)
    if q in {"Q154","Q155"}:
        base_dir = "../../output/question_pipeline_output/q154_q155_predictions"
        filtered = "q154_q155_predictions_filtered.csv"
        unsg    = "q154_q155_predictions_unsg.csv"
        return os.path.join(base_dir, filtered), os.path.join(base_dir, unsg)
    raise ValueError(f"No path mapping for {qid}")

def base_question(qid: str) -> str:
    """
    Returns the base question ID used for label prefixes.

    Parameters
    ----------
    qid : str
        Question identifier.

    Returns
    -------
    str
        'Q152' for 'Q153', 'Q154' for 'Q155', else the input uppercased.
    """
    q = qid.upper()
    return "Q152" if q == "Q153" else "Q154" if q == "Q155" else q

def read_wvs() -> pd.DataFrame:
    """
    Reads the WVS dataset used for response proportions.

    Returns
    -------
    pandas.DataFrame
        WVS dataset.

    Raises
    ------
    FileNotFoundError
        If the WVS CSV file does not exist.
    """
    if not os.path.exists(WVS_PATH):
        raise FileNotFoundError(WVS_PATH)
    return pd.read_csv(WVS_PATH, low_memory=False)

# ----------------------------
# Proportion builders
# ----------------------------
def country_props(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Computes per-country label proportions from filtered predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered predictions DataFrame.
    label_prefix : str
        Label prefix to select (e.g., 'Q154_').

    Returns
    -------
    pandas.DataFrame
        Wide table (rows: ISO3 countries, columns: label proportions).
    """
    df = df[df['predicted_combined_label'].str.startswith(label_prefix)]
    lab = df.groupby(['B_COUNTRY_ALPHA', 'predicted_combined_label']).size().reset_index(name='count')
    tot = df.groupby('B_COUNTRY_ALPHA').size().reset_index(name='total_count')
    lab = lab.merge(tot, on='B_COUNTRY_ALPHA', how='left')
    lab['prop'] = lab['count'] / lab['total_count']
    wide = lab.pivot(index='B_COUNTRY_ALPHA', columns='predicted_combined_label', values='prop').reset_index()
    for lbl in sorted(df['predicted_combined_label'].unique()):
        if lbl not in wide.columns:
            wide[lbl] = 0.0
    return wide

def second_label_country_props(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Computes per-country proportions for the second-most frequent label.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered predictions DataFrame.
    label_prefix : str
        Label prefix to select.

    Returns
    -------
    pandas.DataFrame
        Wide table with a single nonzero label per country (second-most frequent).
    """
    df = df[df['predicted_combined_label'].str.startswith(label_prefix)]
    all_labels = sorted(df['predicted_combined_label'].unique())
    countries = sorted(df['B_COUNTRY_ALPHA'].dropna().unique())

    counts = df.groupby(['B_COUNTRY_ALPHA', 'predicted_combined_label']).size().reset_index(name='count')
    totals = df.groupby('B_COUNTRY_ALPHA').size().reset_index(name='total_count')

    second = (
        counts.sort_values(['B_COUNTRY_ALPHA', 'count'], ascending=[True, False])
              .groupby('B_COUNTRY_ALPHA')
              .nth(1)
              .reset_index()
    )

    out = pd.DataFrame({'B_COUNTRY_ALPHA': countries})
    for lbl in all_labels:
        out[lbl] = 0.0

    if not second.empty:
        second = second.merge(totals, on='B_COUNTRY_ALPHA', how='left')
        second['prop'] = second['count'] / second['total_count']
        for _, r in second.iterrows():
            if r['total_count'] > 0:
                out.loc[out['B_COUNTRY_ALPHA'] == r['B_COUNTRY_ALPHA'], r['predicted_combined_label']] = r['prop']

    return out

def unsg_global_props(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Computes global label proportions from UNSG predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Global predictions DataFrame.
    label_prefix : str
        Label prefix to select.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with columns as label proportions.
    """
    df = df[df['predicted_combined_label'].str.startswith(label_prefix)]
    counts = df.groupby('predicted_combined_label').size().reset_index(name='count')
    total = counts['count'].sum()
    counts['prop'] = counts['count'] / total if total > 0 else 0.0
    wide = counts.pivot_table(index=None, columns='predicted_combined_label', values='prop').reset_index(drop=True)
    for lbl in sorted(df['predicted_combined_label'].unique()):
        if lbl not in wide.columns:
            wide[lbl] = 0.0
    return wide

def wvs_country_props(wvs_df: pd.DataFrame, question_id: str, valid_iso3: set) -> pd.DataFrame:
    """
    Computes weighted per-country proportions from WVS responses.

    Parameters
    ----------
    wvs_df : pandas.DataFrame
        WVS dataset.
    question_id : str
        WVS question column (e.g., 'Q155').
    valid_iso3 : set
        ISO3 codes to include.

    Returns
    -------
    pandas.DataFrame
        Wide table (rows: ISO3 countries, columns: weighted label proportions).
    """
    cols = ['B_COUNTRY_ALPHA', question_id, 'S018']
    wvs = wvs_df[cols].dropna(subset=cols)
    wvs = wvs[wvs['B_COUNTRY_ALPHA'].isin(valid_iso3)]
    wvs[question_id] = pd.to_numeric(wvs[question_id], errors='coerce')
    wvs = wvs.dropna(subset=[question_id])
    wvs = wvs[wvs[question_id] > 0]

    cats = sorted(wvs[question_id].unique())

    def _wprops(g):
        w = g['S018']; tw = w.sum()
        out = {}
        for c in cats:
            out[f"{question_id}_{int(c)}"] = (w * (g[question_id] == c)).sum() / tw if tw > 0 else 0.0
        return pd.Series(out)

    return wvs.groupby('B_COUNTRY_ALPHA').apply(_wprops).reset_index()

# ----------------------------
# Plot and compute
# ----------------------------
def plot_lean(lean_df: pd.DataFrame,
              prefix: str,
              metric: str,
              region_map: dict | None = None,
              region_colors: dict | None = None,
              min_width: float = 0.07,
              fontsize: int = 10):
    """
    Plots horizontal bar chart of lean values with optional region coloring.

    Parameters
    ----------
    lean_df : pandas.DataFrame
        DataFrame with 'B_COUNTRY_ALPHA' and 'lean_norm_<metric>'.
    prefix : str
        Label prefix used in the plot title (e.g., 'Q155_').
    metric : str
        Lean metric suffix (e.g., 'js').
    region_map : dict or None, optional
        ISO3-to-region mapping, by default None.
    region_colors : dict or None, optional
        Region-to-color mapping, by default None.
    min_width : float, optional
        Minimum bar magnitude for centered label placement, by default 0.07.
    fontsize : int, optional
        Country code label font size, by default 10.

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    metric_col = f'lean_norm_{metric}'
    df = lean_df.dropna(subset=[metric_col]).sort_values(metric_col).reset_index(drop=True)
    y = np.arange(len(df))
    val = df[metric_col].values
    iso = df['B_COUNTRY_ALPHA'].values

    fig_h = max(4, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axvline(0, color='gray', linestyle='--', zorder=2)
    ax.set_axisbelow(True)

    if region_map and region_colors:
        colors = [lighten(region_colors.get(region_map.get(c, 'Other'), '#7f7f7f'), 0.3) for c in iso]
    else:
        colors = [lighten('#7f7f7f', 0.3)] * len(iso)

    ax.barh(y, val, color=colors, edgecolor='black', zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, len(df) - 0.5)

    for t in ax.get_xticklabels():
        t.set_fontweight('bold')
        t.set_fontname('Times New Roman')

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)

    for i, (v, c) in enumerate(zip(val, iso)):
        ha = 'center'; x = v / 2
        if abs(v) < min_width:
            x = v + (0.05 if v >= 0 else -0.05)
            ha = 'left' if v >= 0 else 'right'
        ax.text(x, i, c, va='center', ha=ha, fontsize=fontsize, fontweight='bold')

    ax.text(-0.5, -0.08, 'UNSG Speech Zone', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontweight='bold', fontname='Times New Roman')
    ax.text( 0.5, -0.08, 'WVS Response Zone', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontweight='bold', fontname='Times New Roman')

    legend_handles = []
    if region_map and region_colors:
        seen = set()
        present_regions = [region_map.get(c, 'Other') for c in iso]
        ordered_unique = []
        for r in present_regions:
            if r not in seen:
                seen.add(r)
                ordered_unique.append(r)
        legend_regions = [r for r in ordered_unique if r not in {'Antarctica', 'Other', 'Unknown'}]
        for r in legend_regions:
            legend_handles.append(
                Patch(facecolor=lighten(region_colors.get(r, '#7f7f7f'), 0.3),
                      edgecolor='black', label=r)
            )
    if legend_handles:
        ncol = min(4, len(legend_handles))
        leg = ax.legend(handles=legend_handles, loc='upper center',
                        bbox_to_anchor=(0.5, -0.1), ncol=ncol,
                        frameon=False, title='Region',
                        prop={'family': 'Times New Roman', 'size': 10})
        if leg is not None and leg.get_title():
            leg.get_title().set_fontfamily('Times New Roman')
            leg.get_title().set_fontsize(11)
            leg.get_title().set_fontweight('bold')

    ax.grid(axis='x', linestyle=':', alpha=0.6, zorder=0)
    print(f'{prefix.rstrip("_")}: UNGD vs. UNSG Speeches and WVS Responses — Aggregated by Country')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    return fig, ax

def compute_lean(ungd_df: pd.DataFrame,
                 unsg_row: pd.DataFrame,
                 wvs_df: pd.DataFrame,
                 prefix: str,
                 metric: str = 'js',
                 sample_n: int | None = None,
                 random_state: int = 123,
                 region_map: dict | None = None,
                 region_colors: dict | None = None):
    """
    Computes per-country lean values and produces the corresponding plot.

    Parameters
    ----------
    ungd_df : pandas.DataFrame
        Country-level model distributions for the question.
    unsg_row : pandas.DataFrame
        Single-row DataFrame of the global UNSG distribution.
    wvs_df : pandas.DataFrame
        Country-level WVS distributions.
    prefix : str
        Question label prefix (e.g., 'Q155_').
    metric : str, optional
        Lean metric suffix, by default 'js'.
    sample_n : int or None, optional
        Number of countries to sample for plotting; None uses all, by default None.
    random_state : int, optional
        Seed for sampling, by default 123.
    region_map : dict or None, optional
        ISO3-to-region mapping, by default None.
    region_colors : dict or None, optional
        Region-to-color mapping, by default None.

    Returns
    -------
    tuple
        (lean_df, fig, ax) for downstream use.
    """
    cols = sorted(set(qcols(ungd_df, prefix)) & set(qcols(wvs_df, prefix)) & set(unsg_row.columns))
    if not cols:
        raise ValueError(f"No overlapping {prefix} columns after alignment.")

    ungd = ungd_df[['B_COUNTRY_ALPHA'] + cols].copy()
    wvs  = wvs_df[['B_COUNTRY_ALPHA'] + cols].copy()
    ungd[cols] = ungd[cols].apply(normalize_vec, axis=1, result_type='expand')
    wvs[cols]  = wvs[cols].apply(normalize_vec, axis=1, result_type='expand')
    ug = normalize_vec(unsg_row[cols].iloc[0].to_numpy())

    common_iso = sorted(set(ungd['B_COUNTRY_ALPHA']) & set(wvs['B_COUNTRY_ALPHA']))
    ungd = ungd[ungd['B_COUNTRY_ALPHA'].isin(common_iso)].reset_index(drop=True)
    wvs  = wvs[wvs['B_COUNTRY_ALPHA'].isin(common_iso)].reset_index(drop=True)

    recs = []
    for _, r in ungd.iterrows():
        iso3 = r['B_COUNTRY_ALPHA']
        p = r[cols].to_numpy()
        w = wvs.loc[wvs['B_COUNTRY_ALPHA'] == iso3, cols].iloc[0].to_numpy()
        lean_val = js_divergence(p, ug) - js_divergence(p, w)
        recs.append({'B_COUNTRY_ALPHA': iso3, f'lean_norm_{metric}': float(lean_val)})

    lean_df = pd.DataFrame(recs)

    if sample_n is not None:
        rng = np.random.default_rng(random_state)
        ids = lean_df['B_COUNTRY_ALPHA'].unique()
        k = min(sample_n, len(ids))
        keep = set(rng.choice(ids, size=k, replace=False))
        lean_df = lean_df[lean_df['B_COUNTRY_ALPHA'].isin(keep)].copy()

    fig, ax = plot_lean(lean_df, prefix=prefix, metric=metric, region_map=region_map, region_colors=region_colors)
    return lean_df, fig, ax

# ----------------------------
# Driver
# ----------------------------
def run_for_question(qid: str,
                     sample_n: int | None = SAMPLE_N,
                     random_state: int = RANDOM_STATE,
                     metric: str = METRIC,
                     region_colors: dict = REGION_COLORS):
    """
    Executes the full pipeline for a single question and saves the plot.

    Parameters
    ----------
    qid : str
        Question identifier (e.g., 'Q8', 'Q155').
    sample_n : int or None, optional
        Number of countries to sample for plotting; None uses all, by default SAMPLE_N.
    random_state : int, optional
        Seed for sampling, by default RANDOM_STATE.
    metric : str, optional
        Lean metric suffix (e.g., 'js'), by default METRIC.
    region_colors : dict, optional
        Mapping of region names to colors, by default REGION_COLORS.

    Returns
    -------
    None
        Saves a PNG plot to disk and prints the output path.
    """
    f_filtered, f_unsg = prediction_paths(qid)
    if not os.path.exists(f_filtered) or not os.path.exists(f_unsg):
        print(f"Skipping {qid}: missing inputs.")
        return

    df_filt = pd.read_csv(f_filtered)
    df_unsg = pd.read_csv(f_unsg)

    label_prefix = f"{base_question(qid)}_"
    if qid.upper() in {"Q153","Q155"}:
        ungd_country = second_label_country_props(df_filt, label_prefix)
    else:
        ungd_country = country_props(df_filt, label_prefix)

    unsg_one = unsg_global_props(df_unsg, label_prefix)

    ungd_country = rename_label_cols(ungd_country, qid.upper())
    unsg_one     = rename_label_cols(unsg_one, qid.upper())

    wvs_df = read_wvs()
    valid_iso3 = set(ungd_country['B_COUNTRY_ALPHA'].dropna().unique())
    wvs_country = wvs_country_props(wvs_df, qid.upper(), valid_iso3)

    ungd_country = add_region(ungd_country, 'B_COUNTRY_ALPHA')
    wvs_country  = add_region(wvs_country,  'B_COUNTRY_ALPHA')

    prefix = f"{qid.upper()}_"
    common = sorted(set(qcols(ungd_country, prefix)) &
                    set(qcols(unsg_one,     prefix)) &
                    set(qcols(wvs_country,  prefix)))
    if not common:
        print(f"Skipping {qid}: no overlapping label columns.")
        return

    ungd_norm = normalize_rows(ungd_country[['B_COUNTRY_ALPHA', 'Region'] + common].copy(), common)
    wvs_norm  = normalize_rows(wvs_country[['B_COUNTRY_ALPHA', 'Region'] + common].copy(),  common)
    unsg_norm = unsg_one[common].copy()
    unsg_norm.loc[0, common] = normalize_vec(unsg_norm.loc[0, common].to_numpy())

    region_map = dict(zip(ungd_norm['B_COUNTRY_ALPHA'], ungd_norm['Region']))

    _, fig, _ = compute_lean(
        ungd_norm, unsg_norm, wvs_norm,
        prefix=prefix, metric=metric, sample_n=sample_n, random_state=random_state,
        region_map=region_map, region_colors=region_colors
    )
    save_dir = "../../output/visuals_pipeline_output/figures/lean_plots"
    os.makedirs(save_dir, exist_ok=True)

    out = os.path.join(
        save_dir,
        f"lean_plot_{qid.lower()}_aggregated_sample{sample_n if sample_n is not None else 'all'}.png"
    ).replace("\\", "/")

    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

# ----------------------------
# Batch
# ----------------------------
for q in QUESTIONS:
    run_for_question(q)
print("Done.")