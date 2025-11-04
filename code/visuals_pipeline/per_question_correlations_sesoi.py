import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import plotly.graph_objects as go

# File paths (WVS survey results + sentence prediction outputs)
wvs_path = r"../../output/master_code_prep_output/wvs7_full_data.csv"
file_paths = {
    'q8': '../../output/question_pipeline_output/q8_predictions/q8_predictions_filtered.csv',
    'q11': '../../output/question_pipeline_output/q11_predictions/q11_predictions_filtered.csv',
    'q17': '../../output/question_pipeline_output/q17_predictions/q17_predictions_filtered.csv',
    'q152': '../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv',
    'q154': '../../output/question_pipeline_output/q154_q155_predictions/q154_q155_predictions_filtered.csv',
    'q65': '../../output/question_pipeline_output/q65_predictions/q65_predictions_filtered.csv',
    'q69': '../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv',
    'q70': '../../output/question_pipeline_output/q70_predictions/q70_predictions_filtered.csv'
    # Note: q153 and q155 are derived from q152 and q154 respectively (see derived_sources below)

}

# Questions to process
questions = ['q8','q11','q17','q65','q69','q70','q152','q153','q154','q155']

# Treating the Q153 and Q155 'second-choice' questions the same way as the parent Q152 and Q154 questions respectively
derived_sources = {'q153': 'q152', 'q155': 'q154'}

# General Settings
B_BOOT = 3000   # Cluster bootstrap draws (resample countries).
ALPHA  = 0.05   # 95% CIs
SEED   = 42     # RNG seed for reproducibility
EPS    = 0.1    # Smallest Effect Size of Interest (SESOI) band: treat |r| < EPS as "practically zero"
# SESOI: defines a practical significance band.
# Correlations with |r| < 0.1 are interpreted as trivial correspondence between
# speech-implied and survey-measured value distributions. Only correlations whose
# entire bootstrap confidence interval lies outside ±0.1 are considered meaningful.
AX_PAD = 0.0   # x-axis padding; range = [-1-AX_PAD, 1+AX_PAD]

# Helper Functions
def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat DataFrame column names by replacing underscores with spaces and converting
    them to Title Case.

    Args:
        df (pd.DataFrame): Input DataFrame whose column names will be reformatted.

    Returns:
        pd.DataFrame: A new DataFrame with reformatted column names. The original
        DataFrame is not modified.
    """
    return df.rename(columns={c: c.replace('_', ' ').strip().title() for c in df.columns})

# Data prep (country × response)
def load_wvs_subset(path: str, qs: list) -> pd.DataFrame:
    """
    Load a subset of WVS Wave 7 data containing country identifiers, survey weights,
    and the specified question variables.

    Args:
        path (str): File path to the WVS dataset (CSV format).
        qs (list): List of question variable names (e.g., ["q152", "q65"]). These will
            be converted to uppercase to match WVS column naming.

    Returns:
        pd.DataFrame: A DataFrame containing the available columns among:
            B_COUNTRY_ALPHA (country identifier),
            A_YEAR (survey year),
            S018 (survey weight),
            and the requested question variables.
            Rows missing country identifiers or weights are removed.
    """
    hdr = pd.read_csv(path, nrows=0)
    need = ['B_COUNTRY_ALPHA', 'A_YEAR', 'S018'] + [q.upper() for q in qs]
    usecols = [c for c in need if c in hdr.columns]
    w = pd.read_csv(path, low_memory=False, usecols=usecols)
    return w.dropna(subset=['B_COUNTRY_ALPHA', 'S018'])

def wvs_weighted_props_country(wvs: pd.DataFrame, q: str) -> pd.DataFrame:
    """
    Compute the weighted distribution of responses to a WVS question at the country level.

    Args:
        wvs (pd.DataFrame): WVS data containing B_COUNTRY_ALPHA, S018 (weights),
            and the question column.
        q (str): Question variable name (case-insensitive, e.g., "q152").

    Returns:
        pd.DataFrame: A DataFrame with one row per (country, response_value), containing
        the weighted proportion of each response. Responses coded ≤ 0 are excluded.
        If the question is not found or no valid responses exist, an empty DataFrame
        with the appropriate columns is returned.
    """
    var = q.upper()
    if var not in wvs.columns:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','weighted_prop'])
    x = wvs[['B_COUNTRY_ALPHA', var, 'S018']].copy()
    x[var] = pd.to_numeric(x[var], errors='coerce')
    x = x.dropna(subset=[var])
    x = x[x[var] > 0]
    if x.empty:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','weighted_prop'])
    grp = x.groupby(['B_COUNTRY_ALPHA', var], as_index=False)['S018'].sum()
    tot = x.groupby('B_COUNTRY_ALPHA', as_index=False)['S018'].sum().rename(columns={'S018':'WT_TOTAL'})
    out = grp.merge(tot, on='B_COUNTRY_ALPHA')
    out['weighted_prop'] = out['S018'] / out['WT_TOTAL']
    out = out.rename(columns={var:'response_value'})
    out['response_value'] = out['response_value'].astype(int)
    return out[['B_COUNTRY_ALPHA','response_value','weighted_prop']]

def predicted_props_country_standard(pred_path: str, q: str) -> pd.DataFrame:
    """
    Compute the distribution of model-predicted response labels for a WVS question at
    the country level.

    Args:
        pred_path (str): Path to the predictions CSV containing B_COUNTRY_ALPHA and
            predicted_combined_label.
        q (str): Question identifier (e.g., "q152"). Used to select labels with the
            pattern Q##_r.

    Returns:
        pd.DataFrame: A DataFrame with one row per (country, response_value), containing
        the proportion of predicted sentences assigned to each response level. If no
        predictions are available for the question, an empty DataFrame with the correct
        column structure is returned.
    """
    df = pd.read_csv(pred_path)
    df = df.dropna(subset=['B_COUNTRY_ALPHA','predicted_combined_label'])
    prefix = q.upper() + '_'
    df = df[df['predicted_combined_label'].astype(str).str.startswith(prefix, na=False)]
    if df.empty:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','predicted_prop'])
    counts = df.groupby(['B_COUNTRY_ALPHA','predicted_combined_label']).size().reset_index(name='n')
    totals = df.groupby('B_COUNTRY_ALPHA').size().reset_index(name='N')
    m = counts.merge(totals, on='B_COUNTRY_ALPHA')
    m['response_value'] = m['predicted_combined_label'].str.extract(r'(\d+)$').astype(int)
    m = m[m['response_value'] >= 1]
    m['predicted_prop'] = m['n'] / m['N']
    return m[['B_COUNTRY_ALPHA','response_value','predicted_prop']]

def predicted_second_choice_from_base(pred_path: str, base_q: str) -> pd.DataFrame:
    """
    Compute the country-level distribution of 'second-choice' predicted responses for a
    base WVS question. For each (country, year), the second most frequent predicted
    response label is identified, aggregated across years, and normalized to form
    per-country proportions.

    Args:
        pred_path (str): Path to the predictions CSV containing B_COUNTRY_ALPHA,
            A_YEAR, and predicted_combined_label.
        base_q (str): Base question identifier (e.g., "q152"), used to select labels
            of the form BASE_Q_r.

    Returns:
        pd.DataFrame: A DataFrame with one row per (country, response_value) containing
        the normalized proportion of second-choice predictions. If no valid predictions
        exist, an empty DataFrame with the correct column structure is returned.
    """
    df = pd.read_csv(pred_path)
    df = df.dropna(subset=['B_COUNTRY_ALPHA','A_YEAR','predicted_combined_label'])
    prefix = base_q.upper() + '_'
    df = df[df['predicted_combined_label'].astype(str).str.startswith(prefix, na=False)]
    if df.empty:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','predicted_prop'])

    # counts per (country, year, label)
    counts = (df.groupby(['B_COUNTRY_ALPHA','A_YEAR','predicted_combined_label'])
                .size().reset_index(name='n'))

    # within each (country,year), take the 2nd most frequent label
    counts = counts.sort_values(['B_COUNTRY_ALPHA','A_YEAR','n'], ascending=[True, True, False])
    counts['rank'] = counts.groupby(['B_COUNTRY_ALPHA','A_YEAR'])['n'].rank(method='first', ascending=False)

    sec = counts[counts['rank'] == 2].copy()
    if sec.empty:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','predicted_prop'])

    # extract response_value and aggregate across years to country level
    sec['response_value'] = sec['predicted_combined_label'].str.extract(r'(\d+)$').astype(int)
    agg = (sec.groupby(['B_COUNTRY_ALPHA','response_value'], as_index=False)['n']
             .sum().rename(columns={'n':'second_total'}))

    # normalize per country so that second-choice distribution sums to 1
    denom = (agg.groupby('B_COUNTRY_ALPHA', as_index=False)['second_total']
                .sum().rename(columns={'second_total':'DEN'}))
    out = agg.merge(denom, on='B_COUNTRY_ALPHA')
    out = out[out['DEN'] > 0]
    out['predicted_prop'] = out['second_total'] / out['DEN']
    return out[['B_COUNTRY_ALPHA','response_value','predicted_prop']]

def predicted_props_country(q: str) -> pd.DataFrame:
    """
    Select the appropriate prediction aggregation method for a question. Standard
    questions use predicted_props_country_standard, while derived second-choice
    questions (Q153 and Q155) are computed from their respective base questions
    (Q152 and Q154).

    Args:
        q (str): Question identifier (e.g., "q152", "q153").

    Returns:
        pd.DataFrame: Country-level proportions of predicted response values for
        the specified question.
    """
    if q in derived_sources:
        base = derived_sources[q]
        return predicted_second_choice_from_base(file_paths[base], base)
    else:
        return predicted_props_country_standard(file_paths[q], q)

def build_common_grid_country(pred: pd.DataFrame, wvs: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a unified (country, response_value) grid for predicted and WVS
    distributions. The function intersects the set of countries present in both
    inputs and takes the union of available response levels, filling missing
    probabilities with zero to ensure comparable matrices.

    Args:
        pred (pd.DataFrame): Predicted proportions with columns:
            B_COUNTRY_ALPHA, response_value, predicted_prop.
        wvs (pd.DataFrame): WVS weighted proportions with columns:
            B_COUNTRY_ALPHA, response_value, weighted_prop.

    Returns:
        pd.DataFrame: A DataFrame indexed by (country, response_value) containing
        both predicted_prop and weighted_prop. Returns an empty DataFrame with the
        correct column structure if no overlap exists.
    """
    if pred.empty or wvs.empty:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','predicted_prop','weighted_prop'])
    countries = sorted(set(pred['B_COUNTRY_ALPHA']) & set(wvs['B_COUNTRY_ALPHA']))
    if not countries:
        return pd.DataFrame(columns=['B_COUNTRY_ALPHA','response_value','predicted_prop','weighted_prop'])
    levels = sorted(set(pred['response_value']).union(set(wvs['response_value'])))
    grid = pd.DataFrame({'B_COUNTRY_ALPHA': countries}).assign(key=1)\
           .merge(pd.DataFrame({'response_value': levels, 'key':1}), on='key').drop('key', axis=1)
    g = (grid.merge(pred, on=['B_COUNTRY_ALPHA','response_value'], how='left')
              .merge(wvs,  on=['B_COUNTRY_ALPHA','response_value'], how='left'))
    g['predicted_prop'] = g['predicted_prop'].fillna(0.0)
    g['weighted_prop']  = g['weighted_prop'].fillna(0.0)
    return g

# Correlations + cluster bootstrap CIs

def pearson_corr(x, y):
    """
    Compute the Pearson correlation coefficient between two numeric arrays,
    returning NaN if either array has zero variance.

    Args:
        x (array-like): First numeric vector.
        y (array-like): Second numeric vector of the same length.

    Returns:
        float: Pearson correlation coefficient, or NaN if undefined.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x*x).sum() * (y*y).sum())
    return np.nan if denom == 0 else float((x*y).sum() / denom)

def spearman_corr(x, y):
    """
    Compute the Spearman rank correlation coefficient between two numeric arrays.
    Each array is converted to ranks (average method for ties), and the Pearson
    correlation of the ranked values is returned.

    Args:
        x (array-like): First numeric vector.
        y (array-like): Second numeric vector of the same length.

    Returns:
        float: Spearman rank correlation coefficient, or NaN if undefined.
    """
    xr = pd.Series(x).rank(method='average').to_numpy()
    yr = pd.Series(y).rank(method='average').to_numpy()
    return pearson_corr(xr, yr)

def cluster_bootstrap_ci(P_mat, W_mat, B=B_BOOT, alpha=ALPHA, seed=SEED):
    """
    Compute cluster-robust bootstrap confidence intervals for Pearson and Spearman
    correlations between predicted and WVS response distributions. Countries are
    resampled with replacement, and country-level probability vectors are flattened
    across response levels for each bootstrap draw.

    Args:
        P_mat (np.ndarray): Matrix of predicted proportions with shape (Countries, Responses).
        W_mat (np.ndarray): Matrix of WVS weighted proportions with the same shape.
        B (int): Number of bootstrap resamples (default: B_BOOT).
        alpha (float): Significance level for confidence intervals (default: ALPHA).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (Pearson_low, Pearson_high, Spearman_low, Spearman_high) bootstrap
        percentile confidence bounds. Returns NaN values if correlations cannot be
        computed due to zero variance or insufficient data.
    """
    rng = np.random.default_rng(seed)
    C, _ = P_mat.shape
    if C < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    rP, rS = [], []
    for _ in range(B):
        idx = rng.integers(0, C, size=C)
        x = P_mat[idx, :].ravel()
        y = W_mat[idx, :].ravel()
        if np.isclose(np.var(x),0) or np.isclose(np.var(y),0):
            continue
        rP.append(pearson_corr(x, y))
        rS.append(spearman_corr(x, y))
    if len(rP) == 0 or len(rS) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    lo = 100*alpha/2; hi = 100*(1-alpha/2)
    return (np.percentile(rP, lo), np.percentile(rP, hi),
            np.percentile(rS, lo), np.percentile(rS, hi))

def evaluate_question_distributions(pred: pd.DataFrame, wvs: pd.DataFrame, q: str):
    """
    Compute Pearson and Spearman correlations between predicted and WVS response
    distributions for a single question, along with cluster-bootstrap confidence
    intervals based on country-level resampling.

    Args:
        pred (pd.DataFrame): Predicted proportions for the question, containing
            B_COUNTRY_ALPHA, response_value, and predicted_prop.
        wvs (pd.DataFrame): WVS weighted proportions for the same question, containing
            B_COUNTRY_ALPHA, response_value, and weighted_prop.
        q (str): Question identifier (e.g., "q152").

    Returns:
        dict: A dictionary containing the question code, sample sizes (number of
        countries, number of response levels, total grid cells), Pearson and
        Spearman correlations, and their corresponding cluster-bootstrap
        confidence interval bounds. If no valid data exist, NaN values are returned.
    """
    g = build_common_grid_country(pred, wvs)
    if g.empty:
        return {'question': q.upper(), 'countries': 0, 'response': 0, 'cells': 0,
                'pearson': np.nan, 'pearson_low': np.nan, 'pearson_high': np.nan,
                'spearman': np.nan, 'spearman_low': np.nan, 'spearman_high': np.nan}
    levels = sorted(g['response_value'].unique())
    countries = sorted(g['B_COUNTRY_ALPHA'].unique())
    P = g.pivot_table(index='B_COUNTRY_ALPHA', columns='response_value',
                      values='predicted_prop', fill_value=0.0).reindex(index=countries, columns=levels).to_numpy()
    W = g.pivot_table(index='B_COUNTRY_ALPHA', columns='response_value',
                      values='weighted_prop',  fill_value=0.0).reindex(index=countries, columns=levels).to_numpy()

    x = P.ravel(); y = W.ravel()
    if np.isclose(np.var(x),0) or np.isclose(np.var(y),0):
        r_p = r_s = np.nan
        p_lo = p_hi = s_lo = s_hi = np.nan
    else:
        r_p = pearson_corr(x, y)
        r_s = spearman_corr(x, y)
        p_lo, p_hi, s_lo, s_hi = cluster_bootstrap_ci(P, W)

    return {
        'question': q.upper(),
        'countries': len(countries),
        'response': len(levels),  # renamed for output
        'cells': len(x),
        'pearson': r_p, 'pearson_low': p_lo, 'pearson_high': p_hi,
        'spearman': r_s, 'spearman_low': s_lo, 'spearman_high': s_hi
    }

# Execute per-question evaluations

# Load the WVS data that contains all needed questions
wvs_all = load_wvs_subset(wvs_path, questions)

rows = []
for q in questions:
    # Obtain model-estimated response distributions for given question
    pred = predicted_props_country(q)

    # Obtain corresponding WVS weighted response distributions for the same question
    wsub = wvs_weighted_props_country(wvs_all, q)

    # Evaluate correlation and bootstrap uncertainty for the question and store results
    rows.append(evaluate_question_distributions(pred, wsub, q))

# Convert results to DataFrame and impose question ordering for consistent display
results = pd.DataFrame(rows)
results['question'] = pd.Categorical(results['question'], categories=[q.upper() for q in questions], ordered=True)
results = results.sort_values('question').reset_index(drop=True)

# Configure numeric display and select columns for output
pd.options.display.float_format = '{:.3f}'.format
cols = ['question','response','countries','cells',
        'pearson','pearson_low','pearson_high',
        'spearman','spearman_low','spearman_high']

# Display formatted table with header
display(Markdown(f"### Per-question correlation (country level; cluster bootstrap B={B_BOOT}; SESOI ±{EPS})"))

# Reformat column names for readability and present the table
tbl = prettify_columns(results[cols].copy()).rename(columns={'Response': 'No. of Responses'})
display(tbl)

# Define output path for LaTeX table
output_path = "../../output/visuals_pipeline_output/tables/per_question_correlations.tex"

# Convert table to LaTeX core content (no wrapper formatting)
latex_core = tbl.to_latex(index=False,
                          escape=True,
                          float_format="%.3f",
                          column_format='lrrrrrrrrr')  # 1 left + 9 right-aligned columns

# Wrap in LaTeX table environment with adjustbox scaling
latex_wrapped = r"""
\begin{table}[!ht]
\centering
\renewcommand{\arraystretch}{1.15}
\begin{adjustbox}{width=\textwidth}
\Large
%s
\end{adjustbox}
\end{table}
""" % latex_core

# Write LaTeX output to specified file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(latex_wrapped)

print(f"LaTeX table saved -> {output_path}")

# Plot the results
FONT_FAMILY   = "Times New Roman"
PEARSON_COLOR = "#055306"   # deep blue
GRID_MAIN     = "#D9D9D9"   # primary grid
GRID_LIGHT    = "#EFEFEF"   # secondary grid
SESOI_FILL    = "#C9D3E3"   # muted blue-gray

# Identify correlation estimates with confidence intervals indicating
# reliably positive, reliably negative, or reliably negligible effects
certain_pos  = results['pearson_low']  >  EPS
certain_neg  = results['pearson_high'] < -EPS
certain_null = (results['pearson_low'] >= -EPS) & (results['pearson_high'] <= EPS)
results['certain'] = (certain_pos | certain_neg | certain_null)

# Assign higher opacity to effects classified as certain
opacities = np.where(results['certain'], 0.95, 0.55)

fig = go.Figure()

# Highlight the SESOI region (|r| < EPS), indicating correlations that are considered
# practically negligible and therefore not substantively meaningful
fig.add_vrect(
    x0=-EPS, x1=EPS,
    fillcolor=SESOI_FILL,
    opacity=0.18,
    line_width=0,
    layer="below"
)

# Plot Pearson correlation estimates with bootstrap confidence intervals,
# using opacity to distinguish effects classified as statistically "certain"
fig.add_trace(go.Scatter(
    x=results['pearson'],
    y=results['question'],
    mode='markers',
    name='Pearson (country x response)',
    marker=dict(
        symbol='circle',
        size=10,
        color=PEARSON_COLOR,
        opacity=opacities,  # higher opacity = CI indicates clear positive/negative/null
        line=dict(color='black', width=0.7)
    ),
    error_x=dict(
        type='data',
        array=(results['pearson_high'] - results['pearson']).clip(lower=0).fillna(0),
        arrayminus=(results['pearson'] - results['pearson_low']).clip(lower=0).fillna(0),
        thickness=1.2,
        width=10,
        color=PEARSON_COLOR
    ),
    customdata=np.c_[results['countries'], results['response'],
                     results['pearson_low'], results['pearson_high'],
                     results['certain']],
    hovertemplate=("Q=%{y}<br>Pearson=%{x:.3f}"
                   "<br>CI=[%{customdata[2]:.3f}, %{customdata[3]:.3f}]"
                   "<br>countries=%{customdata[0]} • responses=%{customdata[1]}"
                   "<br>certain=%{customdata[4]}<extra></extra>")
))

# Configure overall plot layout: white theme, proportional vertical spacing to
# accommodate all questions, and consistent typography and margins
fig.update_layout(
    template="plotly_white",
    height=60*len(questions) + 240,
    margin=dict(l=120, r=40, t=40, b=50),
    legend_title_text='',
    font=dict(family=FONT_FAMILY, size=14),
    hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12))
)

# Configure x-axis to show correlations on a fixed scale with clear grid and reference lines
fig.update_xaxes(
    title='',
    range=[-1-AX_PAD, 1+AX_PAD],
    tick0=-1, dtick=0.5,
    showgrid=True, gridcolor=GRID_MAIN, gridwidth=1,
    zeroline=True, zerolinewidth=1.2, zerolinecolor="#9E9E9E",
    showline=True, linewidth=1, linecolor="#BDBDBD",
    ticks='outside',
    tickfont=dict(family=FONT_FAMILY, size=18)
)

# Configure y-axis to display questions in a fixed order with consistent spacing and light gridlines
fig.update_yaxes(
    title='',
    categoryorder='array',
    categoryarray=[q.upper() for q in questions],
    autorange='reversed',
    showgrid=True, gridcolor=GRID_LIGHT, gridwidth=1,
    showline=False,
    tickfont=dict(family=FONT_FAMILY, size=18)
)

# Add vertical reference line at r = 0 to indicate no association
fig.add_vline(x=0, line_dash='dash', line_color='#9E9E9E', line_width=1.2, opacity=0.9, layer='below')

# Save static figure as high-resolution PNG
fig.write_image("../../output/visuals_pipeline_output/figures/per_question_correlations_sesoi.png", scale=3)
print("Static figure saved -> ../../output/visuals_pipeline_output/figures/per_question_correlations_sesoi.png")

# Print and show the final figure
fig.show()