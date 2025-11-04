import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Item-to-file and WVS-column mapping (this structure is kept in case of further prints of figures for other questions)
ITEMS = {
    "Q8":   {"pred_path": "../../output/question_pipeline_output/q8_predictions/q8_predictions_filtered.csv",        "wvs_col": "Q8"},
    "Q11":  {"pred_path": "../../output/question_pipeline_output/q11_predictions/q11_predictions_filtered.csv",       "wvs_col": "Q11"},
    "Q17":  {"pred_path": "../../output/question_pipeline_output/q17_predictions/q17_predictions_filtered.csv",       "wvs_col": "Q17"},
    "Q152": {"pred_path": "../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv","wvs_col": "Q152"},
    "Q153": {"pred_path": "../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv","wvs_col": "Q153"},
    "Q154": {"pred_path": "../../output/question_pipeline_output/q154_q155_predictions/q154_q155_predictions_filtered.csv","wvs_col": "Q154"},
    "Q155": {"pred_path": "../../output/question_pipeline_output/q154_q155_predictions/q154_q155_predictions_filtered.csv","wvs_col": "Q155"},
    "Q65":  {"pred_path": "../../output/question_pipeline_output/q65_predictions/q65_predictions_filtered.csv",       "wvs_col": "Q65"},
    "Q69":  {"pred_path": "../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv",       "wvs_col": "Q69"},
    "Q70":  {"pred_path": "../../output/question_pipeline_output/q70_predictions/q70_predictions_filtered.csv",       "wvs_col": "Q70"},
}

# Path to WVS Wave 7 data file
WVS_PATH = r"../../output/master_code_prep_output/wvs7_full_data.csv"

# Helper functions operating at the country level (pooled across years)
def load_predictions(qid: str) -> pd.DataFrame:
    """
    Load the sentence-level prediction file for a given item.

    Parameters
    ----------
    qid : str
        Item identifier (e.g., "Q152"), used to locate the corresponding
        prediction file defined in the ITEMS mapping.

    Returns
    -------
    pd.DataFrame
        Raw prediction data containing at minimum:
        - predicted_combined_label : str
        - B_COUNTRY_ALPHA : str

    Raises
    ------
    ValueError
        If required columns are missing from the loaded file.
    """
    df = pd.read_csv(ITEMS[qid]["pred_path"])
    if "predicted_combined_label" not in df.columns or "B_COUNTRY_ALPHA" not in df.columns:
        raise ValueError(f"Prediction file for {qid} missing required columns.")
    return df

def aggregate_model_country(df_pred: pd.DataFrame, qid: str, level: int) -> pd.DataFrame:
    """
    Compute the per-country proportion of model-predicted sentences at a specified response level.

    Parameters
    ----------
    df_pred : pd.DataFrame
        DataFrame of model predictions containing `predicted_combined_label` and `B_COUNTRY_ALPHA`.
    qid : str
        Item identifier (e.g., "Q152"), used to extract the relevant label prefix.
    level : int
        Target response level (e.g., 2 corresponds to label "Q152_2").

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per country, containing:
        - B_COUNTRY_ALPHA : str
        - model_prop : float
          Proportion of predicted sentences at the target level (aggregated across all years).
    """
    prefix = f"{qid}_"
    target = f"{qid}_{level}" # retained for clarity; not used directly
    cur = df_pred[df_pred["predicted_combined_label"].str.startswith(prefix)].copy()
    cur["level"] = cur["predicted_combined_label"].str.split("_").str[-1].astype(int)

    # Count sentences by country and response level
    counts = cur.groupby(["B_COUNTRY_ALPHA", "level"]).size().rename("model_count").reset_index()
    
    # Compute per-country total sentence counts
    totals = counts.groupby("B_COUNTRY_ALPHA")["model_count"].sum().rename("model_total").reset_index()
    
    # Derive per-country proportions at each level
    merged = counts.merge(totals, on="B_COUNTRY_ALPHA", how="left")
    merged["model_prop"] = merged["model_count"] / merged["model_total"]

    # Retain only the target response level
    out = merged[merged["level"] == level][["B_COUNTRY_ALPHA", "model_prop"]].copy()
    return out

def aggregate_wvs_country(qid: str, level: int) -> pd.DataFrame:
    """
    Compute the per-country WVS weighted proportion at a specified response level.

    The weighted proportion is defined as:
        (country-level response share at `level`) x (country weight S018),
    pooled over all years. The country weight S018 is treated as constant per country.

    Parameters
    ----------
    qid : str
        Item identifier (e.g., "Q152") used to select the WVS column.
    level : int
        Target response level for which the weighted proportion is computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per country containing:
        - B_COUNTRY_ALPHA : str
        - wvs_weighted_proportion : float

    Raises
    ------
    ValueError
        If required columns (`B_COUNTRY_ALPHA`, item column, or `S018`) are missing.

    Notes
    -----
    Input values are coerced to numeric and rows with missing item responses or `S018`
    are dropped prior to aggregation.
    """
    wcol = ITEMS[qid]["wvs_col"]
    wvs = pd.read_csv(WVS_PATH, low_memory=False)

    req = ["B_COUNTRY_ALPHA", wcol, "S018"]
    for c in req:
        if c not in wvs.columns:
            raise ValueError(f"WVS missing required column: {c}")

    wvs[wcol] = pd.to_numeric(wvs[wcol], errors="coerce")
    wvs["S018"] = pd.to_numeric(wvs["S018"], errors="coerce")
    wvs = wvs.dropna(subset=[wcol, "S018"])

    # Count respondents by country and response level (pooled across years)
    level_counts = (
        wvs.groupby(["B_COUNTRY_ALPHA", wcol])
           .size().rename("resp_count").reset_index()
           .rename(columns={wcol: "level"})
    )
    # Compute per-country totals and extract the constant S018 weight
    country_totals = (
        wvs.groupby("B_COUNTRY_ALPHA")
           .agg(total_resp=("B_COUNTRY_ALPHA", "size"),
                country_weight=("S018", "first"))  # constant per country in WVS7 (per your spec)
           .reset_index()
    )

    # Derive per-country weighted proportions at each level
    merged = level_counts.merge(country_totals, on="B_COUNTRY_ALPHA", how="left")
    merged["prop_level"] = merged["resp_count"] / merged["total_resp"]
    merged["wvs_weighted_proportion"] = merged["prop_level"] * merged["country_weight"]
    merged["level"] = merged["level"].astype(int)

    # Retain only the target response level
    out = merged[merged["level"] == level][["B_COUNTRY_ALPHA", "wvs_weighted_proportion"]].copy()
    return out

# Plotting and correlation
def plot_dumbbell(qid: str, level: int, *, header_placeholder: bool = True):
    """
    Create a per-country dumbbell plot comparing model proportions to WVS weighted proportions.

    For the specified item (`qid`) and response `level`, this function:
    (i) aggregates sentence-level predictions to country-level model proportions,
    (ii) aggregates WVS responses to country-level weighted proportions (share x S018),
    (iii) inner-joins countries, computes Pearson's r, and
    (iv) renders a dumbbell chart (WVS vs. Model) using Plotly.

    Parameters
    ----------
    qid : str
        Item identifier (e.g., "Q152") used to locate prediction files and the WVS column.
    level : int
        Target response level to compare across countries.
    header_placeholder : bool, optional
        If True, increases the top margin to reserve header space, by default True.

    Returns
    -------
    go.Figure
        Plotly figure object containing the dumbbell plot.
    pd.DataFrame
        Merged country-level dataset with columns:
        `B_COUNTRY_ALPHA`, `wvs_weighted_proportion`, `model_prop`.
    float
        Pearson correlation coefficient between model proportions and WVS weighted proportions.

    Notes
    -----
    Exceptions from underlying loaders/aggregators (e.g., missing columns) propagate unchanged.
    """
    df_pred = load_predictions(qid)
    model_country = aggregate_model_country(df_pred, qid, level)
    wvs_country   = aggregate_wvs_country(qid, level)

    merged = pd.merge(model_country, wvs_country, on="B_COUNTRY_ALPHA", how="inner").dropna()
    merged = merged.sort_values("B_COUNTRY_ALPHA").reset_index(drop=True)

    r = merged["model_prop"].corr(merged["wvs_weighted_proportion"])

    # Construct figure with per-country connectors and markers
    fig = go.Figure()
    for _, row in merged.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["B_COUNTRY_ALPHA"], row["B_COUNTRY_ALPHA"]],
            y=[row["wvs_weighted_proportion"], row["model_prop"]],
            mode="lines",
            line=dict(color="lightgray", width=2),
            hoverinfo="skip",
            showlegend=False
        ))
        
    # Add WVS reference markers
    fig.add_trace(go.Scatter(
        x=merged["B_COUNTRY_ALPHA"], y=merged["wvs_weighted_proportion"],
        mode="markers", marker=dict(size=8),
        name="WVS (weighted proportion)",
        hovertemplate="Country: %{x}<br>WVS weighted proportion: %{y:.3f}<extra></extra>",
        showlegend=True
    ))
    
    # Add model markers with correlation in the legend label
    fig.add_trace(go.Scatter(
        x=merged["B_COUNTRY_ALPHA"], y=merged["model_prop"],
        mode="markers", marker=dict(size=8),
        name=f"Model (proportion)  |  Pearson r = {r:.3f}",
        hovertemplate="Country: %{x}<br>Model proportion: %{y:.3f}<extra></extra>",
        showlegend=True
    ))

    bold_times = "Times New Roman Bold, Times New Roman, Times, serif"

    # Compute axis range to accommodate both series
    ymax = float(np.nanmax(merged[["wvs_weighted_proportion", "model_prop"]].to_numpy()))
    ymin = float(np.nanmin(merged[["wvs_weighted_proportion", "model_prop"]].to_numpy()))
    pad  = (ymax - ymin) * 0.05 if np.isfinite(ymax - ymin) else 0.05
    y0, y1 = (ymin - pad, ymax + pad) if np.isfinite(ymax) else (0, 1)

    # Configure layout and axes
    fig.update_layout(
        template=None,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80 if header_placeholder else 40, b=70),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, yanchor="top",
                    font=dict(family=bold_times, size=12)),
        font=dict(family=bold_times, size=14),
        hoverlabel=dict(font_family=bold_times, font_size=12)
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(family=bold_times, size=12))
    fig.update_yaxes(range=[y0, y1], tickfont=dict(family=bold_times, size=12))
    return fig, merged, r

# Example usage
for lvl in [1, 2]:
    fig, data, r = plot_dumbbell("Q152", lvl, header_placeholder=True)
    save_path = f"../../output/visuals_pipeline_output/figures/Q152_{lvl}_dumbbell.png"
    
    fig.write_image(save_path, scale=2)  # Save to file
    fig.show()                           # Display in notebook / interactive environment
    
    print(f"Saved: {save_path}  |  Pearson r = {r:.3f}")