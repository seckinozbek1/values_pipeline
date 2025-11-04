import pandas as pd
import numpy as np
from IPython.display import display, HTML

file_paths = {
    'q8': '../../output/question_pipeline_output/q8_predictions/q8_predictions_filtered.csv',
    'q11': '../../output/question_pipeline_output/q11_predictions/q11_predictions_filtered.csv',
    'q17': '../../output/question_pipeline_output/q17_predictions/q17_predictions_filtered.csv',
    'q152': '../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv',
    'q154': '../../output/question_pipeline_output/q154_q155_predictions/q154_q155_predictions_filtered.csv',
    'q65': '../../output/question_pipeline_output/q65_predictions/q65_predictions_filtered.csv',
    'q69': '../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv',
    'q70': '../../output/question_pipeline_output/q70_predictions/q70_predictions_filtered.csv',
}

confidence_cols_map = {
    'q8': 'perc_diff_chance_abs',
    'q11': 'perc_diff_chance_abs',
    'q17': 'perc_diff_chance_abs',
}

response_type_map = {
    'Q8': 'Binary',
    'Q11': 'Binary',
    'Q17': 'Binary',
    'Q152': 'Categorical',
    'Q153': 'Categorical',
    'Q154': 'Categorical',
    'Q155': 'Categorical',
    'Q65': 'Ordinal',
    'Q69': 'Ordinal',
    'Q70': 'Ordinal',
}

combined_samples = []

for ds_name, path in file_paths.items():
    df = pd.read_csv(path)

    confidence_col = confidence_cols_map.get(ds_name, 'perc_above_chance')

    df_filtered = df.copy()

    if 'predicted_combined_label' in df_filtered.columns:
        label_col = 'predicted_combined_label'
    elif 'combined_label' in df_filtered.columns:
        label_col = 'combined_label'
    else:
        raise ValueError(f"No label column found in dataset {ds_name}")

    samples_per_label = []
    for label, group in df_filtered.groupby(label_col):
        group = group.assign(dataset=ds_name, label=label)
        samples_per_label.append(group)

    if samples_per_label:
        ds_samples = pd.concat(samples_per_label)
        combined_samples.append(ds_samples)

final_samples_df = pd.concat(combined_samples).reset_index(drop=True)

final_samples_df['dataset'] = final_samples_df['dataset'].str.upper()

summary_rows = []
for (dataset, label), group in final_samples_df.groupby(['dataset', 'label']):
    conf_col = confidence_cols_map.get(dataset.lower(), 'perc_above_chance')

    n_sentences = len(group)
    joint_mean = group['joint_score'].mean()
    joint_std = group['joint_score'].std()
    conf_mean = group[conf_col].mean()
    conf_std = group[conf_col].std()
    sim_mean = group['semantic_keyphrase_similarity'].mean()
    sim_std = group['semantic_keyphrase_similarity'].std()

    # Extract numeric label part
    if isinstance(label, str) and '_' in label:
        numeric_label = label.split('_')[1]
        try:
            numeric_label = int(numeric_label)
        except ValueError:
            numeric_label = label
    else:
        numeric_label = label

    prefix = dataset.split('_')[0] if '_' in dataset else dataset
    response_type = response_type_map.get(prefix, 'Unknown')

    summary_rows.append({
        'Question': dataset,
        'Response': numeric_label,
        'Response Type': response_type,
        'Count': n_sentences,
        'Mean Joint Score': joint_mean,
        'Std Joint Score': joint_std if not np.isnan(joint_std) else 0.0,
        'Mean Confidence': conf_mean,
        'Std Confidence': conf_std if not np.isnan(conf_std) else 0.0,
        'Mean Cosine Similarity': sim_mean,
        'Std Cosine Similarity': sim_std if not np.isnan(sim_std) else 0.0,
    })

summary_df = pd.DataFrame(summary_rows)

# Replace 'Q152' with 'Q152, Q153' and 'Q154' with 'Q154, Q155' for display and export
def fix_question_name(q):
    if q == 'Q152':
        return 'Q152, Q153'
    elif q == 'Q154':
        return 'Q154, Q155'
    else:
        return q

summary_df['Question'] = summary_df['Question'].apply(fix_question_name)

# Sort numerically by question number then label
summary_df['Q_num'] = summary_df['Question'].str.extract(r'Q(\d+)', expand=False).astype(int)
summary_df = summary_df.sort_values(by=['Q_num', 'Response']).reset_index(drop=True)
summary_df = summary_df.drop(columns=['Q_num'])

# CSS for centering table headers and cells
css = """
<style>
    table.dataframe th, table.dataframe td {
        text-align: center !important;
    }
</style>
"""

# Format floats to 4 decimals and hide index for HTML display
html = summary_df.style.format({
    'Mean Joint Score': '{:.4f}',
    'Std Joint Score': '{:.4f}',
    'Mean Confidence': '{:.4f}',
    'Std Confidence': '{:.4f}',
    'Mean Cosine Similarity': '{:.4f}',
    'Std Cosine Similarity': '{:.4f}',
}).hide(axis='index').to_html()

display(HTML(css + html))

# Save as LaTeX with 4 decimals and without index column

output_path = "../../output/visuals_pipeline_output/tables/pred_scores_summary_table.tex"

summary_df.to_latex(
    output_path,
    index=False,  # hides index column in LaTeX output
    longtable=True,
    caption="Summary Statistics for Prediction Scores by Question and Response",
    label="tab:pred_scores_summary_table",
    float_format="%.4f",
    column_format="lllrccccc",
    escape=True,
)

print("LaTeX table saved as pred_scores_summary_table.tex")