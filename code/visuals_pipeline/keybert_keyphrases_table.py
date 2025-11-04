import json
import pandas as pd
import os
from IPython.display import display, HTML

# Extract KeyBERT keyphrases across questions to support interpretability analysis in downstream evaluation.
# The output is formatted for both on-screen inspection and LaTeX export to ensure reproducible reporting.
json_paths = {
    'q8': '../question_pipelines/q8_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q11': '../question_pipelines/q11_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q17': '../question_pipelines/q17_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q65': '../question_pipelines/q65_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q69': '../question_pipelines/q69_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q70': '../question_pipelines/q70_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q152': '../question_pipelines/q152_q153_pipeline/logs_and_metrics/keybert_keyphrases.json',
    'q154': '../question_pipelines/q154_q155_pipeline/logs_and_metrics/keybert_keyphrases.json',
}

records = []

# Load KeyBERT outputs and store each phrase and score in long-table format.
for q_key, path in json_paths.items():
    if not os.path.exists(path):
        print(f"Warning: File not found for {q_key}: {path}")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data is dict: label -> list of {"phrase":..., "score":...}
    for label, keyphrases_list in data.items():
        # Extract the number part after underscore, fallback to label if no underscore
        response_label = label.split('_')[-1] if '_' in label else label
        for item in keyphrases_list:
            records.append({
                'Question': q_key.upper(),
                'Response': response_label,
                'Keyphrase': item['phrase'],
                'Cosine Similarity Score': item['score'],
            })

keyphrases = pd.DataFrame(records).reset_index(drop=True)

# Display full results in HTML for qualitative review.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(HTML(keyphrases.to_html(index=False)))

# Export a LaTeX longtable for manuscript integration.

output_path = "../../output/visuals_pipeline_output/tables/keybert_keyphrases_table.tex"

keyphrases.to_latex(
    output_path,
    index=False,
    longtable=True,
    caption="Extracted KeyBERT Keyphrases by Question and Response Label",
    label="tab:keybert_keyphrases",
    float_format="%.3f",
    column_format="lllr",
    escape=True
)

print(f"LaTeX table saved as {output_path}")
