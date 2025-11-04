import json
import pandas as pd
import os
from IPython.display import display, HTML

# Load keyphrase JSONs and build a flat table
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

# Ingest JSON files into a DataFrame
records = []
for q, path in json_paths.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for label, phrases_list in data.items():
        resp_label = label.split('_')[-1] if '_' in label else label
        for item in phrases_list:
            records.append({
                'Question': q.upper(),
                'Response': resp_label,
                'Keyphrase': item['phrase'],
                'Cosine Similarity Score with Grouped Hypotheses': item['score'],
            })

keyphrases_simple = pd.DataFrame(records).reset_index(drop=True)

# Compute keyphrase counts per (Question, Response)
counts = (
    keyphrases_simple
    .groupby(['Question', 'Response'])
    .size()
    .reset_index(name='Keyphrase Count')
)

# Join counts and select one example per (Question, Response)
keyphrases_simple = keyphrases_simple.merge(counts, on=['Question', 'Response'])
keyphrases_one_example = (
    keyphrases_simple
    .groupby(['Question', 'Response'], as_index=False)
    .first()
)

# Reorder columns
cols = ['Question', 'Response', 'Keyphrase', 'Cosine Similarity Score with Grouped Hypotheses', 'Keyphrase Count']
keyphrases_one_example = keyphrases_one_example[cols]

# Sort rows by question number and response value (natural order)
keyphrases_one_example['Question_ord'] = keyphrases_one_example['Question'].str.extract(r'(\d+)').astype(int)
keyphrases_one_example['Response_ord'] = pd.to_numeric(keyphrases_one_example['Response'], errors='coerce')
keyphrases_one_example = (
    keyphrases_one_example
    .sort_values(by=['Question_ord', 'Response_ord', 'Response'])
    .drop(columns=['Question_ord', 'Response_ord'])
    .reset_index(drop=True)
)

# Render HTML preview
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(HTML(keyphrases_one_example.to_html(index=False)))

# Export LaTeX table
output_path = "../../output/visuals_pipeline_output/tables/keyphrase_examples.tex"
keyphrases_one_example.to_latex(
    output_path,
    index=False,
    longtable=True,
    caption="Keyphrase Example per Question and Response",
    label="tab:keyphrases_one_example_count_ordered",
    float_format="%.3f",
    column_format="lllrl",
    escape=True
)

print(f"Saved as {output_path}")
