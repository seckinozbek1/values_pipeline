import pandas as pd
from IPython.display import display
import os

ROOT = "../../output/question_pipeline_output"

folders = [
    "q8_predictions",
    "q11_predictions",
    "q17_predictions",
    "q65_predictions",
    "q69_predictions",
    "q70_predictions",
    "q152_q153_predictions",
    "q154_q155_predictions"
]

file_paths = [
    os.path.join(ROOT, folder, f"{folder}_filtered.csv").replace("\\", "/")
    for folder in folders
]

summary_rows = []

for path in file_paths:
    df = pd.read_csv(path)

    # Count occurrences of predicted_combined_label
    counts = df['predicted_combined_label'].value_counts().reset_index()
    counts.columns = ['predicted_combined_label', 'Count']

    # Extract Question and Response from predicted_combined_label
    counts['Question'] = counts['predicted_combined_label'].str.extract(r'^(Q\d+)_')[0]
    counts['Response'] = counts['predicted_combined_label'].str.extract(r'Q\d+_(.+)$')[0]

    # Keep only relevant columns
    counts = counts[['Question', 'Response', 'Count']]

    summary_rows.append(counts)

summary_df = pd.concat(summary_rows, ignore_index=True)

# Sort by numeric part of Question and Response ascending (failsafe)
summary_df['Question_num'] = summary_df['Question'].str.extract(r'Q(\d+)').astype(int)
summary_df = summary_df.sort_values(['Question_num', 'Response']).reset_index(drop=True)
summary_df = summary_df.drop(columns=['Question_num'])

display(summary_df)

# Save to LaTeX
output_path = "../../output/visuals_pipeline_output/tables/pred_data_count.tex"
with open(output_path, 'w') as f:
    f.write(summary_df.to_latex(index=False, float_format="%.0f"))

print(f"LaTeX table saved to: {output_path}")