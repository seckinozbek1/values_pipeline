import os
import pandas as pd
from IPython.display import display

# Define root directory
ROOT = "../../data/labeled_data"

# Define filenames
file_names = [
    'Q152_mmr_selected_labeled_combined.csv',
    'Q154_mmr_selected_labeled_combined.csv',
    'Q8_mmr_selected_labeled_combined.csv',
    'Q11_mmr_selected_labeled_combined.csv',
    'Q17_mmr_selected_labeled_combined.csv',
    'Q65_mmr_selected_labeled_combined.csv',
    'Q69_mmr_selected_labeled_combined.csv',
    'Q70_mmr_selected_labeled_combined.csv'
]

summary_rows = []

for fn in file_names:
    fp = os.path.join(ROOT, fn)   # build full path
    df = pd.read_csv(fp)
    # Extract Question prefix (before underscore)
    df['Question'] = df['combined_label'].str.extract(r'^(Q\d+)_')[0]
    # Extract Response (after underscore)
    df['Response'] = df['combined_label'].str.extract(r'Q\d+_(.+)$')[0]
    counts = df.groupby(['Question', 'Response']).size().reset_index(name='Count')
    summary_rows.append(counts)

summary_df = pd.concat(summary_rows, ignore_index=True)

# Sort by numeric part of Question string
summary_df['Question_num'] = summary_df['Question'].str.extract(r'Q(\d+)').astype(int)
summary_df = summary_df.sort_values(['Question_num', 'Response']).reset_index(drop=True)
summary_df = summary_df.drop(columns=['Question_num'])

display(summary_df)

# Save as LaTeX
output_path = "../../output/visuals_pipeline_output/tables/labeled_data_count.tex"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    f.write(summary_df.to_latex(index=False, float_format="%.0f"))
print(f"LaTeX table saved to: {output_path}")
