import pandas as pd
import numpy as np
import random
from IPython.display import display, HTML

num_sentences = 3

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

# Load master WVC dataset with combined_label and response_hypothesis columns
main_data_path = "../../output/master_code_prep_output/main_data_complete_long.csv"
master_df = pd.read_csv(main_data_path)

# Keep only needed columns and drop duplicates for faster lookups
master_map_df = master_df[['combined_label', 'response_hypothesis']].drop_duplicates()
# Create a dictionary for fast lookup: combined_label -> response_hypothesis
response_hypothesis_lookup = master_map_df.set_index('combined_label')['response_hypothesis'].to_dict()

def int_to_roman(n):
    """
    Convert an integer to its Roman numeral (I-X) or return its string form if out of range.

    Parameters
    ----------
    n : int
        Integer to convert.

    Returns
    -------
    str
        Roman numeral for 1-10, else the integer as a string.
    """
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    return roman_numerals[n-1] if 1 <= n <= 10 else str(n)

RANDOM_SEED = 49202
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

all_rows = []

for ds_name, path in file_paths.items():
    df = pd.read_csv(path)
    confidence_col = confidence_cols_map.get(ds_name, 'perc_above_chance')

    # Copy the full dataframe
    df_filtered = df.copy()

    if 'predicted_combined_label' in df_filtered.columns:
        label_col = 'predicted_combined_label'
    elif 'combined_label' in df_filtered.columns:
        label_col = 'combined_label'
    else:
        raise ValueError(f"No label column found in dataset {ds_name}")

    if 'sentence' in df_filtered.columns:
        text_col = 'sentence'
    elif 'text' in df_filtered.columns:
        text_col = 'text'
    else:
        raise ValueError(f"No text column found in dataset {ds_name}")

    for (label, group) in df_filtered.groupby(label_col):
        # Extract number after underscore in label as string (key for master lookup)
        if isinstance(label, str) and '_' in label:
            numeric_label_str = label.split('_')[1]
            try:
                numeric_label = int(numeric_label_str)
            except ValueError:
                numeric_label = label
                numeric_label_str = label
        else:
            numeric_label = label
            numeric_label_str = str(label)

        sampled_sentences = group[text_col].sample(n=min(num_sentences, len(group)), random_state=RANDOM_SEED).tolist()
        while len(sampled_sentences) < num_sentences:
            sampled_sentences.append('')

        response_hypo = (
            response_hypothesis_lookup.get(label) or
            response_hypothesis_lookup.get(f"{ds_name.upper()}_{numeric_label_str}") or
            response_hypothesis_lookup.get(numeric_label_str) or
            ""
        )
        response_hypo = response_hypo.lower() if response_hypo else ""

        row = {
            'Question': ds_name.upper(),
            'Response': numeric_label,
            'Response Hypothesis': response_hypo,
        }
        for i in range(num_sentences):
            roman_num = int_to_roman(i+1)
            row[f'Sentence {roman_num}'] = sampled_sentences[i]

        all_rows.append(row)

sentences_df = pd.DataFrame(all_rows)

# Replace Q152 and Q154 with combined labels as requested
def fix_question_name(q):
    if q == 'Q152':
        return 'Q152, Q153'
    elif q == 'Q154':
        return 'Q154, Q155'
    else:
        return q

sentences_df['Question'] = sentences_df['Question'].apply(fix_question_name)

# Sort numerically by question number then label
sentences_df['Q_num'] = sentences_df['Question'].str.extract(r'Q(\d+)', expand=False).astype(int)
sentences_df = sentences_df.sort_values(by=['Q_num', 'Response']).reset_index(drop=True)
sentences_df = sentences_df.drop(columns=['Q_num'])

# Generate HTML with centered sentences in sentence columns
html = sentences_df.to_html(index=False)

css = f"""
<style>
    table.dataframe th, table.dataframe td {{
        text-align: left;
    }}
    table.dataframe td:nth-child(n+4):nth-child(-n+{3 + num_sentences}) {{
        text-align: center;
    }}
    table.dataframe th:nth-child(n+4):nth-child(-n+{3 + num_sentences}) {{
        text-align: center;
    }}
</style>
"""

display(HTML(css + html))

# Save as LaTeX with dynamic columns
col_format = 'lll' + 'c' * num_sentences

output_path = "../../output/visuals_pipeline_output/tables/sample_sentences.tex"


sentences_df.to_latex(
    output_path,
    index=False,
    caption=f"Sampled sentences for each Question and Response ({num_sentences} Sentences per Label)",
    label="tab:sampled_sentences",
    column_format=col_format,
    escape=True,
)

print(f"LaTeX table saved as sample_sentences.tex with {num_sentences} sentences per label.")