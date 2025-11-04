import pandas as pd
from IPython.display import display
import warnings
import os
from pathlib import Path

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load predictions CSV
df = pd.read_csv((Path(__file__).resolve().parent / "../../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv").resolve())

# Group by country-year and combined_label, count occurrences
grouped = df.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='count')

# Sort by country-year and descending count
grouped = grouped.sort_values(['B_COUNTRY_ALPHA', 'A_YEAR', 'count'], ascending=[True, True, False])

# Helper function to get top 2 labels per country-year
def top_two_labels(sub_df):
    """
    Return the two most frequent labels and their counts from a grouped DataFrame.

    If fewer than two labels exist, fills missing values with None and 0. Also
    includes a boolean flag indicating whether a second label is present.

    Args:
        sub_df (pandas.DataFrame): DataFrame with 'predicted_combined_label' and 'count' columns,
            sorted in descending order by count.

    Returns:
        pandas.Series: Contains top two labels, their counts, and 'has_second_label' flag.
    """
    top_labels = sub_df.head(2)
    labels = top_labels['predicted_combined_label'].tolist()
    counts = top_labels['count'].tolist()
    while len(labels) < 2:
        labels.append(None)
        counts.append(0)
    has_second_label = labels[1] is not None
    return pd.Series({
        'most_frequent_label': labels[0],
        'most_frequent_count': counts[0],
        'second_most_frequent_label': labels[1],
        'second_most_frequent_count': counts[1],
        'has_second_label': has_second_label
    })

# Apply helper per country-year
top_labels_df = grouped.groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).apply(top_two_labels).reset_index()

# Load WVS full data
wvs_path = r"../../../output/master_code_prep_output/wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Get unique country-year pairs from WVS
wvs_country_years = set(zip(wvs_df['B_COUNTRY_ALPHA'], wvs_df['A_YEAR']))

# Filter top_labels_df to keep only country-year pairs that exist in WVS data
top_labels_df = top_labels_df[
    top_labels_df.apply(lambda row: (row['B_COUNTRY_ALPHA'], row['A_YEAR']) in wvs_country_years, axis=1)
].reset_index(drop=True)

# Separate country-years with and without second most frequent label
without_second_label = top_labels_df[~top_labels_df['has_second_label']]

# Print the list of country-year pairs without second most frequent label
print("\nCountry-year pairs without a second most frequent label:")
for idx, row in without_second_label.iterrows():
    print(f"{row['B_COUNTRY_ALPHA']} - {row['A_YEAR']}")

# Print the full dataset
display(top_labels_df)

# Save the results with new filename
output_path = "../../../output/question_pipeline_output/q152_q153_output/q152_q153_country_year_top2.csv"

# Ensure parent directory exists
parent_dir = os.path.dirname(output_path)
os.makedirs(parent_dir, exist_ok=True)

# Resolve path for safe writing
abs_out = Path(output_path).resolve()

# Write results
top_labels_df.to_csv(abs_out, index=False, encoding="utf-8")
print(f"Saved the results to {output_path}")