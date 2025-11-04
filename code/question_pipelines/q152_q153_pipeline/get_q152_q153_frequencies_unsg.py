import pandas as pd
from IPython.display import display
import warnings
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the new dataset (UN Secretary-General speech corpus predictions)
df = pd.read_csv((Path(__file__).resolve().parent / "../../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv").resolve())

# Group by doc_id and A_YEAR, count occurrences of predicted_combined_label
grouped = df.groupby(['doc_id', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='count')

# Sort by doc_id, year, and descending count
grouped = grouped.sort_values(['doc_id', 'A_YEAR', 'count'], ascending=[True, True, False])

# Helper function to get top 2 labels per doc_id-year
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

# Apply the helper per doc_id-year
top_labels_df = grouped.groupby(['doc_id', 'A_YEAR']).apply(top_two_labels).reset_index()

# Save the results with new filename
output_path = "../../../output/question_pipeline_output/q152_q153_output/q152_year_top2_labels_unsg.csv"
top_labels_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# Display the result
display(top_labels_df)


