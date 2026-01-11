import os
import pandas as pd

# Base directory for input file
BASE_DIR = os.path.join("..", "..", "output", "master_code_prep_output")

# Path to input CSV
input_file = os.path.join(BASE_DIR, "top_scored_sentences.csv")
df = pd.read_csv(input_file)

# Create a subfolder for IQR samples inside BASE_DIR
IQR_SAMPLES_DIR = os.path.join(BASE_DIR, "iqr_samples")
os.makedirs(IQR_SAMPLES_DIR, exist_ok=True)

# Get unique broad_qid values
unique_qids = df['broad_qid'].unique()

for qid in unique_qids:
    df_qid = df[df['broad_qid'] == qid]

    # Calculate 25th and 75th percentile (IQR) for total_score in this group
    lower_bound = df_qid['total_score'].quantile(0.25)
    upper_bound = df_qid['total_score'].quantile(0.75)

    # Filter sentences within the IQR range
    iqr_samples = df_qid[
        (df_qid['total_score'] >= lower_bound) &
        (df_qid['total_score'] <= upper_bound)
    ]

    print(f"broad_qid: {qid} - Selected {len(iqr_samples)} IQR-range candidate samples for annotation.")

    # Save to CSV file inside the iqr_samples subfolder
    output_file = os.path.join(IQR_SAMPLES_DIR, f"{qid}_iqr_samples.csv")
    iqr_samples.to_csv(output_file, index=False)
