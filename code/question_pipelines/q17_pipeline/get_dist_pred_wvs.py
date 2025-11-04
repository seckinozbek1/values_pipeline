import pandas as pd
import matplotlib.pyplot as plt

# Load the Predictions CSV
df = pd.read_csv("../../../output/question_pipeline_output/q17_predictions/q17_predictions_top_score_filtered.csv")

# Count sentences per predicted label
label_counts = df['predicted_combined_label'].value_counts().sort_index()

# Plot histogram (bar chart) of counts per predicted_combined_label
plt.figure(figsize=(10,6))
label_counts.plot(kind='bar')
plt.title("Sentence Counts per Predicted Combined Label")
plt.xlabel("Predicted Combined Label")
plt.ylabel("Sentence Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Load WVS dataset
wvs_path = r"../../../output/master_code_prep_output/wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Filter valid values
valid_values = [1, 2]
q17_values = wvs_df['Q17']
q17_filtered = q17_values[q17_values.isin(valid_values)]

# Count frequencies
counts = q17_filtered.value_counts().sort_index()

# Plot histogram
plt.figure(figsize=(8,5))
counts.plot(kind='bar')
plt.title("Distribution of Q17 Values (1 to 2)")
plt.xlabel("Q17 Response")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
