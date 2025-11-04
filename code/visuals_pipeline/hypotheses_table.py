import os
import random
import pandas as pd
from IPython.display import display, HTML

# Base directory containing the labeled hypothesis CSV files
CSV_BASE = "../../data/labeled_data"

def load_hypotheses(question, resp_label):
    """
    Load the response hypothesis and up to three adapted hypotheses for a given
    question-response label. The function reads the labeled selection CSV,
    extracts the hypothesis text for the specified combined label, and returns
    a list consisting of one response hypothesis followed by three adapted
    hypotheses. Missing entries are returned as None.

    Parameters
    ----------
    question : str
        Short question identifier (e.g., 'q8', 'q152').
    resp_label : str
        Response label suffix used to construct the combined label.

    Returns
    -------
    list
        [response_hypothesis, adapted_I, adapted_II, adapted_III]
    """
    csv_path = os.path.join(CSV_BASE, f"{question.upper()}_mmr_selected_labeled_combined.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file missing: {csv_path}")
        return [None] * 4

    df = pd.read_csv(csv_path)
    full_label = f"{question.upper()}_{resp_label}"

    resp_hypos = df.loc[df['combined_label'] == full_label, 'response_hypothesis'].dropna().unique()
    resp_hypo = resp_hypos[0] if len(resp_hypos) > 0 else None

    adapted_hypos = df.loc[df['combined_label'] == full_label, 'adapted_hypotheses'].dropna().unique()
    if len(adapted_hypos) == 0:
        adapted_split = [None] * 3
    else:
        adapted_split = adapted_hypos[0].split('|')
        adapted_split += [None] * (3 - len(adapted_split))
        adapted_split = adapted_split[:3]

    return [resp_hypo] + adapted_split

def get_response_labels(question):
    """
    Return unique response label suffixes for a question by parsing the
    'combined_label' column in the labeled selection file.

    Parameters
    ----------
    question : str
        Short question identifier (e.g., 'q8', 'q152').

    Returns
    -------
    list
        Unique response label suffixes as strings.
    """
    csv_path = os.path.join(CSV_BASE, f"{question.upper()}_mmr_selected_labeled_combined.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file missing: {csv_path}")
        return []

    df = pd.read_csv(csv_path, usecols=['combined_label'])
    mask = df['combined_label'].astype(str).str.startswith(f"{question.upper()}_")
    labels = (
        df.loc[mask, 'combined_label']
          .astype(str)
          .str.split('_')
          .str[-1]
          .dropna()
          .unique()
    )
    return list(labels)

# Fixed seed for reproducible question sampling
all_questions = ['q8', 'q11', 'q17', 'q65', 'q69', 'q70', 'q152', 'q154']
random.seed(42)
sampled_questions = random.sample(all_questions, 4)

# Construct (Question, Response) pairs from labeled files (e.g., Q8_2 → Q8, 2)
unique_pairs = []
for q in sampled_questions:
    for resp in get_response_labels(q):
        unique_pairs.append((q.upper(), resp))
unique_pairs = sorted(set(unique_pairs))

# Assemble the hypothesis table
records = []
for q, resp in unique_pairs:
    hyp = load_hypotheses(q, resp)
    records.append({
        'Question': q,
        'Response': resp,
        'Response Hypothesis': hyp[0],
        'Adapted Hypothesis I': hyp[1],
        'Adapted Hypothesis II': hyp[2],
        'Adapted Hypothesis III': hyp[3]
    })

hypotheses = pd.DataFrame(records)

# Sort rows in ascending order by question number and response value
hypotheses['Question_ord'] = hypotheses['Question'].str.extract(r'(\d+)').astype(int)
hypotheses['Response_ord'] = pd.to_numeric(hypotheses['Response'], errors='coerce')

hypotheses = (
    hypotheses
      .sort_values(by=['Question_ord', 'Response_ord', 'Response'])
      .drop(columns=['Question_ord', 'Response_ord'])
      .reset_index(drop=True)
)

# Display an HTML preview
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(HTML(hypotheses.to_html(index=False)))

# Export a LaTeX table
output_path = "../../output/visuals_pipeline_output/tables/hypotheses.tex"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
hypotheses.to_latex(
    output_path,
    index=False,
    longtable=True,
    caption="Response and Adapted Hypotheses by Question and Response (Sampled)",
    label="tab:hypotheses",
    escape=True
)
print(f"Saved as {output_path}")
