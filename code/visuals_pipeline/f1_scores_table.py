import pandas as pd
import json
from IPython.display import display, HTML
import re
import os

def load_val_f1(path):
    """
    Load per-class F1 scores from a validation report JSON file.

    The function supports two formats:
    (1) A JSON with a 'classification_report' list containing class entries with F1 scores.
    (2) A JSON dictionary keyed by class name, where each value contains F1 score fields.

    Returns
    -------
    dict
        A dictionary mapping class names to their F1 scores (float or None). Missing or
        unavailable scores are stored as None. Returns an empty dict on failure.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        f1_scores = {}
        # Validation reports may have 'classification_report' list OR dict keyed by class names
        if 'classification_report' in data:
            for entry in data['classification_report']:
                cls = entry.get('class') or entry.get('Class')
                f1 = entry.get('f1_score') or entry.get('f1-score')
                if cls is not None:
                    f1_scores[cls] = f1 if f1 is not None else None
        else:
            # fallback: dict keyed by class names
            for class_key, metrics in data.items():
                if not isinstance(metrics, dict):
                    continue
                f1 = metrics.get('f1_score') or metrics.get('f1-score')
                f1_scores[class_key] = f1 if f1 is not None else None
        return f1_scores
    except Exception as e:
        print(f"Error loading validation F1 scores from {path}: {e}")
        return {}

def load_perm_f1(path):
    """
    Load per-class F1 scores from a permutation test JSON report.

    The function reads F1 values from either the 'classification_report' or
    'classification_report_per_class' list, depending on which key is present
    in the input file. It extracts the 'f1_score' (or 'f1-score') field for
    each class and returns a mapping of class labels to their scores.

    Parameters
    ----------
    path : str
        Path to the permutation test classification report JSON file.

    Returns
    -------
    dict
        A dictionary mapping class names to their F1 scores (float or None).
        Returns an empty dictionary on failure or if no valid report key exists.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        f1_scores = {}
        # Permutation test reports may have 'classification_report' or 'classification_report_per_class'
        report_key = None
        if 'classification_report' in data:
            report_key = 'classification_report'
        elif 'classification_report_per_class' in data:
            report_key = 'classification_report_per_class'
        else:
            print(f"Warning: No classification report key found in {path}")

        if report_key:
            for entry in data[report_key]:
                cls = entry.get('class') or entry.get('Class')
                f1 = entry.get('f1_score') or entry.get('f1-score')
                if cls is not None:
                    f1_scores[cls] = f1 if f1 is not None else None
        return f1_scores
    except Exception as e:
        print(f"Error loading permutation F1 scores from {path}: {e}")
        return {}

# Extract the most recent n_permutations assignment from a training script
def read_last_n_permutations_from_script(path):
    """
    Extract the last assigned value of `n_permutations` from a Python training script.

    The function scans the file line by line, ignores commented-out code, and returns
    the most recent integer assigned to `n_permutations`. Returns None if no valid
    assignment is found or if the file cannot be read.

    Parameters
    ----------
    path : str
        Path to the Python training script.

    Returns
    -------
    int or None
        The last integer value assigned to `n_permutations`, or None if unavailable.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            last_val = None
            for raw_line in f:
                # strip inline comments
                line = raw_line.split('#', 1)[0]
                m = re.search(r'\bn_permutations\s*=\s*(\d+)\b', line)
                if m:
                    last_val = int(m.group(1))
        return last_val
    except Exception as e:
        print(f"Error reading n_permutations from {path}: {e}")
        return None

# Expand combined question keys (e.g., "Q152, Q153") so each maps to the same file path.
def expand_question_to_path(question_to_path):
    """
    Expand comma-separated question keys so that each individual question maps to the same path.

    Parameters
    ----------
    question_to_path : dict
        A dictionary where keys may contain one or more question labels separated by commas,
        and values are corresponding file paths.

    Returns
    -------
    dict
        A dictionary in which each question label appears as an individual key mapped to its path.
    """
    expanded = {}
    for key, path in question_to_path.items():
        parts = [p.strip().upper() for p in key.split(',')]
        for p in parts:
            expanded[p] = path
    return expanded

def extract_f1_table(question_keys, paths_dict, training_paths):
    """
    Construct a per-class F1 summary across questions, combining validation and label-permutation results.

    For each question, loads per-class F1 from the validation report and two permutation-test reports
    (original and augmented), aligns over the union of class labels, optionally retrieves the most recent
    `n_permutations` from the associated training script, and assembles a tidy table suitable for HTML
    display and LaTeX export.

    Parameters
    ----------
    question_keys : list of str
        Question identifiers (e.g., ['q8','q11', ...]) expected as keys in `paths_dict`.
    paths_dict : dict
        Mapping from question key to JSON report paths with keys:
        'val_classification_report', 'permutation_test_report_original',
        and 'permutation_test_report_augmented'.
    training_paths : dict
        Mapping from question labels (e.g., 'Q8' or 'Q152, Q153') to training script file paths.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['Question', 'Response', 'Validation Set F1',
         'Permutation Test F1 (Original Dataset)',
         'Permutation Test F1 (Augmented Dataset)', 'Permutations (N)'].
    """
    # Normalize combined labels (e.g., "Q152, Q153") so each label maps to the same script path.
    expanded_train_map = expand_question_to_path(training_paths)

    records = []

    for q_key in question_keys:
        q_key_upper = q_key.upper()  # normalize for lookups like 'Q8'
        
        # Load per-class F1 scores from validation and permutation reports for this question.
        val_scores = load_val_f1(paths_dict[q_key]['val_classification_report'])
        perm_orig_scores = load_perm_f1(paths_dict[q_key]['permutation_test_report_original'])
        perm_aug_scores = load_perm_f1(paths_dict[q_key]['permutation_test_report_augmented'])

        # Optionally retrieve the most recent n_permutations from the associated training script.
        n_perm = None
        train_script_path = expanded_train_map.get(q_key_upper)
        if train_script_path:
            n_perm = read_last_n_permutations_from_script(train_script_path)

        # Align over the union of class labels so all sources contribute rows consistently.
        all_classes = sorted(set(val_scores) | set(perm_orig_scores) | set(perm_aug_scores))

        for cls in all_classes:
            label_num = cls.split('_')[-1] if '_' in cls else cls
            records.append({
                'Question': q_key_upper,
                'Response': label_num,
                'Validation Set F1': val_scores.get(cls, None),
                'Permutation Test F1 (Original Dataset)': perm_orig_scores.get(cls, None),
                'Permutation Test F1 (Augmented Dataset)': perm_aug_scores.get(cls, None),
                'Permutations (N)': n_perm,
            })

    df = pd.DataFrame(records)

    # Cast permutation count to a nullable integer dtype to allow missing values.
    if 'Permutations (N)' in df.columns:
        try:
            df['Permutations (N)'] = pd.array(df['Permutations (N)'], dtype='Int64')
        except Exception:
            pass

    # Center table headers and cells for HTML display.
    css = """
    <style>
        table.dataframe th, table.dataframe td {
            text-align: center !important;
        }
    </style>
    """
    
    styled_html = (
        df.style
        .format({
            'Validation Set F1': '{:.3f}',
            'Permutation Test F1 (Original Dataset)': '{:.3f}',
            'Permutation Test F1 (Augmented Dataset)': '{:.3f}',
        }, na_rep='0.000') # NaN is rendered as 0.000 here by design, reflecting a known case where the sole missing value corresponds to zero.
        .hide(axis='index')
        .to_html()
    )

    display(HTML(css + styled_html))

    # Export a LaTeX longtable with fixed column specification for reproducible manuscript inclusion.
    # Column spec: 'Question','Response' (ll) + three float columns + one integer column -> 'llrrrr'.
    
    output_path = "../../output/visuals_pipeline_output/tables/f1_scores_table.tex"

    df.to_latex(
        output_path,
        index=False,
        longtable=True,
        caption="F1 Scores for Validation and Permutation Tests by Question and Response",
        label="tab:f1_scores_table",
        float_format="%.3f",
        column_format="llrrrr",
        escape=True,
    )

    print(f"LaTeX table saved as {output_path}")
    return df

# Mapping each question to its training script for retrieving n_permutations.
training_path = {
    'Q8': "../question_pipelines/q8_pipeline/train_catboost_labeled.py",
    'Q11': "../question_pipelines/q11_pipeline/train_catboost_labeled.py",
    'Q17': "../question_pipelines/q17_pipeline/train_catboost_labeled.py",
    'Q65': "../question_pipelines/q65_pipeline/train_catboost_labeled.py",
    'Q69': "../question_pipelines/q69_pipeline/train_catboost_labeled.py",
    'Q70': "../question_pipelines/q70_pipeline/train_catboost_labeled.py",
    'Q152, Q153': "../question_pipelines/q152_q153_pipeline/train_catboost_labeled.py",
    'Q154, Q155': "../question_pipelines/q154_q155_pipeline/train_catboost_labeled.py"
}

# Mapping each question to its validation and permutation-test report outputs.
reports = {
    'q8': {
        'val_classification_report': '../question_pipelines/q8_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q8_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q8_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q11': {
        'val_classification_report': '../question_pipelines/q11_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q11_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q11_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q17': {
        'val_classification_report': '../question_pipelines/q17_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q17_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q17_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q65': {
        'val_classification_report': '../question_pipelines/q65_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q65_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q65_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q69': {
        'val_classification_report': '../question_pipelines/q69_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q69_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q69_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q70': {
        'val_classification_report': '../question_pipelines/q70_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q70_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q70_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q152': {
        'val_classification_report': '../question_pipelines/q152_q153_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q152_q153_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q152_q153_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
    'q154': {
        'val_classification_report': '../question_pipelines/q154_q155_pipeline/logs_and_metrics/val_classification_report.json',
        'permutation_test_report_original': '../question_pipelines/q154_q155_pipeline/logs_and_metrics/permutation_test_report_original.json',
        'permutation_test_report_augmented': '../question_pipelines/q154_q155_pipeline/logs_and_metrics/permutation_test_report_augmented.json',
    },
}

# Define the set of questions to be summarized in the final F1 table.
question_list = ['q8', 'q11', 'q17', 'q65', 'q69', 'q70', 'q152', 'q154']

# Generate the consolidated F1 summary table.
df_f1 = extract_f1_table(question_list, reports, training_path)