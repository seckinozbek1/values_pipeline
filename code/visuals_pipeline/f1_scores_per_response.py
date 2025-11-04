import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import json
import seaborn as sns
import os

sns.set_style('whitegrid')

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
def load_val_f1_scores(path):
    """
    Load per-class F1 scores from a validation report JSON file.

    The function supports two formats:
    (1) A JSON with a 'classification_report' list containing class entries with F1 scores.
    (2) A JSON dictionary keyed by class name, where each value contains F1 score fields.

    Returns
    -------
    dict
        A dictionary mapping class names to their F1 scores (float). Missing or
        unavailable scores are set to `np.nan`. Returns an empty dict on failure.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        f1_scores = {}
        # Prefer 'classification_report' array format if exists
        if 'classification_report' in data:
            for entry in data['classification_report']:
                cls = entry.get('class') or entry.get('Class')
                f1 = entry.get('f1_score') or entry.get('f1-score')
                if cls is not None:
                    f1_scores[cls] = f1 if f1 is not None else np.nan
        else:
            # Else expect dict with keys as class names
            for class_key, metrics in data.items():
                if not isinstance(metrics, dict):
                    continue
                f1 = metrics.get("f1_score") or metrics.get("f1-score")
                if f1 is None:
                    f1 = np.nan
                f1_scores[class_key] = f1
        return f1_scores
    except Exception as e:
        print(f"Error loading val f1 scores from {path}: {e}")
        return {}

def load_perm_f1_scores(path):
    """
    Load per-class F1 scores from a permutation test JSON report.

    The function reads F1 scores from either the 'classification_report' or
    'classification_report_per_class' list in the JSON file.

    Parameters
    ----------
    path : str
        Path to the permutation report JSON file.

    Returns
    -------
    dict
        A dictionary mapping class names to their F1 scores (float). Missing or
        unavailable scores are set to `np.nan`. Returns an empty dict on failure.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        f1_scores = {}
        # Permutation test reports may use 'classification_report' or 'classification_report_per_class'
        report_key = None
        if 'classification_report' in data:
            report_key = 'classification_report'
        elif 'classification_report_per_class' in data:
            report_key = 'classification_report_per_class'

        if report_key:
            for entry in data[report_key]:
                cls = entry.get('class') or entry.get('Class')
                f1 = entry.get('f1_score') or entry.get('f1-score')
                if cls is not None:
                    f1_scores[cls] = f1 if f1 is not None else np.nan
        else:
            print(f"Warning: No classification report found in {path}")
        return f1_scores
    except Exception as e:
        print(f"Error loading perm f1 scores from {path}: {e}")
        return {}

def plot_f1_class_multi(question_keys, paths_dict):
    """
    Plot per-class F1 scores for multiple questions using grouped bar charts.

    This function loads class-level F1 scores from three sources 
    (Validation, Permutation Original, and Permutation Augmented) and 
    draws side-by-side bar groups for each class, arranged across multiple
    subplots. Panels are laid out in a grid (maximum of two plots per row),
    and each subplot corresponds to a single question.

    Parameters
    ----------
    question_keys : list of str
        A list of question identifiers (e.g., ["q8", "q152", "q154"]). Each
        key must exist in `paths_dict`.
    paths_dict : dict
        A dictionary where each key is a question and each value is another
        dictionary containing file paths for:
        - 'val_classification_report'
        - 'permutation_test_report_original'
        - 'permutation_test_report_augmented'

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object containing all subplots.

    Notes
    -----
    - Uses Times New Roman font for academic formatting.
    - Only the first column of subplots displays Y-axis tick labels.
    - Legend is shared across all panels and placed at the bottom.
    - Missing class scores are represented as NaN and result in empty bars.
    """
    labels = ['Validation', 'Permutation Original', 'Permutation Augmented']
    colors = ["#0F346F", "#52DDB3", "#A85555"]

    width = 0.8
    bar_spacing = width * 5
    max_per_row = 2

    num_rows = (len(question_keys) + max_per_row - 1) // max_per_row
    num_cols = min(len(question_keys), max_per_row)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(num_cols * 5, num_rows * 5),
                             squeeze=False)

    academic_font = {'fontname': 'Times New Roman'}

    # Global super-title (suptitle) removed intentionally: it caused layout conflicts with tight_layout() 
    # and crowded the figure, so it has been commented out for cleaner aesthetics.
    #fig.suptitle('Per-Class F1 Scores Across Questions',
     #            fontsize=18, fontweight='semibold',
      #           color='#111111', y=0.95, **academic_font)

    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_axis_off()

    label_box = dict(boxstyle="square,pad=0.3",
                     facecolor="none",
                     edgecolor="#333333",
                     linewidth=1)

    for i, q_key in enumerate(question_keys):
        ax = axes_flat[i]
        ax.set_axis_on()

        val_scores = load_val_f1_scores(paths_dict[q_key]['val_classification_report'])
        perm_orig_scores = load_perm_f1_scores(paths_dict[q_key]['permutation_test_report_original'])
        perm_aug_scores = load_perm_f1_scores(paths_dict[q_key]['permutation_test_report_augmented'])

        # Get union of all class keys from the three reports for alignment
        all_classes = sorted(set(val_scores) | set(perm_orig_scores) | set(perm_aug_scores))

        def align_scores(score_dict, keys):
            return [score_dict.get(k, np.nan) for k in keys]

        val_aligned = align_scores(val_scores, all_classes)
        perm_orig_aligned = align_scores(perm_orig_scores, all_classes)
        perm_aug_aligned = align_scores(perm_aug_scores, all_classes)

        group_start = 0
        for j, cls in enumerate(all_classes):
            base_x = group_start + j * bar_spacing
            scores = [val_aligned[j], perm_orig_aligned[j], perm_aug_aligned[j]]
            x_pos = [base_x - width, base_x, base_x + width]

            ax.bar(x_pos, scores, width=width, color=colors, label='_nolegend_')

            # label with last part after underscore if exists, else full
            label_num = cls.split('_')[-1] if '_' in cls else cls
            ax.text(base_x, -0.07, label_num,
                    ha='center', va='top', fontweight='semibold', fontsize=15,
                    transform=ax.get_xaxis_transform(),
                    color='#222222', **academic_font)

        # Question label (special cases for combined)
        if q_key.lower() == 'q152':
            label_text = "Q152, Q153"
        elif q_key.lower() == 'q154':
            label_text = "Q154, Q155"
        else:
            label_text = q_key.upper()

        ax.text((len(all_classes) - 1) * bar_spacing / 2, -0.23,
                label_text, fontweight='semibold',
                ha='center', va='top',
                fontsize=15, color='#111111',
                bbox=label_box, **academic_font)

        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.linspace(0, 1, 6))

        if i % max_per_row == 0:
            ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)],
                               fontsize=12, color='#444444', **academic_font)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='x', length=0)
        ax.set_xticks([])

    legend_handles = [Patch(color=colors[k], label=labels[k], linewidth=0)
                      for k in range(len(labels))]

    legend = fig.legend(handles=legend_handles,
                        loc='lower center', bbox_to_anchor=(0.5, -0.04),
                        ncol=len(labels),
                        fontsize=12, frameon=False,
                        handlelength=1.5, handleheight=1.5)

    for text in legend.get_texts():
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.35)
    plt.show()
    return fig

# Usage
question_list = ['q8', 'q11', 'q17', 'q65', 'q69', 'q70', 'q152', 'q154']
fig = plot_f1_class_multi(question_list, reports)

# Define output file path
output_path = "../../output/visuals_pipeline_output/figures/f1_scores_per_response.png"

# Ensure the parent directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the figure
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")