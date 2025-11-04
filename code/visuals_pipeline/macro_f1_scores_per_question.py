import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import json
import os
import warnings
warnings.filterwarnings("ignore")

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

def load_macro_f1(path):
    """
    Load and return the macro F1 score from a JSON results file.

    Attempts to extract either 'final_macro_f1' or 'average_macro_f1' keys. 
    Returns None and prints a warning or error message if unavailable or invalid.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if "final_macro_f1" in data:
            return data["final_macro_f1"]
        if "average_macro_f1" in data:
            return data["average_macro_f1"]
        print(f"Warning: No macro F1 key found in {path}")
        return None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def plot_macro_f1(question_keys, paths_dict):
    """
    Generate a grouped bar chart comparing macro F1 scores across multiple question datasets.

    This function visualizes the performance consistency between validation and permutation tests 
    (original and augmented) for each question key. It performs the following operations:
        - Loads macro F1 scores from JSON result files using the `load_macro_f1` utility function.
        - Constructs a grouped bar chart with three bars per question (Validation, Permutation Original,
        Permutation Augmented) using Matplotlib.
        - Applies academic-style formatting, including Times New Roman font, boxed x-axis labels, 
        and publication-quality legend positioning.
        - Annotates each bar with its numeric F1 value and adjusts figure layout for presentation clarity.

    Args:
        question_keys (list of str): Ordered list of question identifiers (e.g., ['q8', 'q11', 'q152']).
        paths_dict (dict): Nested dictionary mapping each question key to file paths containing 
            'val_classification_report', 'permutation_test_report_original', 
            and 'permutation_test_report_augmented' JSON files.

    Returns:
        matplotlib.figure.Figure: The generated figure object for display or export.
    """
    labels = ['Validation', 'Permutation Original', 'Permutation Augmented']
    colors = ['steelblue', 'orange', 'green']
    width = 0.25
    spacing = 1.0
    
    academic_font = {'fontname': 'Times New Roman', 'fontweight': 'semibold'}
    
    label_box = dict(boxstyle="square,pad=0.3",
                     facecolor="none",
                     edgecolor="black",
                     linewidth=1)

    fig, ax = plt.subplots(figsize=(len(question_keys) * 2.5, 6))

    for i, q_key in enumerate(question_keys):
        macro_val = load_macro_f1(paths_dict[q_key]['val_classification_report'])
        macro_perm_orig = load_macro_f1(paths_dict[q_key]['permutation_test_report_original'])
        macro_perm_aug = load_macro_f1(paths_dict[q_key]['permutation_test_report_augmented'])

        print(f"{q_key}: val={macro_val}, perm_orig={macro_perm_orig}, perm_aug={macro_perm_aug}")

        scores = [macro_val, macro_perm_orig, macro_perm_aug]
        base_x = i * spacing
        x_positions = [base_x - width, base_x, base_x + width]

        bars = ax.bar(x_positions, scores, width=width, color=colors, label='_nolegend_')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=14,
                        **academic_font)

        if q_key == 'q152':
            display_label = 'Q152, Q153'
        elif q_key == 'q154':
            display_label = 'Q154, Q155'
        else:
            display_label = q_key.upper()

        ax.text(
            base_x, 
            -0.08, 
            display_label, 
            ha='center', 
            va='top', 
            fontsize=13,
            bbox=label_box,
            **academic_font,
            transform=ax.get_xaxis_transform()
        )

    ax.set_ylim(0, 1)
    #ax.set_title('Macro F1 Scores Across Questions', fontsize=16, **academic_font)

    ax.set_yticklabels(
        [f"{x:.1f}" for x in np.linspace(0, 1, 6)],
        fontsize=12,
        color='#444444',
        fontname='Times New Roman',
        fontweight='semibold'
    )

    ax.set_xticks([])

    legend_handles = [Patch(color=colors[i], label=labels[i], 
                            linewidth=0) for i in range(len(labels))]
    legend = ax.legend(handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            fontsize=12,
            frameon=False,
            handlelength=1.5,
            handleheight=1.5,
            ncol=len(labels),
            columnspacing=1.5)

    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontweight('semibold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    return fig

# Run the plotting function with the specified question list and reports dictionary:
question_list = ['q8', 'q11', 'q17', 'q65', 'q69', 'q70', 'q152', 'q154']
fig = plot_macro_f1(question_list, reports) 

# Define output file path
output_path = "../../output/visuals_pipeline_output/figures/macro_f1_scores_per_question.png"

# Ensure the parent directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the figure
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")