import re
import pandas as pd
from IPython.display import display, HTML

def extract_pred_wvs7_hyperparam(filepath, var_name, func_name=None, question_key=None):
    """
    Extract the last occurrence of a specified variable from a Python script, 
    with special formatting for confidence thresholds in binary questions.

    Parameters
    ----------
    filepath : str
        Path to the Python file to parse.
    var_name : str
        Name of the variable to extract.
    func_name : str, optional
        Function name to search for default argument values if direct assignment is absent.
    question_key : str, optional
        Question identifier(s) used to determine binary-specific formatting.

    Returns
    -------
    str or None
        Extracted value as a string, formatted as a percentage where applicable; otherwise None.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Last direct assignment
    assign_pattern = rf'^{re.escape(var_name)}\s*=\s*([^\n]+)'
    assign_matches = re.findall(assign_pattern, text, flags=re.MULTILINE)
    val = assign_matches[-1].strip() if assign_matches else None
    if val and (val.startswith(("'", '"')) and val.endswith(("'", '"'))):
        val = val[1:-1]

    # Search inside function signature if needed
    if func_name and val is None:
        func_pattern = re.compile(rf'def\s+{re.escape(func_name)}\s*\((.*?)\):', re.DOTALL)
        func_match = func_pattern.search(text)
        if func_match:
            params_str = func_match.group(1)
            params = [p.strip() for p in params_str.split(',')]
            for p in params:
                if p.startswith(var_name + '='):
                    val = p.split('=', 1)[1].strip()
                    if (val.startswith(("'", '"')) and val.endswith(("'", '"'))):
                        val = val[1:-1]
                    break

    # Search dict if still None
    if val is None:
        dict_pattern = re.compile(rf'{re.escape(var_name)}\s*:\s*([^,\n]+)')
        dict_matches = dict_pattern.findall(text)
        if dict_matches:
            val = dict_matches[-1].strip()
            if (val.startswith(("'", '"')) and val.endswith(("'", '"'))):
                val = val[1:-1]

    binary_questions = ('Q8', 'Q11', 'Q17')
    if var_name == 'confidence_threshold' and question_key is not None:
        question_keys = [q.strip() for q in question_key.split(',')]
        if any(q in binary_questions for q in question_keys):
            # Extract last occurrence of 'threshold' variable in the script (number)
            threshold_matches = re.findall(r'^\s*threshold\s*=\s*([0-9]*\.?[0-9]+)', text, flags=re.MULTILINE)
            threshold_val = threshold_matches[-1] if threshold_matches else None

            try:
                percentage_num = float(val.strip('%')) if isinstance(val, str) else float(val)
            except:
                percentage_num = val

            if threshold_val is not None:
                # Format with 1-2 decimals for percentage
                perc_str = f"{percentage_num:.2f}".rstrip('0').rstrip('.')
                return f"{perc_str}% ({threshold_val})"
            else:
                perc_str = f"{percentage_num:.2f}".rstrip('0').rstrip('.')
                return f"{perc_str}%"

    # For non-binary confidence_threshold, format as integer % if possible, else float %
    if var_name == 'confidence_threshold':
        try:
            f = float(val)
            if f.is_integer():
                return f"{int(f)}%"
            else:
                # Keep up to 2 decimals
                return f"{f:.2f}%".rstrip('0').rstrip('.')
        except:
            pass

    return val

def float_to_int(x):
    """Convert float ending with .0 to int, else return original."""
    try:
        f = float(x)
        if f.is_integer():
            return int(f)
        return f
    except:
        return x

# Map questions to filepaths (update your actual paths here)
question_to_path = {
    'Q8': "../question_pipelines/q8_pipeline/predict_catboost_unga_wvs7.py",
    'Q11': "../question_pipelines/q11_pipeline/predict_catboost_unga_wvs7.py",
    'Q17': "../question_pipelines/q17_pipeline/predict_catboost_unga_wvs7.py",
    'Q65': "../question_pipelines/q65_pipeline/predict_catboost_unga_wvs7.py",
    'Q69': "../question_pipelines/q69_pipeline/predict_catboost_unga_wvs7.py",
    'Q70': "../question_pipelines/q70_pipeline/predict_catboost_unga_wvs7.py",
    'Q152, Q153': "../question_pipelines/q152_q153_pipeline/predict_catboost_unga_wvs7.py",
    'Q154, Q155': "../question_pipelines/q154_q155_pipeline/predict_catboost_unga_wvs7.py"
}

# Map questions to response types
response_type_map = {
    'Q8': 'Binary',
    'Q11': 'Binary',
    'Q17': 'Binary',
    'Q152, Q153': 'Categorical',
    'Q154, Q155': 'Categorical',
    'Q65': 'Ordinal',
    'Q69': 'Ordinal',
    'Q70': 'Ordinal',
}

# Extract hyperparameters and compile into DataFrame
rows = []

for question, filepath in question_to_path.items():
    max_phrases = extract_pred_wvs7_hyperparam(filepath, 'MAX_PHRASES')
    ngram_range = extract_pred_wvs7_hyperparam(filepath, 'KEYPHRASE_NGRAM_RANGE')
    diversity = extract_pred_wvs7_hyperparam(filepath, 'DIVERSITY')

    confidence_col = extract_pred_wvs7_hyperparam(filepath, 'confidence_col', func_name='joint_score_filter')
    confidence_weight = extract_pred_wvs7_hyperparam(filepath, 'confidence_weight', func_name='joint_score_filter')
    similarity_weight = extract_pred_wvs7_hyperparam(filepath, 'similarity_weight', func_name='joint_score_filter')

    confidence_threshold = extract_pred_wvs7_hyperparam(filepath, 'confidence_threshold', func_name='joint_score_filter', question_key=question)
    similarity_threshold = extract_pred_wvs7_hyperparam(filepath, 'similarity_threshold', func_name='joint_score_filter')
    joint_threshold = extract_pred_wvs7_hyperparam(filepath, 'joint_threshold', func_name='joint_score_filter')

    # Map confidence_col to readable string
    if confidence_col in ('perc_diff_chance_abs', "'perc_diff_chance_abs'"):
        confidence_col_readable = "% Difference from Chance"
    elif confidence_col in ('perc_above_chance', "'perc_above_chance'"):
        confidence_col_readable = "% Above Chance"
    else:
        confidence_col_readable = confidence_col

    response_type = response_type_map.get(question, 'Unknown')

    rows.append({
        'Question': question,
        'Response Type': response_type,
        'Maximum Extracted Phrases': float_to_int(max_phrases),
        'Ngram Range Extracted Phrases': ngram_range,
        'Diversity Metric for Extracted Phrases': float_to_int(diversity),
        'Confidence Metric Criterion': confidence_col_readable,
        'Minimum Confidence Metric Threshold (Decision Boundary)': confidence_threshold,
        'Confidence Metric Weight': float_to_int(confidence_weight),
        'Keyword Cosine Similarity Metric Weight': float_to_int(similarity_weight),
        'Minimum Cosine Similarity Metric Threshold': float_to_int(similarity_threshold),
        'Minimum Joint Metric Threshold': float_to_int(joint_threshold),
    })

df = pd.DataFrame(rows)

# Display nicely centered in Jupyter Notebook
html = df.to_html(index=False)
css = """
<style>
    table.dataframe th, table.dataframe td {
        text-align: center !important;
    }
</style>
"""
display(HTML(css + html))

output_path = "../../output/visuals_pipeline_output/tables/pred_wvs7_hyperparams_summary.tex"

# Save as LaTeX
col_format = 'll' + 'c' * (len(df.columns) - 2)
df.to_latex(
    output_path,
    index=False,
    caption="Extracted Prediction Hyperparameters for by Model (Full Dataset, N > 25,500)",
    label="tab:pred_wvs7_hyperparams_summary",
    column_format=col_format,
    escape=True,
)

print("LaTeX table saved as pred_wvs7_hyperparams_summary.tex")
