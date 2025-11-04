import re
import pandas as pd
from IPython.display import display, HTML

def extract_training_hyperparam(filepath, var_name, func_name=None, dict_name=None):
    """
    Extract the last occurrence of a hyperparameter value from a Python script.

    Parameters
    ----------
    filepath : str
        Path to the Python file to parse.
    var_name : str
        Name of the variable or key to extract.
    func_name : str, optional
        Function name to search for keyword arguments (e.g., func_name(..., var_name=...)).
    dict_name : str, optional
        Dictionary name to search for key-value pairs (e.g., dict_name = {'var_name': ...}).

    Returns
    -------
    float, int, str, or None
        The extracted value if found; otherwise None.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    num_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    str_pattern = r'["\']([^"\']+)["\']'

    # Try direct variable assignment (number)
    assign_pattern_num = rf'^{var_name}\s*=\s*({num_pattern})'
    assign_matches_num = re.findall(assign_pattern_num, text, flags=re.MULTILINE)
    if assign_matches_num:
        try:
            return float(assign_matches_num[-1])
        except ValueError:
            return assign_matches_num[-1]

    # Try direct variable assignment (string)
    assign_pattern_str = rf'^{var_name}\s*=\s*{str_pattern}'
    assign_matches_str = re.findall(assign_pattern_str, text, flags=re.MULTILINE)
    if assign_matches_str:
        return assign_matches_str[-1]

    # Try dict literal if dict_name provided (number)
    if dict_name:
        dict_pattern_num = rf'{dict_name}\s*=\s*\{{.*?["\']{var_name}["\']\s*:\s*({num_pattern}).*?\}}'
        dict_matches_num = re.findall(dict_pattern_num, text, flags=re.DOTALL)
        if dict_matches_num:
            try:
                return float(dict_matches_num[-1])
            except ValueError:
                return dict_matches_num[-1]

        # Try dict literal (string)
        dict_pattern_str = rf'{dict_name}\s*=\s*\{{.*?["\']{var_name}["\']\s*:\s*{str_pattern}.*?\}}'
        dict_matches_str = re.findall(dict_pattern_str, text, flags=re.DOTALL)
        if dict_matches_str:
            return dict_matches_str[-1]

    # Try function call argument if func_name provided (number)
    if func_name:
        func_pattern_num = rf'{func_name}\s*\(.*?{var_name}\s*=\s*({num_pattern}).*?\)'
        func_matches_num = re.findall(func_pattern_num, text, flags=re.DOTALL)
        if func_matches_num:
            try:
                return float(func_matches_num[-1])
            except ValueError:
                return func_matches_num[-1]

        # Try function call argument (string)
        func_pattern_str = rf'{func_name}\s*\(.*?{var_name}\s*=\s*{str_pattern}.*?\)'
        func_matches_str = re.findall(func_pattern_str, text, flags=re.DOTALL)
        if func_matches_str:
            return func_matches_str[-1]

    print(f"Warning: {var_name} not found in {filepath}")
    return None

def convert_to_int(df, col):
    """
    Convert a numeric column to nullable integer type if all non-missing values are integers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the target column.
    col : str
        Column name to evaluate and convert.

    Notes
    -----
    Operates in place and preserves missing values using the 'Int64' dtype.
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        non_na = df[col].dropna()
        if not non_na.empty and non_na.apply(float.is_integer).all():
            df[col] = df[col].astype('Int64')  


# Response type map as requested
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

# Map your questions to their script paths
question_to_path = {
    'Q8': "../question_pipelines/q8_pipeline/train_catboost_labeled.py",
    'Q11': "../question_pipelines/q11_pipeline/train_catboost_labeled.py",
    'Q17': "../question_pipelines/q17_pipeline/train_catboost_labeled.py",
    'Q65': "../question_pipelines/q65_pipeline/train_catboost_labeled.py",
    'Q69': "../question_pipelines/q69_pipeline/train_catboost_labeled.py",
    'Q70': "../question_pipelines/q70_pipeline/train_catboost_labeled.py",
    'Q152, Q153': "../question_pipelines/q152_q153_pipeline/train_catboost_labeled.py",
    'Q154, Q155': "../question_pipelines/q154_q155_pipeline/train_catboost_labeled.py"
}


rows = []
for question, filepath in question_to_path.items():
    num_augmentations = extract_training_hyperparam(filepath, 'num_augmentations')
    num_replacements = extract_training_hyperparam(filepath, 'num_replacements', func_name='synonym_augment')
    val_size = extract_training_hyperparam(filepath, 'test_size', func_name='train_test_split')
    iterations = extract_training_hyperparam(filepath, 'iterations', dict_name='catboost_params')
    learning_rate = extract_training_hyperparam(filepath, 'learning_rate', dict_name='catboost_params')
    depth = extract_training_hyperparam(filepath, 'depth', dict_name='catboost_params')

    loss_function = extract_training_hyperparam(filepath, 'loss_function', dict_name='catboost_params')
    eval_metric = extract_training_hyperparam(filepath, 'eval_metric', dict_name='catboost_params')

    # Additional hyperparameters extraction
    l2_leaf_reg = extract_training_hyperparam(filepath, 'l2_leaf_reg', dict_name='catboost_params')
    random_strength = extract_training_hyperparam(filepath, 'random_strength', dict_name='catboost_params')
    bagging_temperature = extract_training_hyperparam(filepath, 'bagging_temperature', dict_name='catboost_params')

    response_type = response_type_map.get(question, 'Unknown')

    rows.append({
        'Question': question,
        'Response Type': response_type,
        'Sentence Augmentations': num_augmentations,
        'Word Replacements': num_replacements,
        'Validation Size': val_size,
        'Iterations': iterations,
        'Learning Rate': learning_rate,
        'Depth': depth,
        'L2 Leaf Regularization': l2_leaf_reg,
        'Random Strength': random_strength,
        'Bagging Temperature': bagging_temperature,
        'Loss Function': loss_function,
        'Evaluation Metric': eval_metric,
    })

df = pd.DataFrame(rows)

# Apply integer conversion only if all values are whole numbers
for col in ['Sentence Augmentations', 'Word Replacements', 'Validation Size', 'Iterations', 'Learning Rate', 'Depth',
            'L2 Leaf Regularization', 'Random Strength', 'Bagging Temperature']:
    if col in df.columns:
        convert_to_int(df, col)

# Display centered HTML table in notebook
html = df.to_html(index=False)
css = """
<style>
    table.dataframe th, table.dataframe td {
        text-align: center !important;
    }
</style>
"""
display(HTML(css + html))

# Save as LaTeX with centered columns
col_format = 'llcccccccccclll'  # Adjusted for new columns

output_path = "../../output/visuals_pipeline_output/tables/training_hyperparams_summary.tex"

df.to_latex(
    output_path,
    index=False,
    caption="Training Hyperparameters by Model",
    label="tab:training_hyperparams_summary",
    column_format=col_format,
    escape=True,
)

print("LaTeX table saved as training_hyperparams_summary.tex")
