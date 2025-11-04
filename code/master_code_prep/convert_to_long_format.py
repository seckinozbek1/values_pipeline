import pandas as pd
from IPython.display import display

# Transform wide-format adapted hypotheses into a grouped long-format DataFrame
def melt_group_adapted_hypos(df, adapted_prefix="adapted_hypothesis", group_by_col='combined_label'):
    """
    Convert a wide DataFrame of adapted hypotheses into long format by melting all
    columns starting with the given prefix, grouping by the specified column, and
    concatenating sentences with '[SEP]' separators.
    """
    adapted_cols = [col for col in df.columns if col.startswith(adapted_prefix)]

    df_long = df.melt(
        id_vars=[col for col in df.columns if col not in adapted_cols],
        value_vars=adapted_cols,
        var_name='adapted_hypothesis_num',
        value_name='adapted_hypothesis_sentence'
    )

    df_long = df_long.drop(columns=['adapted_hypothesis_num'])
    df_long = df_long.dropna(subset=['adapted_hypothesis_sentence'])
    df_long = df_long[df_long['adapted_hypothesis_sentence'].str.strip() != '']

    other_cols = [col for col in df_long.columns if col not in ['adapted_hypothesis_sentence', group_by_col]]

    df_grouped = df_long.groupby(group_by_col).agg(
        adapted_hypotheses=('adapted_hypothesis_sentence', lambda x: ' [SEP] '.join(x.tolist())),
        **{col: (col, 'first') for col in other_cols}
    ).reset_index()

    # Reorder columns so adapted_hypotheses is right after response_hypothesis
    cols = list(df_grouped.columns)
    if 'response_hypothesis' in cols and 'adapted_hypotheses' in cols:
        cols.remove('adapted_hypotheses')
        response_idx = cols.index('response_hypothesis')
        cols.insert(response_idx + 1, 'adapted_hypotheses')
        df_grouped = df_grouped[cols]

    return df_grouped


if __name__ == "__main__":
   
   
    input_path = "../../output/master_code_prep_output/main_data_complete.csv"
    df_wide = pd.read_csv(input_path)
    df_grouped = melt_group_adapted_hypos(df_wide, adapted_prefix="adapted_hypothesis", group_by_col='combined_label')

    display(df_grouped.head())
    output_path = "../../output/master_code_prep_output/main_data_complete_long.csv"
    df_grouped.to_csv(output_path, index=False)
    print(f"Saved grouped long-format dataframe to {output_path}")
