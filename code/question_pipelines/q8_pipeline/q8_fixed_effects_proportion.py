import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf

# -----------------------------
# Load data
# -----------------------------
predictions_path = "../../../output/question_pipeline_output/q8_predictions/q8_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

# Select Q8 labels
q8_labels = ['Q8_1', 'Q8_2']
df_q8 = df[df['predicted_combined_label'].isin(q8_labels)]

# -----------------------------
# Aggregate counts per country-year-label
# -----------------------------
agg = df_q8.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='sentence_count')

# Total sentences per country-year (all Q8)
total_counts = agg.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['sentence_count'].sum().reset_index(name='total_q8_count')

agg = agg.merge(total_counts, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# Compute proportions
agg['proportion'] = agg['sentence_count'] / agg['total_q8_count']

# -----------------------------
# Filter for Q8_1 only
# -----------------------------
agg_q8_1 = agg[agg['predicted_combined_label'] == 'Q8_1'].copy()

# -----------------------------
# Fit OLS with country and year fixed effects
# -----------------------------
model = smf.ols('proportion ~ C(B_COUNTRY_ALPHA) + C(A_YEAR)', data=agg_q8_1).fit()

# Add fitted values and residuals
agg_q8_1['fitted_proportion'] = model.fittedvalues
agg_q8_1['residuals'] = model.resid

# -----------------------------
# Create combined country-year string for x-axis
# -----------------------------
agg_q8_1['COUNTRY_YEAR'] = agg_q8_1['B_COUNTRY_ALPHA'] + ' ' + agg_q8_1['A_YEAR'].astype(str)

agg_q8_1['COUNTRY_YEAR'] = pd.Categorical(
    agg_q8_1['COUNTRY_YEAR'],
    categories=sorted(agg_q8_1['COUNTRY_YEAR'].unique()),
    ordered=True
)

# -----------------------------
# Plot adjusted (fitted) proportions
# -----------------------------
fig = px.scatter(
    agg_q8_1,
    x='COUNTRY_YEAR',
    y='fitted_proportion',
    size='sentence_count',
    color='B_COUNTRY_ALPHA',
    hover_name='COUNTRY_YEAR',
    labels={
        'COUNTRY_YEAR': 'Country Year',
        'fitted_proportion': 'Adjusted Proportion of Q8_1',
        'sentence_count': 'Sentence Count'
    },
    title='Adjusted Proportion of Q8_1 Sentences (Year & Country Fixed Effects)',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg_q8_1['COUNTRY_YEAR'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40),
    xaxis_tickangle=45
)

fig.show()
