import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf

# -----------------------------
# Load data
# -----------------------------
predictions_path = "../../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

# Select Q69 labels
q69_labels = ['Q69_1', 'Q69_2', 'Q69_3', 'Q69_4']
df_q69 = df[df['predicted_combined_label'].isin(q69_labels)]

# -----------------------------
# Aggregate counts per country-year-label
# -----------------------------
agg = df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='sentence_count')

# Total sentences per country-year (all Q69)
total_counts = agg.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['sentence_count'].sum().reset_index(name='total_q69_count')

agg = agg.merge(total_counts, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# Compute proportions
agg['proportion'] = agg['sentence_count'] / agg['total_q69_count']

# -----------------------------
# Filter for Q69_1 only
# -----------------------------
agg_q69_1 = agg[agg['predicted_combined_label'] == 'Q69_1'].copy()

# -----------------------------
# Fit OLS with country and year fixed effects
# -----------------------------
model = smf.ols('proportion ~ C(B_COUNTRY_ALPHA) + C(A_YEAR)', data=agg_q69_1).fit()

# Add fitted values and residuals
agg_q69_1['fitted_proportion'] = model.fittedvalues
agg_q69_1['residuals'] = model.resid

# -----------------------------
# Create combined country-year string for x-axis
# -----------------------------
agg_q69_1['COUNTRY_YEAR'] = agg_q69_1['B_COUNTRY_ALPHA'] + ' ' + agg_q69_1['A_YEAR'].astype(str)

agg_q69_1['COUNTRY_YEAR'] = pd.Categorical(
    agg_q69_1['COUNTRY_YEAR'],
    categories=sorted(agg_q69_1['COUNTRY_YEAR'].unique()),
    ordered=True
)

# -----------------------------
# Plot adjusted (fitted) proportions
# -----------------------------
fig = px.scatter(
    agg_q69_1,
    x='COUNTRY_YEAR',
    y='fitted_proportion',
    size='sentence_count',
    color='B_COUNTRY_ALPHA',
    hover_name='COUNTRY_YEAR',
    labels={
        'COUNTRY_YEAR': 'Country Year',
        'fitted_proportion': 'Adjusted Proportion of Q69_1',
        'sentence_count': 'Sentence Count'
    },
    title='Adjusted Proportion of Q69_1 Sentences (Year & Country Fixed Effects)',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg_q69_1['COUNTRY_YEAR'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40),
    xaxis_tickangle=45
)

fig.show()
