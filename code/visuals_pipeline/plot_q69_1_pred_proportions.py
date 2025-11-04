import pandas as pd
import plotly.express as px

# Load predictions from the requested path
df = pd.read_csv("../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv")

# Filter to Q69 labels
q69_labels = ['Q69_1', 'Q69_2', 'Q69_3', 'Q69_4']
df_q69 = df[df['predicted_combined_label'].isin(q69_labels)]

# Aggregate counts by country and label
agg = (
    df_q69.groupby(['B_COUNTRY_ALPHA', 'predicted_combined_label'])
           .size()
           .reset_index(name='sentence_count')
)

# Total Q69 sentences per country for proportional scaling
total_counts = agg.groupby('B_COUNTRY_ALPHA')['sentence_count'].sum().reset_index(name='total_q69_count')
agg = agg.merge(total_counts, on='B_COUNTRY_ALPHA')
agg['proportion'] = agg['sentence_count'] / agg['total_q69_count']

# Subset to Q69_1
agg_q69_1 = agg[agg['predicted_combined_label'] == 'Q69_1'].copy()

# Order countries alphabetically for stable categorical axis rendering
ordered_countries = sorted(agg_q69_1['B_COUNTRY_ALPHA'].unique())
agg_q69_1['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg_q69_1['B_COUNTRY_ALPHA'],
    categories=ordered_countries,
    ordered=True
)

# Scatter plot; no legend; no plot title; no y-axis title; 45° x-tick labels
fig = px.scatter(
    agg_q69_1,
    x='B_COUNTRY_ALPHA',
    y='proportion',
    size='sentence_count',
    color='B_COUNTRY_ALPHA',  
    hover_name='B_COUNTRY_ALPHA',
    labels={'B_COUNTRY_ALPHA': 'Country', 'proportion': '', 'sentence_count': 'Sentence Count'},
    size_max=40,
    height=600
)

# Global typography set to Times New Roman Bold (falls back to Times New Roman if Bold variant unavailable)
bold_times = "Times New Roman Bold, Times New Roman"

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': ordered_countries},
    xaxis_tickangle=45,      # 45° tilted country names
    yaxis_title=None,        # no y-axis title
    showlegend=False,        # remove legend
    title=None,              # no plot title
    font=dict(family=bold_times, size=14),  # global font
    hoverlabel=dict(font_family=bold_times, font_size=12),
    margin=dict(l=40, r=40, t=20, b=40)
)

# Ensure axis tick labels also use the bold Times family explicitly
fig.update_xaxes(tickfont=dict(family=bold_times, size=12), title=None)
fig.update_yaxes(tickfont=dict(family=bold_times, size=12), title=None)

# Save plot as high-resolution PNG
fig.write_image("../../output/visuals_pipeline_output/figures/q69_1_pred_proportions.png", scale=3)
print("Saved -> ../../output/visuals_pipeline_output/figures/q69_1_pred_proportions.png")

fig.show()