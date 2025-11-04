import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load predictions and restrict to the four Q152 response categories
df = pd.read_csv("../../output/question_pipeline_output/q152_q153_predictions/q152_q153_predictions_filtered.csv")
labels = ['Q152_1','Q152_2','Q152_3','Q152_4']

# Human-readable titles for subplots
label_map = {
    'Q152_1': '1 - Economic Growth',
    'Q152_2': '2 - Strong Defence Forces',
    'Q152_3': '3 - Participation in Governance',
    'Q152_4': '4 - Environment'
}
subplot_titles = [label_map[x] for x in labels]

# Filter to selected labels only
d = df[df['predicted_combined_label'].isin(labels)]

# Compute per-country counts and normalized proportions
g = (d.groupby(['B_COUNTRY_ALPHA','predicted_combined_label'])
       .size().reset_index(name='n'))
tot = g.groupby('B_COUNTRY_ALPHA', as_index=False)['n'].sum().rename(columns={'n':'tot'})
g = g.merge(tot, on='B_COUNTRY_ALPHA')
g['prop'] = g['n'] / g['tot']  # within-country proportion

# Select top-K countries per label by highest share
K = 15
tops = {lab: (g[g['predicted_combined_label']==lab]
                .sort_values('prop', ascending=False)
                .head(K).copy())
        for lab in labels}

# Create 2×2 subplot layout
fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                    horizontal_spacing=0.08, vertical_spacing=0.06,
                    subplot_titles=subplot_titles)

# Font family prioritizing a bold Times variant
bold_times = "Times New Roman Bold, Times New Roman, Times, serif"

# Placement mapping for subplots
pos = {0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)}

# Add horizontal bar charts to each facet
for idx, lab in enumerate(labels):
    r, c = pos[idx]
    df_lab = tops[lab].sort_values('prop', ascending=False)
    y_order = df_lab['B_COUNTRY_ALPHA'].tolist()

    fig.add_trace(
        go.Bar(
            x=df_lab['prop'],
            y=df_lab['B_COUNTRY_ALPHA'],
            orientation='h',
            text=(df_lab['prop']*100).round(1).astype(str) + '%',  # display % labels
            textposition='outside',
            textfont=dict(family=bold_times, size=12),
            marker_line_width=0,
            showlegend=False,
            hovertemplate=('Response: ' + label_map[lab] +
                           '<br>Country: %{y}<br>Share: %{x:.2%}<extra></extra>')
        ),
        row=r, col=c
    )

    # Ensure categorical order matches highest-to-lowest ranking
    fig.update_yaxes(categoryorder='array', categoryarray=y_order,
                     autorange='reversed',
                     tickfont=dict(family=bold_times, size=12),
                     title=None, row=r, col=c)

    # Fix x-axis as 0 to 100% scale
    fig.update_xaxes(range=[0,1], tickformat='.0%',
                     tickfont=dict(family=bold_times, size=12),
                     title=None, row=r, col=c)

# Global layout adjustments
fig.update_layout(
    font=dict(family=bold_times, size=14),
    hoverlabel=dict(font_family=bold_times, font_size=12),
    title=None,
    showlegend=False,
    bargap=0.25,
    margin=dict(l=20, r=20, t=28, b=20),
    height=950
)

# Bold facet titles explicitly
for ann in fig.layout.annotations:
    ann.text = f"<b>{ann.text}</b>"
    ann.font.family = bold_times
    ann.font.size = 13
    
#Save
fig.write_image(
    "../../output/visuals_pipeline_output/figures/top_countries_per_response_q152.png",
    scale=3
)
print("Saved -> ../../output/visuals_pipeline_output/figures/top_countries_per_response_q152.png")

fig.show()