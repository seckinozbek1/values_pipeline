import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# Load data
predictions_path = "../../../output/question_pipeline_output/q8_predictions/q8_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

q8_labels = ['Q8_1', 'Q8_2']
df_q8 = df[df['predicted_combined_label'].isin(q8_labels)].copy()

# Create sentence embeddings
model = SentenceTransformer('all-mpnet-base-v2')
sentences = df_q8['sentence'].tolist()
print("Computing embeddings for sentences...")
embeddings = model.encode(sentences, show_progress_bar=True)
df_q8['embedding'] = list(embeddings)

# Aggregate embeddings per country-year
def mean_embedding(emb_list):
    """
    Calculate the element-wise mean of a list of embedding vectors.

    Args:
        emb_list (list[array-like]): List of embedding vectors with compatible dimensions.

    Returns:
        numpy.ndarray: Mean embedding vector across all inputs.
    """
    return np.mean(np.vstack(emb_list), axis=0)

agg_emb = (
    df_q8.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['embedding']
    .apply(mean_embedding)
    .reset_index()
)

# Aggregate sentence counts and proportions
agg_counts = (
    df_q8.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label'])
    .size()
    .reset_index(name='sentence_count')
)
total_counts = (
    agg_counts.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['sentence_count']
    .sum()
    .reset_index(name='total_q8_count')
)
agg_counts = agg_counts.merge(total_counts, on=['B_COUNTRY_ALPHA', 'A_YEAR'])
agg_counts['proportion'] = agg_counts['sentence_count'] / agg_counts['total_q8_count']

agg_q8_1 = agg_counts[agg_counts['predicted_combined_label'] == 'Q8_1'].copy()

# Fix dtypes for merging
agg_q8_1['A_YEAR'] = agg_q8_1['A_YEAR'].astype(str)
agg_emb['A_YEAR'] = agg_emb['A_YEAR'].astype(str)

# Merge embeddings with proportions
agg_merged = agg_q8_1.merge(agg_emb, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# Add numeric year column for grouping
agg_merged['A_YEAR_int'] = agg_merged['A_YEAR'].astype(int)

# Extract latent factors (keep at least 9)
X = np.vstack(agg_merged['embedding'].values)
pca = PCA(n_components=10)
latent_factors = pca.fit_transform(X)
for i in range(latent_factors.shape[1]):
    agg_merged[f'latent_factor_{i+1}'] = latent_factors[:, i]

# Regress latent_factor_6 and latent_factor_9 on country/year FE
model_fe6 = smf.ols('latent_factor_6 ~ C(B_COUNTRY_ALPHA) + C(A_YEAR)', data=agg_merged).fit()
agg_merged['latent_factor_6_resid'] = model_fe6.resid

model_fe9 = smf.ols('latent_factor_9 ~ C(B_COUNTRY_ALPHA) + C(A_YEAR)', data=agg_merged).fit()
agg_merged['latent_factor_9_resid'] = model_fe9.resid

# Compute mean embeddings per year (all countries)
agg_emb['A_YEAR_int'] = agg_emb['A_YEAR'].astype(int)
mean_emb_by_year = agg_emb.groupby('A_YEAR_int')['embedding'].apply(lambda embs: np.mean(np.vstack(embs), axis=0))
mean_emb_by_year = mean_emb_by_year.sort_index()

# Compute cosine distances between consecutive years
years = mean_emb_by_year.index.to_list()
distances = []
for i in range(1, len(years)):
    d = cosine_distances(
        mean_emb_by_year.iloc[i-1].reshape(1, -1),
        mean_emb_by_year.iloc[i].reshape(1, -1)
    )[0][0]
    distances.append(d)

semantic_shift_df = pd.DataFrame({
    'A_YEAR_int': years[1:],
    'mean_cosine_distance': distances
})

# Calculate mean residuals for factors 6 and 9 per year
latent_year_avg6 = agg_merged.groupby('A_YEAR_int')['latent_factor_6_resid'].mean().reset_index()
latent_year_avg9 = agg_merged.groupby('A_YEAR_int')['latent_factor_9_resid'].mean().reset_index()

# Merge everything into one DataFrame
analysis_df = semantic_shift_df.merge(latent_year_avg6, on='A_YEAR_int')
analysis_df = analysis_df.merge(latent_year_avg9, on='A_YEAR_int')

# Step 13: Correlate each factor with semantic shift
corr6 = analysis_df['latent_factor_6_resid'].corr(analysis_df['mean_cosine_distance'])
corr9 = analysis_df['latent_factor_9_resid'].corr(analysis_df['mean_cosine_distance'])
print(f"Correlation between residual latent_factor_6 and mean semantic shift: {corr6:.3f}")
print(f"Correlation between residual latent_factor_9 and mean semantic shift: {corr9:.3f}")

# Plot results — all three lines on one plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Cosine Distance (Semantic Shift)', color=color1)
ax1.plot(analysis_df['A_YEAR_int'], analysis_df['mean_cosine_distance'], marker='o', color=color1, label='Semantic Shift')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
color3 = 'tab:green'
ax2.set_ylabel('Mean Residual Latent Factors', color='black')
ax2.plot(analysis_df['A_YEAR_int'], analysis_df['latent_factor_6_resid'], marker='s', color=color2, label='Latent Factor 6 Residual')
ax2.plot(analysis_df['A_YEAR_int'], analysis_df['latent_factor_9_resid'], marker='^', color=color3, label='Latent Factor 9 Residual')
ax2.tick_params(axis='y', labelcolor='black')

fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.title('Semantic Shift and Residual Latent Factors 6 & 9 Over Years (Country-Year FE Adjusted)')
plt.show()
