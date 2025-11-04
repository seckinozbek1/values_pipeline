import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# Load data
predictions_path = "../../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

q69_labels = ['Q69_1', 'Q69_2', 'Q69_3', 'Q69_4']
df_q69 = df[df['predicted_combined_label'].isin(q69_labels)].copy()

# Create sentence embeddings
model = SentenceTransformer('all-mpnet-base-v2')
sentences = df_q69['sentence'].tolist()
print("Computing embeddings for sentences...")
embeddings = model.encode(sentences, show_progress_bar=True)
df_q69['embedding'] = list(embeddings)

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
    df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['embedding']
    .apply(mean_embedding)
    .reset_index()
)

# Aggregate sentence counts and proportions
agg_counts = (
    df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label'])
    .size()
    .reset_index(name='sentence_count')
)
total_counts = (
    agg_counts.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['sentence_count']
    .sum()
    .reset_index(name='total_q69_count')
)
agg_counts = agg_counts.merge(total_counts, on=['B_COUNTRY_ALPHA', 'A_YEAR'])
agg_counts['proportion'] = agg_counts['sentence_count'] / agg_counts['total_q69_count']

agg_q69_1 = agg_counts[agg_counts['predicted_combined_label'] == 'Q69_1'].copy()

# Fix dtypes for merging
agg_q69_1['A_YEAR'] = agg_q69_1['A_YEAR'].astype(str)
agg_emb['A_YEAR'] = agg_emb['A_YEAR'].astype(str)

# Merge embeddings with proportions
agg_merged = agg_q69_1.merge(agg_emb, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# Add numeric year column for grouping
agg_merged['A_YEAR_int'] = agg_merged['A_YEAR'].astype(int)

# Extract latent factors
X = np.vstack(agg_merged['embedding'].values)
pca = PCA(n_components=20)
latent_factors = pca.fit_transform(X)
for i in range(latent_factors.shape[1]):
    agg_merged[f'latent_factor_{i+1}'] = latent_factors[:, i]

# Regress latent_factor_12 on country and year fixed effects
model_fe = smf.ols('latent_factor_12 ~ C(B_COUNTRY_ALPHA) + C(A_YEAR)', data=agg_merged).fit()
agg_merged['latent_factor_12_resid'] = model_fe.resid

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

# Calculate mean residual latent_factor_12 per year
latent_year_avg = agg_merged.groupby('A_YEAR_int')['latent_factor_12_resid'].mean().reset_index()
latent_year_avg.rename(columns={'latent_factor_12_resid': 'mean_latent_factor_12_resid'}, inplace=True)

# Merge residual latent_factor_12 with semantic shifts
analysis_df = semantic_shift_df.merge(latent_year_avg, on='A_YEAR_int')

# Correlate residual latent_factor_12 and semantic shifts
corr = analysis_df['mean_latent_factor_12_resid'].corr(analysis_df['mean_cosine_distance'])
print(f"Correlation between residual latent_factor_12 and mean semantic shift: {corr:.3f}")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean Cosine Distance (Semantic Shift)', color=color)
ax1.plot(analysis_df['A_YEAR_int'], analysis_df['mean_cosine_distance'], marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Mean Residual Latent Factor 12', color=color)
ax2.plot(analysis_df['A_YEAR_int'], analysis_df['mean_latent_factor_12_resid'], marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Semantic Shift and Residual Latent Factor 12 Over Years (Country-Year FE Adjusted)')
plt.show()