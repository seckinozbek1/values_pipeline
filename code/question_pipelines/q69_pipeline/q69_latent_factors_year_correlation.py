import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px

# --------------------------
# Load data
# --------------------------
predictions_path = "../../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

q69_labels = ['Q69_1', 'Q69_2', 'Q69_3', 'Q69_4']
df_q69 = df[df['predicted_combined_label'].isin(q69_labels)].copy()

# --------------------------
# Create embeddings for each sentence
# --------------------------
model = SentenceTransformer('all-mpnet-base-v2')
sentences = df_q69['sentence'].tolist()
print("Computing embeddings for sentences...")
embeddings = model.encode(sentences, show_progress_bar=True)
df_q69['embedding'] = list(embeddings)

# --------------------------
# Aggregate embeddings per country-year by mean
# --------------------------
def mean_embedding(emb_list):
    """
    Compute the mean embedding vector from a list of embeddings.

    Args:
        emb_list (list[array-like]): List of embedding vectors.

    Returns:
        numpy.ndarray: Mean embedding vector.
    """
    return np.mean(np.vstack(emb_list), axis=0)

agg_emb = (
    df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['embedding']
    .apply(mean_embedding)
    .reset_index()
)

# --------------------------
# Aggregate sentence counts and proportions
# --------------------------
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

# Ensure year types match for merging
agg_q69_1['A_YEAR'] = agg_q69_1['A_YEAR'].astype(str)
agg_emb['A_YEAR'] = agg_emb['A_YEAR'].astype(str)

# Merge embeddings with proportions
agg_merged = agg_q69_1.merge(agg_emb, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# --------------------------
# Extract latent factors using PCA
# --------------------------
X = np.vstack(agg_merged['embedding'].values)
pca = PCA(n_components=20)
latent_factors = pca.fit_transform(X)

for i in range(latent_factors.shape[1]):
    agg_merged[f'latent_factor_{i+1}'] = latent_factors[:, i]

# Convert year to numeric for correlation
agg_merged['A_YEAR_num'] = agg_merged['A_YEAR'].astype(int)

# --------------------------
# Correlation between latent factors and year
# --------------------------
for i in range(1, 21):
    corr = agg_merged[f'latent_factor_{i}'].corr(agg_merged['A_YEAR_num'])
    print(f'Correlation of latent_factor_{i} with year: {corr:.3f}')