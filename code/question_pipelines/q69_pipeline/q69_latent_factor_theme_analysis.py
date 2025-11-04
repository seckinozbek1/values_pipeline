import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# -------------------
# Load data and filter
# -------------------
predictions_path = "../../../output/question_pipeline_output/q69_predictions/q69_predictions_filtered.csv"
df = pd.read_csv(predictions_path)

q69_labels = ['Q69_1', 'Q69_2', 'Q69_3', 'Q69_4']
df_q69 = df[df['predicted_combined_label'].isin(q69_labels)].copy()

# -------------------
# Compute embeddings
# -------------------
model = SentenceTransformer('all-mpnet-base-v2')
print("Computing sentence embeddings...")
df_q69['embedding'] = list(model.encode(df_q69['sentence'].tolist(), show_progress_bar=True))

# -------------------
# Aggregate embeddings per country-year
# -------------------
def mean_embedding(emb_list):
    """
    Calculate the element-wise mean of a list of embedding vectors.

    Args:
        emb_list (list[array-like]): List of embedding vectors with compatible dimensions.

    Returns:
        numpy.ndarray: Mean embedding vector across all inputs.
    """
    return np.mean(np.vstack(emb_list), axis=0)
agg_emb = df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['embedding'].apply(mean_embedding).reset_index()

# -------------------
# Aggregate counts and proportions
# -------------------
agg_counts = df_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='sentence_count')
total_counts = agg_counts.groupby(['B_COUNTRY_ALPHA', 'A_YEAR'])['sentence_count'].sum().reset_index(name='total_q69_count')
agg_counts = agg_counts.merge(total_counts, on=['B_COUNTRY_ALPHA', 'A_YEAR'])
agg_counts['proportion'] = agg_counts['sentence_count'] / agg_counts['total_q69_count']
agg_q69_1 = agg_counts[agg_counts['predicted_combined_label'] == 'Q69_1'].copy()
agg_q69_1['A_YEAR'] = agg_q69_1['A_YEAR'].astype(str)
agg_emb['A_YEAR'] = agg_emb['A_YEAR'].astype(str)
agg_merged = agg_q69_1.merge(agg_emb, on=['B_COUNTRY_ALPHA', 'A_YEAR'])

# -------------------
# PCA for latent factors
# -------------------
X = np.vstack(agg_merged['embedding'].values)
pca = PCA(n_components=20) 
latent_factors = pca.fit_transform(X)
for i in range(latent_factors.shape[1]):
    agg_merged[f'latent_factor_{i+1}'] = latent_factors[:, i]

# -------------------
# Merge factor values back to original sentences
# -------------------
agg_factor = agg_merged[['B_COUNTRY_ALPHA', 'A_YEAR', 'latent_factor_12']].copy()
agg_factor['A_YEAR'] = agg_factor['A_YEAR'].astype(str)
df_q69['A_YEAR'] = df_q69['A_YEAR'].astype(str)
df_q69_with_factor = df_q69.merge(agg_factor, on=['B_COUNTRY_ALPHA', 'A_YEAR'], how='left')

# -------------------
# Function for analysis and plotting per factor
# -------------------
def analyze_factor(df, factor_name):
    """
    Identify dominant semantic themes associated with a given latent factor
    and visualize their prevalence over time.

    The function extracts the top 200 sentences with the highest values for
    the specified latent factor, derives keyphrases using KeyBERT, embeds
    these themes with Sentence-BERT, computes cosine similarities between
    all sentences and each theme, and estimates the yearly prevalence of
    each theme based on a similarity threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing sentence texts, embeddings, year information,
        and latent factor scores.
    factor_name : str
        The column name of the latent factor to analyze (e.g., 'latent_factor_12').

    Returns
    -------
    None
        Displays a line plot showing yearly prevalence of the top 10 extracted themes.
    """
    # Get top 200 sentences by this factor
    top_pos = df.sort_values(factor_name, ascending=False).head(200)

    # Extract themes using KeyBERT
    kw_model = KeyBERT('all-mpnet-base-v2')
    def extract_themes(texts, top_n=10):
        corpus = " ".join(texts)
        return kw_model.extract_keywords(corpus, keyphrase_ngram_range=(1, 4), stop_words='english', top_n=top_n)

    themes_pos = extract_themes(top_pos['sentence'].tolist())
    print(f"\nThemes for high {factor_name} sentences:")
    for theme, score in themes_pos:
        print(f"- {theme} ({score:.3f})")

    # Embed the theme phrases
    theme_texts = [t[0] for t in themes_pos]
    theme_embeddings = model.encode(theme_texts)

    # Compute cosine similarity between every sentence and each theme
    sentence_embeddings = np.vstack(df['embedding'].values)
    similarity_matrix = cosine_similarity(sentence_embeddings, theme_embeddings)

    # Set similarity threshold to count a sentence as expressing that theme
    threshold = 0.5

    # Compute yearly prevalence
    df['A_YEAR_int'] = df['A_YEAR'].astype(int)
    theme_prevalence = {}
    for i, theme in enumerate(theme_texts):
        theme_mask = similarity_matrix[:, i] > threshold
        df[f'theme_{i}_match'] = theme_mask
        prevalence = df.groupby('A_YEAR_int')[f'theme_{i}_match'].mean()
        theme_prevalence[theme] = prevalence

    prevalence_df = pd.DataFrame(theme_prevalence)

    # Plot
    plt.figure(figsize=(12, 6))
    for theme in prevalence_df.columns[:10]:
        plt.plot(prevalence_df.index, prevalence_df[theme], label=theme)

    plt.xlabel('Year')
    plt.ylabel('Theme Prevalence (fraction of sentences)')
    plt.title(f'Semantic Theme Prevalence Over Time (High {factor_name} Sentences)')
    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.show()

# -------------------
# Run for latent_factor_12
# -------------------
analyze_factor(df_q69_with_factor.copy(), 'latent_factor_12')