import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
from IPython.display import display


nltk.download('punkt', quiet=True)

# Clean and normalize input text by lowercasing, standardizing quotes, and removing unwanted characters
def clean_text(text):
    """
    Lowercase and normalize text, standardize quotation marks,
    remove non-alphanumeric characters (except basic punctuation),
    and collapse extra whitespace.
    """
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9\s.,!?'\"]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Directory where the speech files are stored
base_dir = r"../../data/unsg_speeches"

# Collect sentences from all files
sentence_rows = []

# List all text files matching "unsg_*.txt"
files = [f for f in os.listdir(base_dir) if re.match(r"unsg_\d{4}\.txt", f)]

for filename in tqdm(files, desc="Processing files"):
    filepath = os.path.join(base_dir, filename)
    # Extract year from filename, e.g., "unsg_2017.txt" -> 2017
    year_match = re.search(r"unsg_(\d{4})\.txt", filename)
    if not year_match:
        continue
    year = int(year_match.group(1))
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_clean = clean_text(text)
    sentences = sent_tokenize(text_clean)
    
    # Filter sentences between 10 and 40 words (inclusive)
    for sent in sentences:
        word_count = len(sent.split())
        if 10 < word_count <= 40:
            sentence_rows.append({
                "doc_id": f"unsg_{year}",
                "A_YEAR": year,
                "sentence_text": sent
            })

# Create DataFrame
df_sentences = pd.DataFrame(sentence_rows)

# Shuffle the dataframe rows
df_sentences = df_sentences.sample(frac=1).reset_index(drop=True)

# Save to CSV
output_path = r"../../output/master_code_prep_output/unsg_address_corpus.csv"
df_sentences.to_csv(output_path, index=False, encoding="utf-8")

print(f"Processed {len(df_sentences)} sentences from {len(files)} files.")
print(f"Saved output to: {output_path}") 
pd.set_option('display.max_colwidth', None)
display(df_sentences.head(5))
