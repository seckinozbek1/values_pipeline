import pandas as pd
from IPython.display import display

# Load the dataset
unga_df = pd.read_csv('../../output/master_code_prep_output/unga_wvs7_hashed_corpus.csv')

valid_years = list(range(2017, 2023))
valid_countries = {
    "AND","ARG","AUS","BGD","ARM","BOL","BRA","MMR","CAN","CHL","CHN","COL","CYP","CZE",
    "ECU","ETH","DEU","GRC","GTM","IND","IDN","IRN","IRQ","JPN","KAZ","JOR","KEN","KOR",
    "KGZ","LBN","LBY","MYS","MDV","MEX","MNG","MAR","NLD","NZL","NIC","NGA","PAK","PER",
    "PHL","ROU","RUS","SRB","SGP","SVK","VNM","ZWE","TJK","THA","TUN","TUR","UKR","EGY",
    "GBR","USA","URY","UZB","VEN"
}

# Filter the UNGA data according to WVS Wave 7 countries and years
unga_filtered = unga_df[
    unga_df['B_COUNTRY_ALPHA'].isin(valid_countries)
    & unga_df['A_YEAR'].isin(valid_years)
]

# Get the counts of sentences per country
unga_counts = (
    unga_filtered
    .groupby('B_COUNTRY_ALPHA')['sentence_text']
    .count()
    .reset_index(name='UNGD Sentence Count')
    .rename(columns={'B_COUNTRY_ALPHA': 'Country'})
)

# Load WVS Wave 7 full data
wvs_path = r"../../output/master_code_prep_output/wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Filter the data according to valid questions and countries
questions = ['Q8','Q11','Q17','Q65','Q69','Q70','Q152','Q153','Q154','Q155']
counts_df = pd.DataFrame(index=sorted(valid_countries), columns=questions)

# Loop over WVS questions and get sample sizes per country
for q in questions:
    if q not in wvs_df.columns:
        print(f"Warning: {q} not in WVS file - skipped.")
        counts_df.drop(columns=q, inplace=True)
        continue
    q_counts = (
        wvs_df[(wvs_df['B_COUNTRY_ALPHA'].isin(valid_countries)) & wvs_df[q].notna()]
        .groupby('B_COUNTRY_ALPHA')[q]
        .count()
    )
    counts_df.loc[q_counts.index, q] = q_counts

counts_df = counts_df.fillna(0).astype(int)
sample_size_same = counts_df.nunique(axis=1) == 1
wvs_sizes = (
    pd.DataFrame({
        'Country': counts_df.index,
        'WVS Sample Size': counts_df[questions[0]].values
    })
    .loc[sample_size_same.values]
    .reset_index(drop=True)
)

combined_table = pd.merge(unga_counts, wvs_sizes, on='Country', how='inner')

print("Combined table: Country, UNGD Sentence Count, WVS Sample Size")
display(combined_table)

# Save the table as LaTeX
output_path = "../../output/visuals_pipeline_output/tables/sample_size_table.tex"
combined_table.to_latex(
    output_path,
    index=False,
    caption="Combined UNGD sentence counts and WVS sample sizes",
    label="tab:combined",
    column_format="lrr"         
)
print(f"LaTeX saved -> {output_path}")