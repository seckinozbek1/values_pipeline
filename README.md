This pipeline was prepared for the empirical study titled *"Values in UN General Debate Speeches"* by Seckin Ozbek, which was based on the final Master's dissertation in Applied Social Data Science programme at the London School of Economics and Political Science in August 2025.  

# Data

For demonstration purposes, the required subfolders in downloaded datasets were included in the data directory. Please follow the instructions below for complete replication and refer to the citations for the producers of the datasets as mentioned in the references section.

## UNGD Corpus and Metadata

To analyze the UN General Debate Speeches, the text corpus of Jankin et al. (2025) was used. Hence, in order to replicate the analyses in this study, please first download the corpus from the following link and extract the `UNGDC_1946-2024.tar` file in the data directory and process the TXT file folder for the analysis:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y

To acquire the speaker metadata, please download the `Speakers_by_session.xlsx` excel file into the `data` directory using the same link.

## WVS Wave 7 Dataset

In order to acquire the Wave 7 Dataset, please download `WVS Cross-National Wave 7 R v6 0.zip (rdata)` from the following link and save into the `data` directory: 

https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp

## World Values Corpus (WVC) Hypothesis Sentences

The WVC corpus sentences are obtained from the supplementary materials of the study of Benkler et al. (2024) using the following link:

https://aclanthology.org/2024.lrec-main.1195/

Please download the `.csv` file in the following path into the data directory:

"2024.lrec-main.1195.OptionalSupplementaryMaterial/RVR_SUPPLEMENT_COLING_2024/data/WVC/WVC_stem_encoding.csv"

## UNSG Addresses

The UN Secretary-General's Addresses to the General Assembly and opening remarks between the years 2017 and 2022 were taken from the UN's official website (United Nations, 2017, 2018, 2019, 2020, 2021, 2022).

For 2020, due to the richness of content in comparison to other speeches, the opening remarks for the General Assembly were obtained from the official website instead of the address. The speech files are saved into `unsg_speeches` folder in the `data` directory.

# Directory Structure

The directory structure excluding the `cache` folder in `code` and `TXT` folder in `data` directory is provided below. The aforementioned files were omitted due to having a large number of contents.
```
values_pipeline/
├── README.md                          # how to reproduce core results
├── code/
│   ├── master_code_prep/              # data cleaning & dataset assembly
│   ├── question_pipelines/            # item-specific training & inference (Q8, Q11, …)
│   └── visuals_pipeline/              # scripts to build figures & LaTeX tables
├── data/
│   ├── labeled_data/                  # human-labeled training sets
│   └── unsg_speeches/                 # UNSG address texts (raw)
├── output/
│   ├── master_code_prep_output/       # processed, analysis-ready datasets
│   ├── question_pipeline_output/      # predictions & per-item summaries
│   └── visuals_pipeline_output/
│       ├── figures/                   # final PNGs used in the paper
│       └── tables/                    # final .tex tables used in the paper
└── requirements.txt                   # runtime dependencies
```
For transparency and reproducibility, the full directory structure can be generated automatically. Execute `print_folder_tree.py` from `master_code.ipynb` (located in `code/master_code_prep/`) to produce the complete project tree.

# Computing Environment and Package Requirements

Model training was GPU-accelerated using an NVIDIA GeForce RTX 3070 Ti Laptop GPU under CUDA. CPU-only replication is fully possible but substantially slower. All analyses were executed in Python 3.10.16.

Please install the packages mentioned in `requirements.txt` file in `code/master_code_prep` directory before replication. For ease of reference, the requirements list is also provided below:
```
IPython
ast
catboost
glob
hashlib
itertools
json
keybert
logging
matplotlib
nltk
numpy
os
pandas
pathlib
pickle
plotly
pycountry_convert
random
re
rpy2
seaborn
sentence_transformers
sklearn
statsmodels
sys
torch
tqdm
traceback
transformers
umap
warnings
```
# Instructions

- After downloading the datasets or ensuring that the data files exist in the directory, please run "values_pipeline\code\master_code_prep\master_code.ipynb" first and then run the following `.ipynb` files, in order:

    ```
    values_pipeline\code\question_pipelines\q8_pipeline\master_q8.ipynb
    values_pipeline\code\question_pipelines\q11_pipeline\master_q11.ipynb
    values_pipeline\code\question_pipelines\q17_pipeline\master_q17.ipynb
    values_pipeline\code\question_pipelines\q65_pipeline\master_q65.ipynb
    values_pipeline\code\question_pipelines\q69_pipeline\master_q69.ipynb
    values_pipeline\code\question_pipelines\q70_pipeline\master_q70.ipynb
    values_pipeline\code\question_pipelines\q152_q153_pipeline\master_q152_q153.ipynb
    values_pipeline\code\question_pipelines\q154_q155_pipeline\master_q154_q155.ipynb
    values_pipeline\code\question_pipelines\q8_pipeline\master_country_year_fe_experiments.ipynb
    values_pipeline\code\question_pipelines\q69_pipeline\master_country_year_fe_experiments.ipynb
    ```
    You may check the output files in the output directory. You may also get further insights on which criteria were considered for domain adaptations from surveys to speeches from each question-specific directory.

- After having run the code for the questions, please run the following code to acquire the figures and tables used in the research. 
    ```
    values_pipeline\code\visuals_pipeline\master_visuals.ipynb
    ```
The figures and tables will be saved into the `visuals_pipeline_output` folder in the `output` directory. 

**Note:** The only non-replicable output is the case mentioned as the confounded metrics for Q69 of the WVS, which could not be replicated in further runs due to initial non-deterministic structure of the code. The visuals and rescued metrics for this could be found in the `confounded_q69` folder in the `output` directory. 

# References: 

- Jankin, S., Baturo, A., & Dasandi, N. (2025). Words to unite nations: The complete United Nations General Debate Corpus, 1946–present. Journal of Peace Research, 62(4), 1339–1351. https://doi.org/10.1177/00223433241275335

- Alexander Baturo, Niheer Dasandi, and Slava Mikhaylov, "Understanding State Preferences With Text As Data: Introducing the UN General Debate Corpus" Research & Politics, 2017.

- Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen (eds.). 2022. World Values Survey: Round Seven – Country-Pooled Datafile Version 6.0. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.24

- Benkler, N. K., Friedman, S., Schmer-Galunder, S., Mosaphir, D. M., Goldman, R. P., Wheelock, R., Sarathy, V., Kantharaju, P., & McLure, M. D. (2024). Recognizing value resonance with resonance-tuned RoBERTa: Task definition, experimental validation, and robust modeling. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) (pp. 13688–13698). ELRA & ICCL.

- United Nations. (2017, September 19). Secretary-General’s address to the General Assembly. https://www.un.org/sg/en/content/sg/statement/2017-09-19/secretary-generals-address-the-general-assembly

- United Nations. (2018, September 25). Secretary-General’s address to the General Assembly. https://www.un.org/sg/en/content/sg/statement/2018-09-25/secretary-generals-address-the-general-assembly-delivered-trilingual-scroll-further-down-for-all-english

- United Nations. (2019, September 24). Secretary-General’s address to the General Assembly. https://www.un.org/sg/en/content/sg/statement/2019-09-24/secretary-generals-address-the-general-assembly-trilingual-delivered-scroll-down-for-all-english

- United Nations. (2020, September 22). Secretary-General’s address to the opening of the general debate of the 75th session of the General Assembly. https://www.un.org/sg/en/content/sg/statement/2020-09-22/secretary-generals-address-the-opening-of-the-general-debate-of-the-75th-session-of-the-general-assembly

- United Nations. (2021, September 21). Secretary-General’s address to the General Assembly. https://www.un.org/sg/en/content/sg/statement/2021-09-21/secretary-general%E2%80%99s-address-the-general-assembly

- United Nations. (2022, September 20). Secretary-General’s address to the General Assembly. https://www.un.org/sg/en/content/sg/statement/2022-09-20/secretary-generals-address-the-general-assembly-trilingual-delivered-follows-scroll-further-down-for-all-english-and-all-french

# Citation

If you use this pipeline, please cite:
> Ozbek, Seckin (2025). *Values in UN General Debate Speeches*. MSc Dissertation, London School of Economics and Political Science.

When the journal version becomes available, please update the citation accordingly.