from pathlib import Path
from loguru import logger
from io import StringIO
from pathlib import Path
from numpy import integer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.stats import spearmanr

import typer
import zipfile
import pandas as pd
import re
import nltk

from slt_positional_bias.features import sacrebleu_corpus, rouge_corpus, meteor_corpus, bertscore_corpus
from slt_positional_bias.plots import export_table_txt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = typer.Typer()




def calc_metric(dfs, indexed: bool, normalized: bool):
    new_dfs = {}

    for name, d in dfs.items():
        nd = d.copy()
        nd.rename(columns={'rel_3_doc_position': 'Position of Oracle', 'nr_rel_0_doc': 'Number of Documents', 'oracle': 'references', 'answer': 'predictions'}, inplace=True)
        nd.drop(columns=['nr_rel_3_doc'], inplace=True)
        nd['Position of Oracle'] = nd['Position of Oracle'] + 1
        nd['Number of Documents'] = nd['Number of Documents'] + 1

        nd['Jaccard Coefficient'] = nd.apply(
            lambda row: jaccard(row['predictions'], row['references']), axis=1
        )
        nd[['Spearman Correlation Coefficient', 'Spearman p-value']] = nd.apply(
            lambda row: pd.Series(spearman_word_order_correlation(row['predictions'], row['references'])), axis=1
        )
        nd['SacreBLEU'] = nd.apply(
            lambda row: pd.Series(sacrebleu_corpus(row['predictions'], row['references'])), axis=1
        )
        nd['ROUGE_1', 'ROUGE_2', 'ROUGE_L', 'ROUGE_Lsum'] = nd.apply(
            lambda row: pd.Series(rouge_corpus(row['predictions'], row['references'])), axis=1
        )
        nd['METEOR'] = nd.apply(
            lambda row: pd.Series(meteor_corpus(row['predictions'], row['references'])), axis=1
        )
        nd[["BERTScore-Precision", "BERTScore-Recall", "BERTScore-F1"]] = nd.apply(
            lambda row: pd.Series(bertscore_corpus(row['predictions'], row['references'])), axis=1
        )

        new_dfs[f'n{name}'] = nd

    dfs_jaccard = {}
    dfs_spearman = {}
    dfs_bleu = {}
    dfs_rouge = {}
    dfs_meteor = {}
    dfs_berts = {}

    for name, d in new_dfs.items():
        df_jaccard = d.groupby('Position of Oracle')[['Jaccard Coefficient']].mean().reset_index()
        df_spearman = d.groupby('Position of Oracle')[['Spearman Correlation Coefficient', 'Spearman p-value']].mean().reset_index()
        df_bleu = d.groupby('Position of Oracle')[['SacreBLEU']].mean().reset_index()
        df_rouge = d.groupby('Position of Oracle')[['ROUGE_1', 'ROUGE_2', 'ROUGE_L', 'ROUGE_Lsum']].mean().reset_index()
        df_meteor = d.groupby('Position of Oracle')[['METEOR']].mean().reset_index()
        df_berts = d.groupby('Position of Oracle')[['BERTScore-Precision', 'BERTScore-Recall', 'BERTScore-F1']].mean().reset_index()

        if normalized:
            dfs_jaccard[f'Norm-{name}_jaccard'] = df_jaccard
            dfs_spearman[f'Norm-{name}_spearman'] = df_spearman
            dfs_bleu[f'Norm-{name}_bleu'] = df_bleu
            dfs_rouge[f'Norm-{name}_rouge'] = df_rouge
            dfs_meteor[f'Norm-{name}_meteor'] = df_meteor
            dfs_berts[f'Norm-{name}_berts'] = df_berts
        else:
            dfs_jaccard[f'{name}_jaccard'] = df_jaccard
            dfs_spearman[f'{name}_spearman'] = df_spearman
            dfs_bleu[f'{name}_bleu'] = df_bleu
            dfs_rouge[f'{name}_rouge'] = df_rouge
            dfs_meteor[f'{name}_meteor'] = df_meteor
            dfs_berts[f'{name}_berts'] = df_berts

    for dfs in [dfs_jaccard, dfs_spearman, dfs_bleu, dfs_rouge, dfs_meteor, dfs_berts]:
        for name, d in dfs.items():
            if dfs == dfs_jaccard:
                export_table_txt(d, "Jaccard Coefficient", f"{name}", indexed, normalized)
            if dfs == dfs_spearman:
                export_table_txt(d, "Spearman Correlation Coefficient", f"{name}", indexed, normalized)
            if dfs == dfs_bleu:
                export_table_txt(d, "SacreBLEU", f"{name}", indexed, normalized)
            if dfs == dfs_rouge:
                export_table_txt(d, "ROUGE", f"{name}", indexed, normalized)
            if dfs == dfs_meteor:
                export_table_txt(d, "METEOR", f"{name}", indexed, normalized)
            if dfs == dfs_berts:
                export_table_txt(d, "BERTScore", f"{name}", indexed, normalized)


def normalize_and_tokenize_list(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words

def normalize_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return set(words)

def spearman_word_order_correlation(predictions, references):
    oracle_tokenized = normalize_and_tokenize_list(references)
    answer_tokenized = normalize_and_tokenize_list(predictions)

    intersection = set(oracle_tokenized) & set(answer_tokenized)
    if len(intersection) < 2:
        return None, None

    oracle_ranks = [oracle_tokenized.index(word) for word in intersection]
    answer_ranks = [answer_tokenized.index(word) for word in intersection]

    coef, p = spearmanr(oracle_ranks, answer_ranks)
    return coef, p

def jaccard(predictions, references):
    set_1 = normalize_and_tokenize(references)
    set_2 = normalize_and_tokenize(predictions)
    return len(set_1 & set_2) / len(set_1 | set_2) if set_1 | set_2 else 0.0

def sort_data_frame(df: pd.DataFrame, nr_rel_3: int, nr_rel_0: int) -> pd.DataFrame:
    # Group by topic_ids
    groups = df.groupby("topic_id", group_keys=False)

    # List of topic_ids with count(rel_scoring==0) < nr_rel_0 && count(rel_scoring==3) < nr_rel_3
    valid_topic_ids = [
        topic for topic, group in groups
        if (group["rel_scoring"] == 0).sum() >= nr_rel_0 and (group["rel_scoring"] == 3).sum() >= nr_rel_3
    ]

    # Include only topic_id entries that fulfils the filter's condition
    df_filtered = df[df["topic_id"].isin(valid_topic_ids)]

    # Group by filtered topic_ids
    grouped = df_filtered.groupby("topic_id", group_keys=False)

    # Cut off excess entries
    df_filtered_rel = grouped.apply(
        lambda g: pd.concat([
            g[g["rel_scoring"] == 0].head(nr_rel_0),
            g[g["rel_scoring"] == 3].head(nr_rel_3)
        ])
    )

    df_filtered_rel = df_filtered_rel.reset_index(drop=True)

    return df_filtered_rel

def generate_merged_data_frame():
    df_docs = generate_data_frame("data/interim/inputs/corpus.jsonl")
    df_topics = generate_data_frame("data/interim/inputs/queries.jsonl")
    df_qrels = generate_data_frame("data/interim/inputs/qrels.rag24.test-umbrela-all.txt")

    df_qrels.columns = ["topic_id", "q0", "doc_id", "rel_scoring"]

    df_topics = df_topics.rename(columns={"qid": "topic_id","query": "topic"})
    df_docs = df_docs.rename(columns={"docno": "doc_id", "text": "doc"})

    df_merged = (
        df_qrels
        .loc[:, ["topic_id", "doc_id", "rel_scoring"]]
        .merge(
            df_topics.loc[:, ["topic_id", "topic"]],
            on="topic_id",
            how="inner"
        )
        .merge(
            df_docs.loc[:, ["doc_id", "doc"]],
            on="doc_id",
            how="inner"
        )
    )

    df = df_merged[df_merged["doc_id"].duplicated(keep=False) == False]
    df = df.dropna()

    logger.info(f"df column 'doc_id' is unique: {df["doc_id"].is_unique}")
    logger.info(f"df contains NaN: {df.isna().any().any()}")

    return df

def store_df_as_parquet_time(df: pd.DataFrame, file_name: str):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "data/processed"

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %Hh-%Mm-%Ss')
    name = f"{file_name}-1-{timestamp}.parquet"
    path = f_path_from_dir / name

    counter = 1
    while path.exists():
        name = f"{file_name}-{counter}-{timestamp}.parquet"
        path = f_path_from_dir / name
        counter += 1

    df.to_parquet(path)
    logger.info(f"Data frame saved to {path}")
    
def store_df_as_parquet(df: pd.DataFrame, file_name: str):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "data/processed"

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %Hh-%Mm-%Ss')
    name = f"{file_name}.parquet"
    path = f_path_from_dir / name

    while path.exists():
        name = f"{file_name}.parquet"
        path = f_path_from_dir / name

    df.to_parquet(path)
    logger.info(f"Data frame saved to {path}")

def load_parquet_as_df(file_name: str) -> pd.DataFrame:
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path = f"data/processed/{file_name}.parquet"
    f_path_from_dir = SCRIPT_DIR / f_path
    df = pd.read_parquet(f_path_from_dir)

    return df

def load_and_sort_data(start: int = 1, end: int = 39) -> pd.DataFrame:
    df = generate_merged_data_frame()
    df_sorted = sort_data_frame(df, start, end)
    return df_sorted

def generate_data_frame(f_path: str):

    # convert json or txt files to pandas data frame
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / f_path
    file_type = f_path_from_dir.suffix

    if f_path_from_dir.exists():
        logger.info("file path exists")
        with open(f_path_from_dir, 'r') as f:
            file_string = f.read()

        if file_type in [".json", ".jsonl"]:
            data_frame = pd.read_json(StringIO(file_string), lines=True)
            return data_frame
        elif file_type == ".txt":
            data_frame = pd.read_csv(StringIO(file_string), sep=" ")
            return data_frame
        else:
            logger.error("file type not supported by function")
            return pd.DataFrame()

    else:
        logger.error(f"file path '{f_path_from_dir}' doesn't exist")

@app.command()
def main():

    # mkdir external, interim, processed
    base = Path("../data")
    dirs = ["external", "interim", "processed"]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)

    # unzip inputs.zip
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    inputs_zip_path = SCRIPT_DIR / "data/raw/.ir_datasets/corpus-subsamples/inputs.zip"
    inputs_output_dir = SCRIPT_DIR / "data/interim"

    if inputs_zip_path.exists():
        logger.info("input.zip exists")
        with zipfile.ZipFile(inputs_zip_path, 'r') as zip_ref:
            zip_ref.extractall(inputs_output_dir)
            logger.success("input.zip extracted")
    else:
        logger.error("input.zip missing in data/raw/.ir_datasets/corpus-subsamples")


if __name__ == "__main__":
    app()
