from pathlib import Path
from loguru import logger
from io import StringIO

import typer
import zipfile
import pandas as pd
from pathlib import Path

from numpy import integer

app = typer.Typer()

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
            g[g["rel_scoring"] == 0].head(36),
            g[g["rel_scoring"] == 3].head(4)
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
