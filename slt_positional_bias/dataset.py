from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import zipfile

from slt_positional_bias.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
app = typer.Typer()

from ir_datasets_subsample import register_subsamples
import ir_datasets

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    input_zip_path = Path("../data/external/.ir_datasets/corpus-subsamples/inputs.zip")
    input_output_dir = Path("../data/interim")

    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_output_dir)

    logger.info("Registering IR subsamples...")
    register_subsamples()

    logger.info("Loading sub-sampled dataset...")
    dataset = ir_datasets.load("../data/interim/inputs/corpus.jsonl")

    logger.info("Iterating over documents...")
    dataset.docs_iter()

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    app()
