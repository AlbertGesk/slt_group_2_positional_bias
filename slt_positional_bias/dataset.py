from pathlib import Path
from loguru import logger
from tqdm import tqdm
from slt_positional_bias.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from ir_datasets_subsample import register_subsamples

import os
import typer
import zipfile
import ir_datasets

app = typer.Typer()


@app.command()
def main():

    base = Path("../data")
    dirs = ["external", "interim", "processed"]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)

    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    input_zip_path = SCRIPT_DIR / "data/raw/.ir_datasets/corpus-subsamples/inputs.zip"
    input_output_dir = SCRIPT_DIR / "data/interim"

    if input_zip_path.exists():
        logger.info("input.zip exists")
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_output_dir)
            logger.success("input.zip extracted")
    else:
        logger.error("input.zip missing in data/raw/.ir_datasets/corpus-subsamples")

    # register_subsamples()
    # dataset = ir_datasets.load("data/interim/inputs/corpus")
    # dataset.docs_iter()

    print(list(ir_datasets.registry))

if __name__ == "__main__":
    app()
