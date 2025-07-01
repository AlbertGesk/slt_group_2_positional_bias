from pathlib import Path
from loguru import logger
from io import StringIO

import typer
import zipfile
import pandas as pd

app = typer.Typer()


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
