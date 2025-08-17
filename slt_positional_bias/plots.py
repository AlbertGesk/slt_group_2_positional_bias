from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from slt_positional_bias.config import FIGURES_DIR, PROCESSED_DATA_DIR
from matplotlib.patches import Rectangle

app = typer.Typer()



def export_table_txt(df, table_name, df_name, include_index=False):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/tables" / f"{table_name}-{df_name}.txt"

    df.to_csv(
        f_path_from_dir,
        sep="\t",
        index=include_index,
        float_format="%.6f",
        na_rep="NA",
            )


def savetable(df, table_name, df_name):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/tables" / f"{table_name}-{df_name}.csv"
    
    df.to_csv(f_path_from_dir)

def savefig(plt, plt_name, df_name, dpi):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/figures" / f"{plt_name}-{df_name}.png"

    plt.savefig(f_path_from_dir, dpi=dpi)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
