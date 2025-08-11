from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt

from slt_positional_bias.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()

def savetable(df, table_name, df_name):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/tables" / f"{table_name}-{df_name}.csv"
    
    df.to_csv(f_path_from_dir)

def savefig(plt, plt_name, df_name, dpi):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/figures" / f"{plt_name}-{df_name}.png"

    plt.savefig(f_path_from_dir, dpi=dpi)

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
