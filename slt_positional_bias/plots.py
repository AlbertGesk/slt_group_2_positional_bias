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




def plot_spearman_dict(dfs_spearman, save=False, xlim=(0, 21), ylim=(0, 1)):
    """
    Make a Spearman p-value line plot for each df in a dict.

    Parameters
    ----------
    dfs_spearman : dict[str, pd.DataFrame]
        Dict mapping a name -> dataframe.
    save : bool
        If True, saves each plot as PNG in `folder`.
    xlim, ylim : tuple
        Axis limits.
    """
    # allow for slight column-name variations
    x_candidates = [
        "Position of oracle document",
        "Position of Oracle document",
        "Position of Oracle",
    ]
    y_candidates = ["Spearman p-value", "Spearman p value", "Spearman p_value"]

    for name, df in dfs_spearman.items():
        # pick columns that exist
        xcol = next((c for c in x_candidates if c in df.columns), None)
        ycol = next((c for c in y_candidates if c in df.columns), None)
        if xcol is None or ycol is None:
            print(f"Skipped '{name}': expected columns not found.")
            continue

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df, x=xcol, y=ycol, marker='o')

        # ticks and labels
        plt.xticks(df[xcol].tolist())
        title = f"{name}: Average Spearman p-value per Oracle Position"
        plt.title(title)
        plt.xlabel("Position of oracle document")
        plt.ylabel("Spearman p-value")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.grid(True)
        plt.tight_layout()

        # annotate each point
        for x, y in zip(df[xcol], df[ycol]):
            ax.text(x, y + 0.02, f"{y:.3f}", ha="center", va="bottom", fontsize=9, color="black")

        # significance line
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1)
        ax.text(xlim[0] + 0.5, 0.055, "Significance Threshold (0.05)", color="red", fontsize=9)

        if save:
            savefig(plt, "Spearman's", name, 300)

        plt.show()


def export_table_txt(df, table_name, df_name):
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    f_path_from_dir = SCRIPT_DIR / "reports/tables" / f"{table_name}-{df_name}.txt"

    df.to_csv(
        f_path_from_dir,
        sep="\t",
        float_format="%.2f",
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
