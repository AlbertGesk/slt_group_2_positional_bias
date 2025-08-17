from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import evaluate
from openai import OpenAI
import pandas as pd

from slt_positional_bias.config import PROCESSED_DATA_DIR

app = typer.Typer()


def sacrebleu_corpus(predictions, references):
    """
    Corpus level BLEU Metric

    :param predictions: list of predictions
    :param references: list of references
    :return: return a BLEU score
    :rtype: float
    """

    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(
        predictions=[predictions],
        references=[references],
        tokenize="13a",
        lowercase=False,          # Goal of Experiment is Semantic not Syntax
        smooth_method="exp",
        use_effective_order=True)

    return round(results["score"], 1)

def rouge_corpus(predictions, references):
    """
    Corpus level ROUGE Metric

    :param predictions: list of predictions
    :param references: list of references
    :return: ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    :rtype: float
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=[predictions], references= [references])

    return round(results["rouge1"], 2), round(results["rouge2"], 2), round(results["rougeL"], 2), round(results["rougeLsum"], 2)

def meteor_corpus(predictions, references):
    """
    Compute METEOR metric

    :param predictions: list of predictions
    :param references: list of references
    :return: a meteor score
    :rtype: float
    """
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=[predictions], references= [references])

    return round(results["meteor"], 2)

def bertscore_corpus(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=[predictions], references= [references], lang="en")

    p = results["precision"][0]
    r = results["recall"][0]
    f = results["f1"][0]

    return round(p, 2), round(r, 2), round(f, 2)

def bertscore_verbose(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references= references, lang="en", verbose=True)

    p = results["precision"][0]
    r = results["recall"][0]
    f = results["f1"][0]

    return round(p, 2), round(r, 2), round(f, 2)

def shorten_documents(df: pd.DataFrame, client: OpenAI, n_to_clean: int) -> pd.DataFrame:
    df_cleaned = df.copy()
    system_prompt = (
    "You are a language model specialized in content distillation. "
    "Your goal is to reduce passages to around 200 tokens while preserving all critical information. "
    "You must avoid removing any content that could be essential to understanding the topic, its implications, or its technical depth. You should use the same vocabularies used in the original document."
    )

    for idx in tqdm(range(n_to_clean)):
        topic = df.iloc[idx]['topic']
        original_text = df.iloc[idx]['doc']
        user_prompt = f"""
        Your task is to shorten the passage to approximately 200 tokens while retaining all essential and topic-relevant information. 
        Preserve technical content, facts, processes, and insights mentioned in the text. 
        You should use the vocabularies used in the original document and don't add any new information.

        Avoid:
        - Structural or editorial elements (like tables of contents, section numbers, or headings)
        - Redundant or general-purpose filler text

        Keep:
        - All technical explanations
        - Data and findings
        - Descriptions of methods and applications

        Input passage:
        {original_text}

        Shortened version (~200 tokens, no critical information lost):
        """.strip()

        response = client.chat.completions.create(
            model="alias-large",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_output = response.choices[0].message.content.strip()
        if "</think>" in raw_output.lower():
            cleaned_text = raw_output.lower().split("</think>")[-1].strip()
        else:
            cleaned_text = raw_output
        df_cleaned.at[idx, 'doc'] = cleaned_text

    return df_cleaned

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
