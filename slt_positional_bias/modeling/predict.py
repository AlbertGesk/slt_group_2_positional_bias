from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from slt_positional_bias.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def run_inference_on_topic(df_topic, client, positions, rel3_count, rel0_count, topic_id):
    results = []
    
    rel_3_docs = df_topic[df_topic['rel_scoring'] == 3].head(rel3_count)
    rel_0_docs = df_topic[df_topic['rel_scoring'] == 0].head(rel0_count)

    query = rel_3_docs['topic'].iloc[0]
    rel_3_text = ' '.join(rel_3_docs['doc'].tolist())
    rel_0_texts = rel_0_docs['doc'].tolist()

    for pos in positions:
        text_list = rel_0_texts.copy()
        if pos >= len(text_list):
            text_list.append(rel_3_text)
        else:
            text_list.insert(pos, rel_3_text)

        context_string = ' '.join(text_list)

        user_prompt = f"""
        Context:
        {context_string}

        Question:
        {query}

        Answer:"""
        system_prompt = "You are a helpful assistant answering a question based on retrieved context information."

        response = client.chat.completions.create(
            model= "1 - Llama3 405 the best general model and big context size",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        answer = response.choices[0].message.content.strip()

        results.append({
            'topic_id': topic_id,
            'topic' : query,
            'rel_3_doc_position': pos,
            'nr_rel_3_doc': rel3_count,
            'nr_rel_0_doc': rel0_count,
            'oracle': rel_3_text,
            'answer': answer
        })

    return results

def run_pipeline(df_final, client, positions, rel3_count, rel0_count, max_qids=56):
    all_results = []
    topic_ids = df_final['topic_id'].unique()[:max_qids]
    for topic_id in topic_ids:
        df_topic = df_final[df_final['topic_id'] == topic_id]
        results = run_inference_on_topic(df_topic, client, positions, rel3_count, rel0_count, topic_id)
        all_results.extend(results)
    return pd.DataFrame(all_results)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
