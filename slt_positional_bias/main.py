from loguru import logger
from openai import OpenAI
from slt_positional_bias.dataset import load_and_sort_data
from slt_positional_bias.features import shorten_documents
from slt_positional_bias.modeling.predict import run_pipeline

API_KEY = "glpat-r_ZnnQU3cwc59xQ7M2XP"
API_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"

client = OpenAI(api_key=API_KEY, base_url=API_URL)

df_sorted = load_and_sort_data(1, 39)
df_cleaned = shorten_documents(df_sorted, client, n_to_clean=2240)

df_results_1 = run_pipeline(df_cleaned, client, [0, 2, 4, 7, 9], 1, 9)
df_results_2 = run_pipeline(df_cleaned, client, [0, 4, 9, 14, 19], 1, 19)
df_results_3 = run_pipeline(df_cleaned, client, [0, 7, 14, 22, 29], 1, 29)
df_results_4 = run_pipeline(df_cleaned, client, [0, 9, 19, 29, 39], 1, 39)