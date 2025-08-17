from openai import OpenAI
from slt_positional_bias.dataset import load_and_sort_data, calc_metric
from slt_positional_bias.features import shorten_documents
from slt_positional_bias.modeling.predict import run_pipeline

API_KEY = "glpat-r_ZnnQU3cwc59xQ7M2XP"
API_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"

client = OpenAI(api_key=API_KEY, base_url=API_URL)

df_sorted = load_and_sort_data(1, 39)
df_cleaned = shorten_documents(df_sorted, client, n_to_clean=2240)

df_10 = run_pipeline(df_cleaned, client, [0, 2, 4, 7, 9], 1, 9)
df_20 = run_pipeline(df_cleaned, client, [0, 4, 9, 14, 19], 1, 19)
df_30 = run_pipeline(df_cleaned, client, [0, 7, 14, 22, 29], 1, 29)
df_40 = run_pipeline(df_cleaned, client, [0, 9, 19, 29, 39], 1, 39)

df_10_norm = df_10.copy(deep=True)
df_20_norm = df_20.copy(deep=True)
df_30_norm = df_30.copy(deep=True)
df_40_norm = df_40.copy(deep=True)

col = "Position of Oracle"
mapping_10 = {1: 0.00, 3: 0.25, 5: 0.50, 8: 0.75, 10: 1.00}
mapping_20 = {1: 0.00, 5: 0.25, 10: 0.50, 15: 0.75, 20: 1.00}
mapping_30 = {1: 0.00, 8: 0.25, 15: 0.50, 23: 0.75, 30: 1.00}
mapping_40 = {1: 0.00, 10: 0.25, 20: 0.50, 30: 0.75, 40: 1.00}

df_10_norm[col] = df_10_norm[col].astype(int).replace(mapping_10)
df_20_norm[col] = df_20_norm[col].astype(int).replace(mapping_20)
df_30_norm[col] = df_30_norm[col].astype(int).replace(mapping_30)
df_40_norm[col] = df_40_norm[col].astype(int).replace(mapping_40)

dfs = {'df_10': df_10, 'df_20':df_20, 'df_30':df_30, 'df_40':df_40}
dfs_norm = {'df_10': df_10_norm, 'df_20':df_20_norm, 'df_30':df_30_norm, 'df_40':df_40_norm}

calc_metric(dfs, True, False)
calc_metric(dfs_norm, True, True)