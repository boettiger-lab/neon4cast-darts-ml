from plotting_utils import (
    plot_forecast,
    plot_crps_bydate,
    score_improvement_bysite,
    plot_improvement_bysite,
    plot_global_percentages,
    plot_region_percentages,
    plot_site_type_percentages_bymodel,
    plot_site_type_percentages_global,
    plot_window_and_sitetype_performance,
    generate_metadata_df,
    plot_crps_over_time_agg,
)
import pandas as pd
from utils import (
    establish_s3_connection,
    NaiveEnsembleForecaster
)
from s3_utils import (
    download_df_from_s3,
)
import warnings
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", default='shared-neon4cast-darts', type=str,
                    help="Bucket name to connect to.")
parser.add_argument("--endpoint", default='https://minio.carlboettiger.info', type=str,
                    help="S3 Endpoint.")
parser.add_argument("--accesskey", default='credentials.json', type=str,
                    help="JSON file with access key for bucket (if required).")
args = parser.parse_args()

# Ignore all warnings; admittedly not the best practice here :/
warnings.filterwarnings("ignore")

targets = pd.read_csv("targets.csv.gz")
metadata = generate_metadata_df()

# Make s3 bucket connection
try:
    s3_client = establish_s3_connection(
        endpoint=args.endpoint,
        json_file=args.accesskey,
    )
    s3_dict = {'client': s3_client, 'bucket': args.bucket}
except:
    s3_client = None

# Lists and dictionaries to store dataframes and 
best_performers_dfs = {}
best_models_listform = {}
target_variables = ['oxygen', 'temperature', 'chla']
model_names = [
    "BlockRNN", "Transformer", "NBEATS", 
    "TCN", "RNN", "TFT", 
    "NLinear", "DLinear", "NaiveEnsemble",
]

# Read csv's and store into best_performers_df
for target_variable in target_variables:
    best_performers_dfs[target_variable] = {}
    for pos in ['inter', 'intra']:
        # Allowing ability to use remote or local
        if s3_client:
            best_performers_dfs[target_variable][pos] = download_df_from_s3(
                f'dataframes/{target_variable}_{pos}_all.csv', 
                s3_dict=s3_dict,
            )
        else:
            best_performers_dfs[target_variable][pos] = pd.read_csv(
                f'dataframes/{target_variable}_{pos}_all.csv'
            )
        if pos == 'inter':
            df = best_performers_dfs[target_variable][pos]
            best_models_listform[target_variable] = [
                [model, int(df[df['model'] == model]['model_id'].unique())] for model in model_names
            ]

# 
for target_variable in target_variables:
    # These plots incorporate forecasts from all models
    plot_crps_over_time_agg(
        best_performers_dfs[target_variable]['intra'], 
        target_variable,
        png_name=f'intra_historical_{target_variable}',
    )
    plot_crps_over_time_agg(
        best_performers_dfs[target_variable]['intra'], 
        target_variable,
        historical=False,
        png_name=f'intra_naive_{target_variable}',
    )
    plot_global_percentages(
        best_performers_dfs[target_variable]['inter'], 
        f'Global Performance -- {target_variable}',
        png_name=f'global_historical_{target_variable}'
    )
    plot_global_percentages(
        best_performers_dfs[target_variable]['inter'], 
        f'Global Performance -- {target_variable}',
        historical=False,
        png_name=f'global_naive_{target_variable}'
    )
    plot_site_type_percentages_global(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        png_name=f'type_global_historical_{target_variable}'
    )
    plot_site_type_percentages_global(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        historical=False,
        png_name=f'type_global_naive_{target_variable}'
    )
    plot_site_type_percentages_bymodel(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        png_name=f'type_local_historical_{target_variable}'
    )
    plot_site_type_percentages_bymodel(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        historical=False,
        png_name=f'type_local_naive_{target_variable}'
    )
    plot_window_and_sitetype_performance(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        png_name=f'date_type_global_historical_{target_variable}'
    )
    plot_window_and_sitetype_performance(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        historical=False,
        png_name=f'date_type_global_naive_{target_variable}'
    )
    plot_region_percentages(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        png_name=f'region_global_historical_{target_variable}'
    )
    plot_region_percentages(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'{target_variable}',
        historical=False,
        png_name=f'region_global_naive_{target_variable}'
    )
    plot_improvement_bysite(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'Global Performance across sites -- {target_variable}',
        png_name=f'global_loc_historical_{target_variable}'
    )
    plot_improvement_bysite(
        best_performers_dfs[target_variable]['inter'], 
        metadata, 
        f'Global Performance across sites -- {target_variable}',
        historical=False,
        png_name=f'global_loc_naive_{target_variable}'
    )

    # These plots look at individual models

    df = best_performers_dfs[target_variable]['inter']
    for model in model_names:
        score_df = df[df['model'] == model]
        plot_improvement_bysite(
            score_df, 
            metadata, 
            f'{model} Performance -- {target_variable}',
            png_name=f'{model}_historical_{target_variable}'
        )
        plot_improvement_bysite(
            score_df, 
            metadata, 
            f'{model} Performance -- {target_variable}',
            historical=False,
            png_name=f'{model}_naive_{target_variable}'
        )

