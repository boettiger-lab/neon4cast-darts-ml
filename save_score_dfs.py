from plotting_utils import (
    score_improvement_bysite,
)
import pandas as pd
from utils import (
    establish_s3_connection,
    NaiveEnsembleForecaster
)
from s3_utils import (
    upload_df_to_s3,
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


def return_best_id(df):
    '''
    This function returns the best performing id from scoring DataFrames
    '''
    medians = df.groupby('model_id')['skill_historical_ml_crps'].median()
    best_median_id = medians.idxmax()

    return best_median_id

# Ignore all warnings; admittedly not the best practice here :/
warnings.filterwarnings("ignore")
# Note might want to change this to updated csv
targets = pd.read_csv("targets.csv.gz")

try:
    # Make s3 bucket connection
    s3_client = establish_s3_connection(
        endpoint=args.endpoint,
        json_file=args.accesskey,
    )
except:
    s3_client = None

# Collect the forecasts made with Darts Models, and store 
# in scores_dict
model_names = [
    "BlockRNN", "Transformer", 
    "NBEATS", "TCN", "RNN", "TFT", 
    "NLinear", "DLinear",
]
target_variables = ["oxygen", "temperature", "chla"]
ids = [0, 1, 2, 3, 4]
s3_dict = {'client': s3_client, 'bucket': args.bucket}
scores_dict = {}
for model in model_names:
    scores_dict[model] = {}
    for id_ in ids:
        # Did not train more than 2 [.]Linear models, so need to
        # treat these separately
        if model == "NLinear" or model == "DLinear":
            if id_ > 1:
                continue
        scores_dict[model][id_] = {}
        for target_variable in target_variables:
            inter_merged, intra_merged = score_improvement_bysite(
                model,
                id_,
                targets, 
                target_variable, 
                s3_dict=s3_dict,
            )
            scores_dict[model][id_][target_variable] = {}
            scores_dict[model][id_][target_variable]['inter'] = inter_merged
            scores_dict[model][id_][target_variable]['intra'] = intra_merged

# With all the individual forecast score_dicts, we will now
# concatanate them all
global_dfs = {}
for target_variable in target_variables:
    global_dfs[target_variable] = {}
    for pos in ['inter', 'intra']:
        concat_list = []
        for model in model_names:
            for id_ in ids:
                if model == "DLinear" or model == "NLinear":
                    if id_ > 1:
                        continue
                concat_list.append(scores_dict[model][id_][target_variable][pos])
        global_dfs[target_variable][pos] = pd.concat(concat_list)

# We only want to use the best models, so let's find those
best_models = {}
for model in model_names:
    best_models[model] = {}
    for target_variable in target_variables:
        df_ = global_dfs[target_variable]['inter']
        df = df_[df_['model'] == model]
        best_models[model][target_variable] = return_best_id(df)

# We will collect the best performing models in one dataframe and save
best_performers_dfs = {}
for target_variable in target_variables:
    best_performers_dfs[target_variable] = {}
    for pos in ['inter', 'intra']:
        best_performers_dfs[target_variable][pos] = {}
        best_performers_dfs[target_variable][pos] = pd.concat(
            [scores_dict[model][best_models[model][target_variable]]
             [target_variable][pos] for model in model_names]
        )

# Need to get all the dates used in the different forecasts
site_date_dict = {}
for target_variable in target_variables:
    site_date_dict[target_variable] = (
        best_performers_dfs[target_variable]['inter']
        .groupby('site_id')['date']
        .unique()
        .apply(list)
        .to_dict()
    )

# Don't need to run this cell if the dataframes have been loaded
best_models_listform = {}
for target_variable in target_variables:
    best_models_listform[target_variable] = [
        [model, best_models[model][target_variable]] for model in model_names
    ]

# Making forecasts with the naive ensemble model
for target_variable in target_variables:
    sites_w_forecasts = site_date_dict[target_variable].keys()
    for site in sites_w_forecasts:
        forecasted_dates = site_date_dict[target_variable][site]
        for date in forecasted_dates:
            ensemble_model = NaiveEnsembleForecaster(
                model_list=best_models_listform[target_variable],
                site_id=site,
                target_variable=target_variable,
                output_name='model_0',
                date=date,
                s3_dict=s3_dict,
            )
            ensemble_model.make_forecasts()

# Saving these scores in a dataframe
model_names = ["NaiveEnsemble"]
ids = [0]
ne_scores = {}
for model in model_names:
    ne_scores[model] = {}
    for id_ in ids:
        ne_scores[model][id_] = {}
        for target_variable in target_variables:
            inter_merged, intra_merged = score_improvement_bysite(
                model,
                id_,
                targets, 
                target_variable, 
                s3_dict=s3_dict,
            )
            ne_scores[model][id_][target_variable] = {}
            ne_scores[model][id_][target_variable]['inter'] = inter_merged
            ne_scores[model][id_][target_variable]['intra'] = intra_merged


for target_variable in target_variables:
    best_performers_dfs[target_variable]['inter'] = pd.concat(
        [
            ne_scores['NaiveEnsemble'][0][target_variable]['inter'], 
            best_performers_dfs[target_variable]['inter']
        ], 
        ignore_index=True,
    )
    best_performers_dfs[target_variable]['intra'] = pd.concat(
        [
            ne_scores['NaiveEnsemble'][0][target_variable]['intra'], 
            best_performers_dfs[target_variable]['intra']
        ], 
        ignore_index=True,
    )

# And saving (remotel)
if s3_client:
    for target_variable in target_variables:
        for pos in ['inter', 'intra']:
            upload_df_to_s3(
                f'dataframes/{target_variable}_{pos}_all.csv', 
                best_performers_dfs[target_variable][pos], 
                s3_dict=s3_dict,
            )
else:
    if not os.path.exists('dataframes/'):
        os.makedirs('dataframes/')
    for target_variable in target_variables:
        for pos in ['inter', 'intra']:
            best_performers_dfs[target_variable][pos].to_csv(
                f'dataframes/{target_variable}_{pos}_all.csv', 
                index=False,
            )