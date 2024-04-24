from darts.utils.likelihood_models import (
    QuantileRegression
)
from utils import (
    BaseForecaster, 
    TimeSeriesPreprocessor,
    handle_nn_architecture,
    establish_s3_connection,
)
import argparse
import pandas as pd
import time
import sys
import os
import copy
import json
import yaml
import numpy as np
from datetime import datetime, timedelta

start = time.time()


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="BlockRNN", type=str,
                    help="Specify which Darts model to train with.")
parser.add_argument("--target", default="oxygen", type=str,
                    help="Specify which target time series to train on "+\
                    "[oxygen, temperature, chla].")
parser.add_argument("--site", default="BARC", type=str,
                    help="Denotes which site to use.")
parser.add_argument("--epochs", default=200, type=int, 
                    help="The number of epochs to train a model for.")
parser.add_argument("--nocovs", default=False, action="store_true",
                    help="This nullifies the use of the other target time series "+\
                    "at that site for covariates.")
parser.add_argument("--verbose", default=False, action="store_true",
                    help="An option to use if more verbose output is desired "+\
                    "while training.")
parser.add_argument("--test", default=True, action="store_false",
                    help="This boolean flag if called will stop hyperparameters "+\
                    "from being saved.")
parser.add_argument("--device", default=0, type=int,
                    help="Specify which GPU device to use [0,1].")
parser.add_argument("--bucket", default='shared-neon4cast-darts', type=str,
                    help="Bucket name to connect to.")
parser.add_argument("--endpoint", default='https://minio.carlboettiger.info', type=str,
                    help="S3 Endpoint.")
parser.add_argument("--accesskey", default='credentials.json', type=str,
                    help="JSON file with access key for bucket (if required).")
parser.add_argument("--tblog", default=False, action="store_true",
                    help="Flag to make tensorboard logs or not.")
args = parser.parse_args()

# For non-quantile regression, add 2 CL flags, one to store true another
# to say which non-quantile regression to use, also need to save these differently

# Need to flag to say forecast didn't use covariates; also need to be careful with
# time axis encoder here, need to save these differently
if __name__ == "__main__":
    try:
        s3_client = establish_s3_connection(
            endpoint=args.endpoint,
            json_file=args.accesskey,
        )
        s3_dict = {'client': s3_client, 'bucket': args.bucket}
    except:
        s3_client, s3_dict = None, None
        
    # Selecting the device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # Accessing the validation split date from targets csv
    targets = pd.read_csv("aquatics-targets.csv.gz")
    most_recent_date_str = np.sort(targets['datetime'].unique())[-1]
    most_recent_date = datetime.strptime(most_recent_date_str, '%Y-%m-%d')
    one_year_before = most_recent_date - timedelta(days=365)
    validation_split_date = one_year_before.strftime('%Y-%m-%d')
    
    # Loading hyperparameters
    hyperparams_loc = f"hyperparameters/train/{args.target}/{args.model}"
    with open(f"{hyperparams_loc}.yaml") as f:
        hyperparams_dict = yaml.safe_load(f)
    # To make the forecast probabilistic, we the train the models
    # to perform quantile regression
    model_likelihood = {"likelihood": QuantileRegression([0.01, 0.05, 0.1, 
                                                          0.3, 0.5, 0.7, 
                                                          0.9, 0.95, 0.99])}
    
    # Using data as covariates besides the target series
    covariates_list = ["air_tmp", "chla", "temperature", "oxygen"]
    covariates_list.remove(args.target)
    if args.nocovs:
        covariates_list = None

    # Loading data preprocessors for training and validation
    preprocessors = []
    for suffix in ['train', 'validate']:
        split_date = most_recent_date_str if suffix == 'validate' \
                        else validation_split_date
        preprocessor = TimeSeriesPreprocessor(
            input_csv_name = 'aquatics-targets.csv.gz',
            load_dir_name = f"preprocessed_{suffix}/",
            s3_dict=s3_dict,
            validation_split_date=split_date,
        )
        preprocessor.load(args.site)
        preprocessors.append(preprocessor)

    if preprocessors[0].site_missing_variables != \
         preprocessors[1].site_missing_variables:
        print("Missing data edge case. Training not performed.")
        sys.exit()
    
    output_name = f"{args.site}/{args.target}/{args.model}"
    
    # Instantiating the model
    extras = {"epochs": args.epochs,
              "verbose": args.verbose,}

    nn_args = handle_nn_architecture(args.model)

    for i, nn_arg in enumerate(nn_args):
        model_hyperparameters = {
            **hyperparams_dict["model_hyperparameters"],
            **nn_arg,
        }
        
        forecaster = BaseForecaster(
            model=args.model,
            target_variable=args.target,
            train_preprocessor=preprocessors[0],
            validate_preprocessor=preprocessors[1],
            covariates_names=covariates_list,
            output_name=f"{output_name}/model_{i}/",
            validation_split_date=validation_split_date,
            model_hyperparameters=model_hyperparameters,
            model_likelihood=model_likelihood,
            site_id=args.site,
            s3_dict=s3_dict,
            log_tensorboard=args.tblog,
            **extras,
        )
        
        if not args.test:
            forecaster.output_csv_name = None
                
        forecaster.make_forecasts()
    
        # For organizational purposes, saving information about the model
        # in a log directory where forecast csv is outputtred
        if args.test:
            log_directory = f"forecasts/{args.site}/{args.target}/{args.model}/logs/"
            # REVISIT THIS BLOCK
            if not s3_client:
                if not os.path.exists(log_directory):
                    os.makedirs(log_directory)

            csv_title = forecaster.output_name.split("/")[-1].split(".")[0]
            log_file_name = log_directory + csv_title + f"model_{i}.yaml"
    
            hyperparams = {"model_hyperparameters": forecaster.hyperparams, 
                           "model_likelihood": forecaster.model_likelihood,
                           "epochs": args.epochs}
            if not s3_client:
                yaml.dump(
                    hyperparams, 
                    log_file_name, 
                    default_flow_style=False,
                )
            else:
                yaml_content = yaml.dump(hyperparams)
                s3_client.put_object(
                    Body=yaml_content, 
                    Bucket=args.bucket, 
                    Key=log_file_name,
                )
    
    print(f"Runtime for {args.model} on {args.target} at {args.site}: {(time.time() - start)/60:.2f} minutes")