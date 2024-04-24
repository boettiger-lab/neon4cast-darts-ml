from stats_utils import AutoThetaForecaster
from utils import (
    TimeSeriesPreprocessor,
    establish_s3_connection,
)
import pandas as pd
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--site", default="BARC", type=str,
                    help="Denotes which site to use.")
parser.add_argument("--target", default="oxygen", type=str,
                    help="Specify which target time series to train on "+\
                    "[oxygen, temperature, chla].")
parser.add_argument("--bucket", default='shared-neon4cast-darts', type=str,
                    help="Bucket name to connect to.")
parser.add_argument("--endpoint", default='https://minio.carlboettiger.info', type=str,
                    help="S3 Endpoint.")
parser.add_argument("--accesskey", default='credentials.json', type=str,
                    help="JSON file with access key for bucket (if required).")
args = parser.parse_args()

start = time.time()

if __name__ == "__main__":
    try:
        s3_client = establish_s3_connection(
            endpoint=args.endpoint,
            json_file=args.accesskey,
        )
        s3_dict = {'client': s3_client, 'bucket': args.bucket}
    except:
        s3_client, s3_dict = None, None

    # Accessing the validation split date from targets csv
    targets = pd.read_csv("aquatics-targets.csv.gz")
    most_recent_date_str = np.sort(targets['datetime'].unique())[-1]
    most_recent_date = datetime.strptime(most_recent_date_str, '%Y-%m-%d')
    one_year_before = most_recent_date - timedelta(days=365)
    validation_split_date = one_year_before.strftime('%Y-%m-%d')
    
    # Loading data preprocessors for training and validation
    # Note that having different preprocessors is not required
    # for the Theta model, but I am still loading both as to
    # see what site to train on.
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

    output_name = f"{args.site}/{args.target}/AutoTheta/model_0/"

    model = AutoThetaForecaster(
        validate_preprocessor=preprocessors[1],
        target_variable=args.target,
        site_id=args.site,
        output_name=output_name,
        s3_dict=s3_dict,
    )
    model.make_forecasts()

    print(f"Runtime for AutoTheta on {args.target} at {args.site}: {(time.time() - start)/60:.2f} minutes")