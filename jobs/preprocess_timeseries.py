from utils import (
    TimeSeriesPreprocessor,
    establish_s3_connection,
)
import pandas as pd
import os
import time
import argparse
import numpy as np
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", default='shared-neon4cast-darts', type=str,
                    help="Bucket name to connect to.")
parser.add_argument("--endpoint", default='https://minio.carlboettiger.info', type=str,
                    help="S3 Endpoint.")
parser.add_argument("--accesskey", default='credentials.json', type=str,
                    help="JSON file with access key for bucket (if required).")
parser.add_argument("--device", default=0, type=int,
                    help="Specify which GPU device to use [0,1].")
args = parser.parse_args()

if __name__=="__main__":
    start = time.time()

    # Flag which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    targets = pd.read_csv("aquatics-targets.csv.gz")

    # Make s3 bucket connection
    try:
        s3_client = establish_s3_connection(
            endpoint=args.endpoint,
            json_file=args.accesskey,
        )
        s3_dict = {'client': s3_client, 'bucket': args.bucket}
    except:
        s3_dict = {'client': None, 'bucket': None}

    # Defining the training data split date to be one year before the most
    # date time in aquatics-targets.csv.gz. Validation split date will be
    # the most recent date.
    most_recent_date_str = np.sort(targets['datetime'].unique())[-1]
    most_recent_date = datetime.strptime(most_recent_date_str, '%Y-%m-%d')
    one_year_before = most_recent_date - timedelta(days=365)
    one_year_before_str = one_year_before.strftime('%Y-%m-%d')

    # For the training set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date=one_year_before_str,
        load_dir_name='preprocessed_train/',
        s3_dict=s3_dict,
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    # For the validation set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date=most_recent_date_str,
        load_dir_name='preprocessed_validate/',
        s3_dict=s3_dict,
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    
    print('\n' + f"Runtime: {(time.time() - start_)/60:.2f} minutes")