from utils import (
    TimeSeriesPreprocessor,
    establish_s3_connection,
)
import pandas as pd
import os
import time

# TO DO: PUT IN ARG PARSER SO THAT PEOPLE CAN SPECIFY
# S3 CONNECTION

if __name__=="__main__":
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    targets = pd.read_csv("targets.csv.gz")

    s3_client = establish_s3_connection() 

    # For the training set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date='2022-07-19',
        load_dir_name='preprocessed_train/',
        s3_client=s3_client,
        bucket_name='shared-neon4cast-darts',
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    # For the validation set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date='2023-07-19',
        load_dir_name='preprocessed_validate/',
        s3_client=s3_client,
        bucket_name='shared-neon4cast-darts',
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    
    print ("Runtime to preprocess the time series: ", time.time()-start)