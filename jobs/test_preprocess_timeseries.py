import sys
sys.path.append("/home/rstudio/neon4cast-darts-ml/")

from utils import (
    TimeSeriesPreprocessor,
    establish_s3_connection,
)
import pandas as pd
import os
from darts import TimeSeries
import numpy as np
import time

if __name__=="__main__":
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    targets = pd.read_csv("targets.csv.gz")

    s3_client = establish_s3_connection() 

    # For the training set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date='2022-07-19',
        load_dir_name='preprocessed_train/',
        s3_client=s3_client,
        bucket_name='shared-neon4cast-darts',
    )
    
    data_preprocessor.preprocess_data('FLNT')
    
    data_preprocessor.save()

    #del data_preprocessor
#
    #data_preprocessor = TimeSeriesPreprocessor(
    #    validation_split_date='2022-07-19',
    #    load_dir_name='preprocessed_train/',
    #)
    #
    #data_preprocessor.load('FLNT')
#
    #
    #print ("Runtime to preprocess the time series: ", time.time()-start)