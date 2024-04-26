import pandas as pd
import numpy as np
from darts.models import StatsForecastAutoTheta
from typing import Optional
import os
from datetime import datetime, timedelta
from s3_utils import (
    upload_df_to_s3,
)

class AutoThetaForecaster():
    def __init__(self,
                 validate_preprocessor: Optional = None,
                 target_variable: Optional[str] = None,
                 season_length: Optional[int] = 1,
                 decomposition_type: Optional[str] = 'additive',
                 input_chunk_length: Optional[int] = 31,
                 output_name: Optional[str] = "default",
                 forecast_horizon: Optional[int] = 30,
                 site_id: Optional[str] = None,
                 num_samples: Optional[int] = 500,
                 targets_csv: Optional[str] = "aquatics-targets.csv.gz",
                 s3_dict: Optional[dict] = {'client': None, 'bucket': None},
                 ):
        self.validate_preprocessor = validate_preprocessor
        self.s3_dict = s3_dict
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.input_chunk_length = input_chunk_length
        self.target_variable = target_variable
        self.output_name = output_name
        self.forecast_horizon = forecast_horizon
        self.site_id = site_id
        self.num_samples = num_samples
        self.targets = pd.read_csv(targets_csv)
        most_recent_date_str = np.sort(self.targets['datetime'].unique())[-1]
        most_recent_date = datetime.strptime(most_recent_date_str, '%Y-%m-%d')
        one_year_before = most_recent_date - timedelta(days=365)
        self.split_date = one_year_before.strftime('%Y-%m-%d')

        self.inputs = self._preprocess_data(validate_preprocessor)
    
        if not s3_dict['client']:
            # Handling csv names and directories for the final forecast
            if not os.path.exists(f"forecasts/{args.site}/{args.target}/"):
                os.makedirs(f"forecasts/{args.site}/{args.target}/")
        
    def _preprocess_data(self, data_preprocessor):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept; train_set flag is to 
        """
        stitched_series_dict = data_preprocessor.sites_dict[self.site_id]

        # If there was failure when filtering then we can't do preprocessing
        if self.target_variable in \
                data_preprocessor.site_missing_variables:
            raise ValueError("Cannot fit this target time series as no GP fit was performed.")
        inputs = stitched_series_dict[self.target_variable]
            
        return inputs.median()

    def get_predict_set(self, input_chunk_length):
        # Similar to get_validation_set, except here I want to create 
        # a window that just has the data to use as an input, nothing to validate
        # a prediction
        interval = pd.Timedelta(days=self.forecast_horizon)
        dates = pd.date_range(self.split_date, periods=12, freq=interval)
        predict_set_list = []
        for date in dates:
            predict_set_list.append(
                self.inputs.slice_n_points_before(
                    date, 
                    input_chunk_length
                )
            )
        return predict_set_list
        
    def make_forecasts(self):
        """
        This function fits a Darts model to the training_set
        """

        self.model = StatsForecastAutoTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
        )
        predict_series = self.get_predict_set(self.input_chunk_length)

        predictions = []
        for series in predict_series:
            self.model.fit(series)
            pred = self.model.predict(
                n=self.forecast_horizon, 
                num_samples=self.num_samples
            )
            predictions.append(pred)

        for prediction in predictions:
            csv_name = 'forecasts/' + self.output_name + \
                           prediction.time_index[0].strftime('%Y_%m_%d.csv')
            df = prediction.pd_dataframe(suppress_warnings=True)
            if self.s3_dict['client']:
                upload_df_to_s3(csv_name, df, self.s3_dict)
            else:
                df.to_csv(csv_name)