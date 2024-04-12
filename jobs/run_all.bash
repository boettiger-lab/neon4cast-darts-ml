#!/bin/bash

./jobs/create_logs_dir.bash

# ./jobs/get_targets.bash This is only for submission repo

python jobs/preprocess_timeseries.py

./jobs/run_training.bash

python save_score_dfs.py # Add CL flags here if using different storage

python save_plots.py # Add CL flags here if using different storage

