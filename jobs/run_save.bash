#!/bin/bash

export PYTHONPATH=~/neon4cast-darts-ml/:$PYTHONPATH

python save_score_dfs.py # Add CL flags here if using different storage

python save_plots.py # Add CL flags here if using different storage

