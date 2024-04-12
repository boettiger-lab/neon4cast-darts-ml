#!/bin/bash

# Setting up virtual environment
python -m venv ~/.virtualenv/neon4cast-darts-ml
source ~/.virtualenv/neon4cast-darts-ml/bin/activate

# Installing software into virtual environment
pip install -r requirements.txt