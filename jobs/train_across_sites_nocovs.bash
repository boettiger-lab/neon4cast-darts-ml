#!/bin/bash

sites=("ARIK" "BARC" "BIGC" "BLDE" "BLUE" "BLWA" "CARI" 
       "COMO" "CRAM" "CUPE" "FLNT" "GUIL" "HOPB" "KING" 
       "LECO" "LEWI" "LIRO" "MART" "MAYF" "MCDI" "MCRA" 
       "OKSR" "POSE" "PRIN" "PRLA" "PRPO" "REDB" "SUGG" 
       "SYCA" "TECR" "TOMB" "TOOK" "WALK" "WLOU")

# Training the model specified at CL at every site
for site in "${sites[@]}"; do
  > "logs/${site}/$2/train_$1.log"
  python -u train_tune.py --model $1 --target $2 --site "$site" \
    --epochs 200 --device $3 --nocovs &> "logs/${site}/$2/train_$1.log"
done