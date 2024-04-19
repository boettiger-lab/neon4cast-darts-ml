#!/bin/bash

sites=("ARIK" "BARC" "BIGC" "BLDE" "BLUE" "BLWA" "CARI" 
       "COMO" "CRAM" "CUPE" "FLNT" "GUIL" "HOPB" "KING" 
       "LECO" "LEWI" "LIRO" "MART" "MAYF" "MCDI" "MCRA" 
       "OKSR" "POSE" "PRIN" "PRLA" "PRPO" "REDB" "SUGG" 
       "SYCA" "TECR" "TOMB" "TOOK" "WALK" "WLOU")

# Training the model specified at CL at every site
for site in "${sites[@]}"; do
  > "logs/${site_}/$2/train_AutoTheta.log"
  python predict_stats.py --site "$site" --target $1 \
      &> "logs/${site}/$1/train_AutoTheta.log"
done