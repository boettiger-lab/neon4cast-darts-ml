#!/bin/bash

sites=("ARIK" "BARC" "BIGC" "BLDE" "BLUE" "BLWA" "CARI" 
       "COMO" "CRAM" "CUPE" "FLNT" "GUIL" "HOPB" "KING" 
       "LECO" "LEWI" "LIRO" "MART" "MAYF" "MCDI" "MCRA" 
       "OKSR" "POSE" "PRIN" "PRLA" "PRPO" "REDB" "SUGG" 
       "SYCA" "TECR" "TOMB" "TOOK" "WALK" "WLOU")

target_vars=("oxygen" "chla" "temperature")

for site in "${sites[@]}"; do
    for var in "${target_vars[@]}"; do
        directory="logs/${site}/${var}/"
        if [ ! -d "$directory" ]; then
            mkdir -p "$directory"
        fi
    done
done
