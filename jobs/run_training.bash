#!/bin/bash

./jobs/create_logs_dir.bash

./jobs/hierarch_train_across.bash oxygen 0
./jobs/hierarch_train_across_nocovs.bash oxygen 0

./jobs/hierarch_train_across.bash temperature 1
./jobs/hierarch_train_across_nocovs.bash temperature 1

./jobs/hierarch_train_across.bash chla 1
./jobs/hierarch_train_across_nocovs.bash chla 1