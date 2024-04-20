library(dplyr)
library(tidyverse)
library(neon4cast)
library(lubridate)

remotes::install_github("eco4cast/neon4cast")

weather_stage3 <- neon4cast::noaa_stage3()

targets <- read_csv('https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz') |>
  pivot_wider(names_from = variable, values_from = observation)

site_ids <- targets |> distinct(site_id)

air_tmp <- weather_stage3 |> 
  filter(variable == "air_temperature", site_id %in% site_ids$site_id) |>
  mutate(datetime = as_date(datetime)) |>
  group_by(site_id, variable, datetime) |>
  summarize(observation = mean(prediction, na.rm = TRUE)) |>
  mutate(observation = observation - 273.15) |>
  collect() |>
  pivot_wider(names_from = variable, values_from = observation)

merged_data <- air_tmp |>
  left_join(targets, by = c("site_id", "datetime")) |>
  rename(air_tmp = air_temperature)

write_csv(merged_data, path = "aquatics-targets.csv.gz")