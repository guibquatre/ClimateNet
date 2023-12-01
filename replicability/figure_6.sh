#!/bin/sh

mkdir -p Figure6
rm -rf data*
./ProjectiveFriction ./scenes/Square1/iterations_300.json > log_alart
./alart_curnier_data_processing.py log_alart

mv log_alart data0
mv data0 Figure6/full_sim
