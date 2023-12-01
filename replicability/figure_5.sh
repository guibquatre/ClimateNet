#!/bin/sh

mkdir -p Figure5
rm -rf data*
./ProjectiveFriction ./scenes/Analytic/analytic.json > log_analytic
./analytic_example_data_processing.py log_analytic

mv log_analytic data0
mv data0 Figure5/full_sim
