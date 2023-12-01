#!/bin/sh
rm -rf data*
./ProjectiveFriction ./scenes/Square3/mu_0.1.json > /dev/null

mkdir -p Figure1
cp data0/output/out_000100.obj Figure1/left.obj
cp data0/output/out_000225.obj Figure1/middle.obj
cp data0/output/out_000258.obj Figure1/right.obj

mv data0 Figure1/full_sim
