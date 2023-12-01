#!/bin/sh

mkdir -p Figure2
for number_iterations in 5 10 20 30; do
    rm -rf data*;
    echo "Square1 iterations ${number_iterations}";
    ./ProjectiveFriction ./scenes/Square1/iterations_${number_iterations}.json > /dev/null;
    cp data0/output/out_000040.obj \
       Figure2/right_${number_iterations}_iterations.obj;
    mv data0 Figure2/full_sim_right_${number_iterations}_iterations
    echo "Ribbon iterations ${number_iterations}";
    ./ProjectiveFriction ./scenes/Ribbon/iterations_${number_iterations}.json > /dev/null;
    cp data0/output/out_000170.obj \
        Figure2/left_${number_iterations}_iterations.obj;
    mv data0 Figure2/full_sim_left_${number_iterations}_iterations
done
