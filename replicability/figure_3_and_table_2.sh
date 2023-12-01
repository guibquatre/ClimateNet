#!/bin/sh

rm -rf data*
# data0
echo "Ribbon mu=0.3"
./ProjectiveFriction ./scenes/Ribbon/mu_0.3.json > /dev/null

# data1
echo "Ribbon mu=0.6"
./ProjectiveFriction ./scenes/Ribbon/mu_0.6.json > /dev/null


# data2
echo "Square1 mu=0.1"
./ProjectiveFriction ./scenes/Square1/mu_0.1.json > /dev/null
# data3
echo "Square1 mu=0.3"
./ProjectiveFriction ./scenes/Square1/mu_0.3.json > /dev/null


# data4
echo "Square3 mu=0.1"
./ProjectiveFriction ./scenes/Square3/mu_0.1.json > /dev/null

# data5
echo "Square3 mu=0.3"
./ProjectiveFriction ./scenes/Square3/mu_0.3.json > /dev/null


mkdir -p Figure3
cp data0/output/out_000171.obj Figure3/row_1_left.obj
cp data1/output/out_000171.obj Figure3/row_1_right.obj
cp data2/output/out_000248.obj Figure3/row_2_left.obj
cp data3/output/out_000248.obj Figure3/row_2_right.obj
cp data4/output/out_000247.obj Figure3/row_3_left.obj
cp data5/output/out_000247.obj Figure3/row_3_right.obj

mkdir -p Table2/Ribbon/mu0.3
mkdir -p Table2/Ribbon/mu0.6
mkdir -p Table2/Square1/mu0.1
mkdir -p Table2/Square1/mu0.3
mkdir -p Table2/Square3/mu0.1
mkdir -p Table2/Square3/mu0.3

cp data0/stats.txt Table2/Ribbon/mu0.3/
cp data1/stats.txt Table2/Ribbon/mu0.6/
cp data2/stats.txt Table2/Square1/mu0.1/
cp data3/stats.txt Table2/Square1/mu0.3/
cp data4/stats.txt Table2/Square3/mu0.1/
cp data5/stats.txt Table2/Square3/mu0.3/

mv data0 Figure3/full_sim_row_1_left
mv data1 Figure3/full_sim_row_1_right
mv data2 Figure3/full_sim_row_2_left
mv data3 Figure3/full_sim_row_2_right
mv data4 Figure3/full_sim_row_3_left
mv data5 Figure3/full_sim_row_3_right
