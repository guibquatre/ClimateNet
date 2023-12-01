#!/bin/sh

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPRINT_ALART_CURNIER_ERROR=ON ../..
make
cd ..
ln -sf build/ProjectiveFriction
