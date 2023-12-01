#!/bin/sh

git submodule init
git submodule update
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libeigen3-dev libcgal-dev libcgal-qt5-dev libboost-all-dev cmake python3-matplotlib
