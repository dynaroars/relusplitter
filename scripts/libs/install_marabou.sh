#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
mkdir -p $LIBS_DIR
cd  $LIBS_DIR

echo "===> Installing Dependencies"
sudo apt update -y 
sudo apt install -y gfortran python3-dev


# Download, extract, and build Marabou
wget -nc https://github.com/NeuralNetworkVerification/Marabou/archive/refs/tags/v2.0.0.zip 
unzip -o v2.0.0.zip
cd Marabou-2.0.0

# change the threshold as it might help with unsound results
echo "===> Changing the threshold to 0.0000000001"
echo "Before:"
sed -n '81p' src/configuration/GlobalConfiguration.cpp
sed -i '81s|.*|const double GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD = 0.0000000001;|' src/configuration/GlobalConfiguration.cpp
echo "After:"
sed -n '81p' src/configuration/GlobalConfiguration.cpp


mkdir build 
cd build

# cmake ../         # use this line to install without Gurobi
cmake ../ -DENABLE_GUROBI=ON
cmake --build ./  -j 10
