#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
mkdir -p $LIBS_DIR
cd libs



# Download, extract, and build Marabou
wget https://github.com/NeuralNetworkVerification/Marabou/archive/refs/tags/v2.0.0.zip
unzip v2.0.0.zip
cd Marabou-2.0.0
mkdir build 
cd build

# cmake ../         # use this line to install without Gurobi
cmake ../ -DENABLE_GUROBI=ON
cmake --build ./  -j 10

# Clean up
rm -rf $LIBS_DIR/Marabou-2.0.0

echo "Marabou has been installed successfully."