#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
ENVS_DIR=$(realpath "$SCRIPT_DIR/../../.envs")
TOOL_NAME="abcrown"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/abcrown.yaml


mkdir -p $LIBS_DIR
cd  $LIBS_DIR


# Install IBM CPLEX >= 22.1.0
# Download from https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
# place the downloaded file in the libs folder
cd  $LIBS_DIR
chmod +x cplex_studio2210.linux_x86_64.bin  # Any version >= 22.1.0 should work. Change executable name here.
# You can directly run the installer: ./cplex_studio2210.linux_x86_64.bin; the response.txt created below is for non-interactive installation.
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2210.linux_x86_64.bin -f response.txt
# Build the C++ code for CPLEX interface. Assuming we are still inside the alpha-beta-CROWN folder.
sudo apt install build-essential  # A modern g++ (>=8.0) is required to compile the code.
# Change CPX_PATH in complete_verifier/cuts/CPLEX_cuts/Makefile if you installed CPlex to a non-default location, like inside your home folder.


# install abcrown
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
cd alpha-beta-CROWN
git checkout dc32df038440a9726e97547b88f9913743773e7f
git submodule init
git submodule update --init --recursive


# Remove the old environment, if necessary.
# conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
# conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda env remove --prefix $CONDA_PREFIX 
conda env create --prefix $CONDA_PREFIX -f $ENV_FILE_PATH 
make -C complete_verifier/cuts/CPLEX_cuts/
