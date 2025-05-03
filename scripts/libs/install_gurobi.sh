#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
mkdir -p $LIBS_DIR
cd  $LIBS_DIR




export GRB_LICENSE_PATH="$LIBS_DIR/gurobi.lic"
export INSTALL_DIR=/opt
# intall gurobi
wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz
sudo tar xvfz gurobi11.0.3_linux64.tar.gz -C $INSTALL_DIR
cd $INSTALL_DIR/gurobi1103/linux64/src/build
sudo make
sudo cp libgurobi_c++.a ../../lib/

# create symlink if 95.so is not present
if [ ! -e $INSTALL_DIR/gurobi1103/linux64/lib/libgurobi95.so ]; then
  sudo ln -s $INSTALL_DIR/gurobi1103/linux64/lib/libgurobi110.so $INSTALL_DIR/gurobi1103/linux64/lib/libgurobi95.so
fi

GUROBI_HOME="/opt/gurobi1103/linux64"

# Add Gurobi environment variables to ~/.bashrc
{
  echo ""
  echo "# Gurobi environment setup"
  echo "export GUROBI_HOME=\"$GUROBI_HOME\""
  echo "export PATH=\"\${PATH}:\${GUROBI_HOME}/bin\""
  echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:\${GUROBI_HOME}/lib\""
  echo "export GRB_LICENSE_FILE=\"$GRB_LICENSE_PATH\""
} >> ~/.bashrc

# Apply the changes immediately for the current session
export GUROBI_HOME="$GUROBI_HOME"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="$GRB_LICENSE_PATH"


# Clean up
rm -rf $LIBS_DIR/gurobi11.0.3_linux64.tar.gz
echo "Gurobi has been installed successfully."
