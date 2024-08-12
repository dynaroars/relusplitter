#!/bin/bash

# export MKL_SERVICE_FORCE_INTEL=1
cmd="$CONDA_HOME/envs/marabou/bin/Marabou  $@"
echo $cmd
$cmd
