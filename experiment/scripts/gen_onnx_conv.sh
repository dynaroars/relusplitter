#!/bin/bash

# List of conv benchmarks
benchmarks=("oval21" "collins_rul" "metaroom" "cifar2020")

# Loop through benchmarks and seeds
for benchmark in "${benchmarks[@]}"; do
    for seed in {0..4}; do
        echo "Generating ONNX model for $benchmark, seed $seed"
        python gen_onnx.py --benchmark "$benchmark" --seed "$seed" --mode conv
    done
done
