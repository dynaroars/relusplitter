#!/bin/bash

# List of fc benchmarks
benchmarks=("resnet_a")
# benchmarks=("acasxu" "mnist_fc" "reach_prob" "rl_benchmarks")

# Loop through benchmarks and seeds
for benchmark in "${benchmarks[@]}"; do
    for seed in {0..4}; do
        echo "Generating ONNX model for $benchmark, seed $seed"
        python gen_onnx.py --benchmark "$benchmark" --seed "$seed" --mode fc
    done
done
