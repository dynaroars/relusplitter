# Computes the neuron composition statistics for each benchmark
# we only use the 1st layer.

import os
import sys
import signal

from tqdm import tqdm
from pathlib import Path
from itertools import product
import multiprocessing as mp

from helpers import *
from settings import *

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier




for benchmark in ["acasxu", "mnist_fc"]:
    # Load the benchmark
    benchmark = benchmarks[benchmark]
    instances = get_instances(benchmark)
    composition_info = list(map(lambda i: ReluSplitter.info_s(i[0], i[1])[0], instances))
    # print(composition_info[0].keys())

    print(f"Neuron Compostion Report: {benchmark['name']}")
    for neuron_type in ["stable+", "stable-", "unstable"]:
        counts = [info[neuron_type] for info in composition_info]
        # print(counts)
        counts.sort()
        # number of instances
        print(f"Neuron Type: {neuron_type}")
        print(f"Total Instances: {len(instances)}")
        # max, min, mean, median
        print(f"Mean: {sum(counts)/len(counts)}")
        print(f"25th percentile: {counts[len(counts)//4]}")
        print(f"Min: {min(counts)}")
        print(f"Median: {counts[len(counts)//2]}")
        print(f"75th percentile: {counts[3*len(counts)//4]}")
        print(f"Max: {max(counts)}")
        print("\n")
    print("=====================================")


for benchmark in ["tll"]:
    # Load the benchmark
    benchmark = benchmarks[benchmark]
    instances = get_instances(benchmark)
    composition_info = list(map(lambda i: ReluSplitter.info_s(i[0], i[1])[0], instances))
    # print(composition_info[0].keys())

    print(f"Neuron Compostion Report: {benchmark['name']}")
    for neuron_type in ["stable+", "stable-", "unstable"]:
        # use percentage for tll as the number of neurons in each instance is different
        counts = [info[neuron_type]/info["all"] for info in composition_info]
        # print(counts)
        counts.sort()
        # number of instances
        print(f"Neuron Type: {neuron_type}")
        print(f"Total Instances: {len(instances)}")
        # max, min, mean, median
        print(f"Mean: {sum(counts)/len(counts)}")
        print(f"25th percentile: {counts[len(counts)//4]}")
        print(f"Min: {min(counts)}")
        print(f"Median: {counts[len(counts)//2]}")
        print(f"75th percentile: {counts[3*len(counts)//4]}")
        print(f"Max: {max(counts)}")
        print("\n")
    print("=====================================")


