import os
import subprocess

def get_instances(benchmark):
    instances = []
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx_path, vnnlib_path, _ = line.strip().split(',')
            instances.append((onnx_path, vnnlib_path))
    return instances
