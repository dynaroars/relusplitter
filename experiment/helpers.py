import os
import subprocess
import sqlite3
from pathlib import Path
from multiprocessing import cpu_count

from relu_splitter.verify import init_verifier
from relu_splitter.core import ReluSplitter

TOOL_ROOT = Path(os.environ["TOOL_ROOT"])

def get_instances(benchmark):
    instances = []
    path = benchmark["path"]
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx, vnnlib, _ = line.strip().split(',')
            instances.append((path / onnx, path / vnnlib))
    return instances

def run_splitter(onnx_path, vnnlib_path, output_dir, log_dir, split_idx, strat_n_mask, min_splits, max_splits, seed, atol, rtol, lambdas = None):
    wd = os.environ["TOOL_ROOT"]
    strat, mask = strat_n_mask
    fname = f"{onnx_path.stem}~{vnnlib_path.stem}~RS~{split_idx}~{mask}~{strat}~{min_splits}~{max_splits}~{seed}"
    if lambdas:
        fname += f"~{lambdas[0]}~{lambdas[1]}"
    output_path = output_dir / f"{fname}.onnx"
    log_path  = log_dir / f"{fname}.log"

    cmd =   f"python main.py split --net {onnx_path} --spec {vnnlib_path} --output {output_path} "\
            f"--split_strategy {strat} --mask {mask} --split_idx {split_idx} "\
            f"--min_splits {min_splits} --max_splits {max_splits} --seed {seed} --atol {atol} --rtol {rtol}"
    if lambdas:
        cmd += f"  --scale_factor {lambdas[0]} {lambdas[1]}"
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, shell=True, cwd=wd, stdout=f, stderr=f)
    return ret.returncode, output_path, log_path

def get_splitter_baseline(onnx_path, output_dir, n_splits, split_idx=0, atol=1e-4, rtol=1e-4):
    wd = os.environ["TOOL_ROOT"]
    output_path = output_dir / f"{onnx_path.stem}~{n_splits}~baseline.onnx"
    cmd = f"python main.py baseline --net {onnx_path} --n_splits {n_splits} --split_idx {split_idx} --atol {atol} --rtol {rtol} --output {output_path}"
    # drop stdout and stderr
    ret = subprocess.run(cmd, shell=True, cwd=wd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return ret.returncode, output_path

# special config for verifiers
marabou_cpu = 32 if cpu_count() > 100 else 10
marabou_ram = "64G"

exp_root                    = TOOL_ROOT / "experiment"
abcrown_acasxu_config       = f"{exp_root}/config/acasxu.yaml"
abcrown_mnist_x2_config     = f"{exp_root}/config/mnistfc_small.yaml"
abcrown_mnist_x4x6_config   = f"{exp_root}/config/mnistfc.yaml"
abcrown_tll_config          = f"{exp_root}/config/tllVerifyBench.yaml"


def get_verification_config(verifier, benchmark, onnx, vnnlib, log_path, timeout):
    conf = {
        "onnx_path": onnx,
        "vnnlib_path": vnnlib,
        "timeout": timeout,
        "log_path": log_path
    }
    if verifier.name == "marabou":
        conf["num_workers"] = marabou_cpu
        conf["ram"] = marabou_ram
    if verifier.name == "abcrown":
        if benchmark['name'] == "acasxu":
            conf["config_path"] = abcrown_acasxu_config
        elif benchmark['name'] == "mnist_fc":
            if "x2" in onnx.stem:
                conf["config_path"] = abcrown_mnist_x2_config
            else:
                conf["config_path"] = abcrown_mnist_x4x6_config
        elif benchmark['name'] == "tllverifybench":
            conf["config_path"] = abcrown_tll_config
    return conf

