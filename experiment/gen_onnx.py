import os
import sys
import signal

import argparse

from tqdm import tqdm
from pathlib import Path
from itertools import product
import multiprocessing as mp

from helpers import *
from settings import *

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier

EXP_NAME = "exp1"

TOOL_ROOT = Path(os.environ.get("TOOL_ROOT"))
EXP_ROOT = TOOL_ROOT / "experiment"
STORAGE_ROOT = EXP_ROOT / "generated" / EXP_NAME

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, default="all", choices=["fc", "conv", "all"])

    args = parser.parse_args()

    benchmark_name = args.benchmark
    instances = get_instances(benchmarks[benchmark_name])

    conf = {
        'split_mask': "stable",
        'n_splits': None,
        'split_idx': 0,
        'scale_factor': [1.0,-1.0],
        'random_seed': args.seed,
        'create_baseline': True,
        'closeness_check': False,
        'bounding_method': "backward",
    }
    

    output_path = STORAGE_ROOT / benchmark_name / f"{benchmark_name}_idx-{conf['split_idx']}_method-{conf['bounding_method']}_mask-{conf['split_mask']}_mode-{args.mode}_seed-{args.seed}"
    new_model_dir = output_path / "onnx"
    baseline_dir = output_path / "baseline"
    output_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "fc":
        output_f = output_path / f"{benchmark_name}_info.csv"
        with open(output_f, "w") as f:
            f.write("onnx, vnnlib, stable, stable+, stable-, unstable, total\n")
            for instance in instances:
                onnx_path, vnnlib_path, timeout = instance
                onnx_name, vnnlib_name = onnx_path.stem, vnnlib_path.stem

                # collect info
                splitter = ReluSplitter(onnx_path, vnnlib_path, conf=conf)
                info = splitter.fc_info(conf['split_idx'], mode=args.mode, bounding_method=conf['bounding_method'])
                f.write(f"{onnx_name}, {vnnlib_name}, {info['stable']}, {info['stable+']}, {info['stable-']}, {info['unstable']}, {info['all']}\n")
                if info['stable'] == 0:
                    conf2 = conf.copy()
                    conf2['split_mask'] = "all"
                    conf2['n_splits'] = int(info['all']/4)
                    new_model, baseline = splitter.split(0, args.mode , conf2)
                else:
                    new_model, baseline = splitter.split(0, args.mode , conf)
                new_model.save(new_model_dir / f"{onnx_name}_Rsplitter_{vnnlib_name}.onnx")
                baseline.save(baseline_dir / f"{onnx_name}_baseline_{vnnlib_name}.onnx")
    if args.mode == "conv":
        if benchmark_name in ["collins_rul","metaroom"]:
            conf['bounding_method'] = "ibp"

        output_f = output_path / f"{benchmark_name}_info.csv"
        with open(output_f, "w") as f:
            f.write("onnx, vnnlib\n")
            for instance in instances:
                onnx_path, vnnlib_path, timeout = instance
                onnx_name, vnnlib_name = onnx_path.stem, vnnlib_path.stem

                splitter = ReluSplitter(onnx_path, vnnlib_path, conf=conf)
                f.write(f"{onnx_name}, {vnnlib_name}\n")
                new_model, baseline = splitter.split(0, args.mode , conf)
                new_model.save(new_model_dir / f"{onnx_name}_Rsplitter_{vnnlib_name}.onnx")
                baseline.save(baseline_dir / f"{onnx_name}_baseline_{vnnlib_name}.onnx")