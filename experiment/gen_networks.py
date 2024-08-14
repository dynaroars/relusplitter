import os
from time import sleep
from pathlib import Path
from itertools import product
import multiprocessing as mp

import tqdm

from time import sleep

from helpers import *

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier



tool_root = Path(os.environ["TOOL_ROOT"])
exp_root = Path(tool_root/'experiment')
output_root = Path(exp_root/'onnx')
log_root = Path(exp_root/'logs/relusplitter')

p_split_strat           = ["single", "reluS+", "reluS-", "random"]
p_masks                 = ["stable+", "stable-", "stable", "unstable", "all"]
p_nsplits = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
seeds = list(range(5))
split_idx = 0
atol, rtol = 1e-5, 1e-5

invalid_combinations    = [("reluS+", "stable-"), ("reluS-", "stable+"), ("reluS+", "stable"), ("reluS-", "stable"), ("reluS+", "all"), ("reluS-", "all")]
valid_strat_n_mask      = [pair for pair in product(p_split_strat, p_masks) if pair not in invalid_combinations]

num_workers = 10

acasxu = {
    "name"      : "acasxu",   
    "path"      : Path(tool_root/'data'/'acasxu_converted'),
    "timeout"   : 300,
}
mnist_fc = {
    "name"      : "mnist_fc",
    "path"      : Path(tool_root/'data'/'mnist_fc'),
    "timeout"   : 600,
}
benchmarks = [acasxu, mnist_fc]




def run_splitter(onnx_path, vnnlib_path, output_dir, log_dir, split_idx, strat_n_mask, nsplits, seed, atol, rtol):
    wd = os.environ["TOOL_ROOT"]
    strat, mask = strat_n_mask
    fname = f"{onnx_path.stem}_{vnnlib_path.stem}_{split_idx}_{mask}_{strat}_{nsplits}_{seed}"
    output_path = output_dir / f"{fname}.onnx"
    log_path  = log_dir / f"{fname}.log"

    cmd =   f"python main.py split --net {onnx_path} --spec {vnnlib_path} --output {output_path} "\
            f"--split_strategy {strat} --mask {mask} --split_idx {split_idx} "\
            f"--n_splits {nsplits} --seed {seed} --atol {atol} --rtol {rtol}"
    print(cmd)
    with open(log_path, "w") as f:
        subprocess.run(cmd, shell=True, cwd=wd, stdout=f, stderr=f)



if __name__ == "__main__":
    for benchmark in benchmarks:
        print(f"Processing {benchmark['name']}\n\n\n")

        instances = get_instances(benchmark)
        timeout = benchmark["timeout"]
        output_dir = Path(output_root / benchmark['name'])
        log_dir = Path(log_root / benchmark['name'])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # for onnx_path, vnnlib_path in tqdm(instances):
        for onnx_path, vnnlib_path in instances:
            onnx_path = benchmark["path"] / onnx_path
            vnnlib_path = benchmark["path"] / vnnlib_path
            
            
            tasks = product([onnx_path],
                            [vnnlib_path], 
                            [output_dir],
                            [log_dir],
                            [split_idx],
                            valid_strat_n_mask,
                            p_nsplits,
                            seeds,
                            [atol],
                            [rtol])
            
            with mp.Pool(num_workers) as pool:
                pool.starmap(run_splitter, tasks)