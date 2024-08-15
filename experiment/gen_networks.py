import os
import sys

from tqdm import tqdm
from pathlib import Path
from itertools import product
import multiprocessing as mp

from helpers import *
from benchmarks import benchmarks

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier

import signal



tool_root   = Path(os.environ["TOOL_ROOT"])
exp_root    = Path(tool_root/'experiment')
output_root = exp_root/'onnx'
log_root    = exp_root/'logs/relusplitter'
db_root     = exp_root/'dbs'
db_root.mkdir(parents=True, exist_ok=True)


p_split_strat           = ["reluS+", "reluS-", "random"]   # "single" 
p_masks                 = ["stable+", "stable-", "unstable"] # "all", "stable"
p_nsplits = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
seeds = list(range(5))
split_idx = 0
atol, rtol = 1e-5, 1e-5

invalid_combinations    = [("reluS+", "stable-"), ("reluS-", "stable+"), ("reluS+", "stable"), ("reluS-", "stable"), ("reluS+", "all"), ("reluS-", "all")]
valid_strat_n_mask      = [pair for pair in product(p_split_strat, p_masks) if pair not in invalid_combinations]





if __name__ == "__main__":
    benchmark = benchmarks[sys.argv[1]]
    num_workers = int(sys.argv[2])
    try:
        print(f"Processing {benchmark['name']}\n\n\n")
        benchmark_name = benchmark["name"]
        instances = get_instances(benchmark)
        timeout = benchmark["timeout"]
        output_dir = Path(output_root / benchmark['name'])
        log_dir = Path(log_root / benchmark['name'])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        db = get_gen_network_db(db_root/f'gen_network_{benchmark_name}.db')


        # for onnx_path, vnnlib_path in tqdm(instances):
        for onnx_path, vnnlib_path in tqdm(instances):
            onnx_path = benchmark["path"] / onnx_path
            vnnlib_path = benchmark["path"] / vnnlib_path
            max_nsplits = ReluSplitter.info_s(onnx_path, vnnlib_path)[split_idx]
            
            tasks = list(product(   [onnx_path],        # 0
                                    [vnnlib_path],      # 1
                                    [output_dir],       # 2
                                    [log_dir],          # 3
                                    [split_idx],        # 4
                                    valid_strat_n_mask, # 5
                                    p_nsplits,          # 6
                                    seeds,              # 7
                                    [atol],             # 8
                                    [rtol]))            # 9
            
            valid_tasks     = [task for task in tasks if task[6] <= max_nsplits[task[5][1]]]
            invalid_tasks   = [task for task in tasks if task not in valid_tasks]

            tasks_todo      = [task for task in valid_tasks if not already_in_gen_network_db(db, benchmark_name, task)]
            # {"tasks_todo": len(tasks_todo), "valid_tasks": len(valid_tasks), "total_tasks": len(tasks)}
            tqdm.write(f"Tasks to do: {len(tasks_todo)}\tValid tasks: {len(valid_tasks)}\tTotal tasks: {len(tasks)}")
            if len(tasks_todo) == 0:
                tqdm.write("All tasks already in db, skipping...")
                continue
            with mp.Pool(num_workers) as pool:
                pool.starmap(run_splitter, tasks_todo)

            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            
            tqdm.write(">>> writing to db...ignoring sigint")
            for task in valid_tasks:
                insert_into_gen_network_db(db, benchmark_name, task, "DONE")
            for task in invalid_tasks:
                insert_into_gen_network_db(db, benchmark_name, task, "SKIP: not_enough_neurons")
            tqdm.write(f"db length: {size_of_db(db)}")
            tqdm.write(">>> finished, restoring sigint")

            signal.signal(signal.SIGINT, original_handler)
        db.close()
            
                
    except KeyboardInterrupt:
        db.close()
        print("KeyboardInterrupt: Database closed")
        exit(0)