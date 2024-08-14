import os
from time import sleep
from pathlib import Path
from itertools import product
import multiprocessing as mp

from tqdm import tqdm
from tinydb import TinyDB, Query
from time import sleep

from helpers import *

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier

import signal
global sigint_flag
sigint_flag = False
def set_flag(sig, frame):
    sigint_flag = True



tool_root   = Path(os.environ["TOOL_ROOT"])
exp_root    = Path(tool_root/'experiment')
output_root = Path(exp_root/'onnx')
log_root    = Path(exp_root/'logs/relusplitter')
db          = TinyDB(exp_root/'db.json')

p_split_strat           = ["reluS+", "reluS-", "random"]   # "single" 
p_masks                 = ["stable+", "stable-", "unstable"] # "all", "stable"
p_nsplits = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
seeds = list(range(5))
split_idx = 0
atol, rtol = 1e-5, 1e-5

invalid_combinations    = [("reluS+", "stable-"), ("reluS-", "stable+"), ("reluS+", "stable"), ("reluS-", "stable"), ("reluS+", "all"), ("reluS-", "all")]
valid_strat_n_mask      = [pair for pair in product(p_split_strat, p_masks) if pair not in invalid_combinations]

num_workers = 90 if mp.cpu_count()>100 else 10

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





if __name__ == "__main__":
    try:
        for benchmark in benchmarks:
            print(f"Processing {benchmark['name']}\n\n\n")

            instances = get_instances(benchmark)
            timeout = benchmark["timeout"]
            output_dir = Path(output_root / benchmark['name'])
            log_dir = Path(log_root / benchmark['name'])
            output_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            # for onnx_path, vnnlib_path in tqdm(instances):
            for i, (onnx_path, vnnlib_path) in tqdm(enumerate(instances)):
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

                tasks_todo      = [task for task in valid_tasks if not already_in_db(db, task)]
                # {"tasks_todo": len(tasks_todo), "valid_tasks": len(valid_tasks), "total_tasks": len(tasks)}
                tqdm.write(f"Instance: {i+1}/{len(instances)}")
                tqdm.write(f"Tasks to do: {len(tasks_todo)} \n Valid tasks: {len(valid_tasks)} \n Total tasks: {len(tasks)}")

                with mp.Pool(num_workers) as pool:
                    pool.starmap(run_splitter, tasks_todo)

                original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, set_flag)
                
                tqdm.write(">>> writing to db...")
                for task in valid_tasks:
                    insert_into_db(db, task, "DONE")
                for task in invalid_tasks:
                    insert_into_db(db, task, "SKIP: not_enough_neurons")
                tqdm.write(">>> finished")

                signal.signal(signal.SIGINT, original_handler)
                if sigint_flag:
                    raise KeyboardInterrupt
                
    except KeyboardInterrupt:
        db.close()
        print("KeyboardInterrupt: Database closed")
        exit(0)