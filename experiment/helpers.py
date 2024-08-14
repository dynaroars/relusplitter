import os
import subprocess
from tinydb import Query



def get_instances(benchmark):
    instances = []
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx_path, vnnlib_path, _ = line.strip().split(',')
            instances.append((onnx_path, vnnlib_path))
    return instances

def already_in_db(db, task):
    Task = Query()
    return db.search(
        (Task.onnx == str(task[0].stem)) &
        (Task.vnnlib == str(task[1].stem)) &
        (Task.split_idx == task[4]) &
        (Task.split_strat == task[5][0]) &
        (Task.mask == task[5][1]) &
        (Task.nsplits == task[6]) &
        (Task.seed == task[7]) &
        (Task.atol == task[8]) &
        (Task.rtol == task[9])
    ) != []

def insert_into_db(db, task, val):
    if not already_in_db(db, task):
        db.insert(
                        {
                            "onnx"      : str(task[0].stem),
                            "vnnlib"    : str(task[1].stem),
                            "split_idx" : task[4],
                            "split_strat": task[5][0],
                            "mask"      : task[5][1],
                            "nsplits"   : task[6],
                            "seed"      : task[7],
                            "atol"      : task[8],
                            "rtol"      : task[9],
                            "status"    : val,
                        }
                    )
    
def run_splitter(onnx_path, vnnlib_path, output_dir, log_dir, split_idx, strat_n_mask, nsplits, seed, atol, rtol):
    wd = os.environ["TOOL_ROOT"]
    strat, mask = strat_n_mask
    fname = f"{onnx_path.stem}_{vnnlib_path.stem}_RS_{split_idx}_{mask}_{strat}_{nsplits}_{seed}"
    output_path = output_dir / f"{fname}.onnx"
    log_path  = log_dir / f"{fname}.log"

    cmd =   f"python main.py split --net {onnx_path} --spec {vnnlib_path} --output {output_path} "\
            f"--split_strategy {strat} --mask {mask} --split_idx {split_idx} "\
            f"--n_splits {nsplits} --seed {seed} --atol {atol} --rtol {rtol}"
    with open(log_path, "w") as f:
        subprocess.run(cmd, shell=True, cwd=wd, stdout=f, stderr=f)


