import os
import subprocess
from tinydb import Query

import signal
import time


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
    
global sigint_flag
sigint_flag = False
def set_flag(sig, frame):
    sigint_flag = True

def exec_ignor_sigint(fn, params):
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, set_flag)
    try:
        fn(*params)
    finally:
        # Restore the original SIGINT handler
        signal.signal(signal.SIGINT, original_handler)
    if sigint_flag:
        raise KeyboardInterrupt
    