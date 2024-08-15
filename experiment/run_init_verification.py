import os
import sys
import signal
from pathlib import Path

from multiprocessing import cpu_count

import sqlite3


from tqdm import tqdm

from helpers import *
from benchmarks import benchmarks
from relu_splitter.verify import init_verifier



tool_root   = Path(os.environ["TOOL_ROOT"])
exp_root    = Path(tool_root/'experiment')
log_root    = exp_root/'logs/verification/init_veri'
db_root     = exp_root/'dbs'

log_root.mkdir(parents=True, exist_ok=True)
db_root.mkdir(parents=True, exist_ok=True)



repeat = 5


if __name__=="__main__":
    benchmark_name = sys.argv[1]
    verifier_name = sys.argv[2]
    
    benchmark       = benchmarks[benchmark_name]
    verifier = init_verifier(verifier_name)


    benchmark_path  = benchmark['path']
    db = get_init_veri_db(db_root/f"init_veri.db")
    
    try:
        for onnx_path, vnnlib_path in tqdm(get_instances(benchmark)):
            onnx_path = benchmark_path/onnx_path
            vnnlib_path = benchmark_path/vnnlib_path
            onnx_name = onnx_path.stem
            vnnlib_name = vnnlib_path.stem
            
            for i in range(repeat):
                if already_in_veri_db(db, benchmark_name, onnx_name, vnnlib_name, verifier_name, i):
                    tqdm.write(f"Already in db: {onnx_name}~{vnnlib_name}~{verifier_name}, skipping")
                    continue

                conf = {
                    'onnx_path': onnx_path,
                    'vnnlib_path': vnnlib_path,
                    'log_path': log_root/f"{onnx_name}~{vnnlib_name}~{verifier_name}~{i}.log",
                    'timeout': benchmark['timeout'],
                }
                if verifier_name == "marabou":
                    conf['num_workers'] = 32 if cpu_count() > 100 else 10
                    conf['ram']         = "64G"
                res, time = verifier.execute(conf)

                original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                insert_into_veri_db(db, benchmark_name, onnx_name, vnnlib_name, verifier_name, i, res, time)
                signal.signal(signal.SIGINT, original_handler)

    except KeyboardInterrupt:
        db.close()
        print("KeyboardInterrupt: Database closed")
        sys.exit(0)