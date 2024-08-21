import os
import sys
import signal
import sqlite3

from tqdm import tqdm
from time import sleep
from pathlib import Path
from multiprocessing import cpu_count

from helpers import *
from benchmarks import benchmarks
from relu_splitter.verify import init_verifier


original_handler = signal.getsignal(signal.SIGINT)

tool_root   = Path(os.environ["TOOL_ROOT"])
exp_root    = Path(tool_root/'experiment')
log_root    = exp_root/'logs/verification/init_veri'
db_root     = exp_root/'dbs'

log_root.mkdir(parents=True, exist_ok=True)
db_root.mkdir(parents=True, exist_ok=True)

# special config for verifiers
marabou_cpu = 32 if cpu_count() > 100 else 10
marabou_ram = "64G"

# abcrown_acasxu_config       = f"{exp_root}/config/acasxu.yaml"
# abcrown_mnist_x2_config     = f"{exp_root}/config/mnistfc_small.yaml"
# abcrown_mnist_x4x6_config   = f"{exp_root}/config/mnistfc.yaml"
# abcrown_tll_config          = f"{exp_root}/config/tllVerifyBench.yaml"

repeat = 3

if __name__=="__main__":
    benchmark = benchmarks[sys.argv[1]]
    verifier_name = sys.argv[2]
    
    benchmark_name       = benchmark['name']
    verifier = init_verifier(verifier_name)


    benchmark_path  = benchmark['path']
    db = get_init_veri_db(db_root/f"init_veri.db")
    
    try:
        for onnx_path, vnnlib_path in tqdm(get_instances(benchmark)):
            onnx_path = benchmark_path/onnx_path
            vnnlib_path = benchmark_path/vnnlib_path
            onnx_name = onnx_path.stem
            vnnlib_name = vnnlib_path.stem

            (log_root/benchmark_name).mkdir(parents=True, exist_ok=True)
            
            for i in range(repeat):
                if already_in_veri_db(db, benchmark_name, onnx_name, vnnlib_name, verifier_name, i):
                    tqdm.write(f"Already in db: {onnx_name}~{vnnlib_name}~{verifier_name}~{i}, skipping")
                    sleep(0.01)
                    continue

                conf = {
                    'onnx_path': onnx_path,
                    'vnnlib_path': vnnlib_path,
                    'log_path': log_root/benchmark_name/f"{onnx_name}~{vnnlib_name}~{verifier_name}~{i}.log",
                    'timeout': benchmark['timeout'],
                }
                
                if verifier_name == "marabou":
                    conf['num_workers'] = marabou_cpu
                    conf['ram']         = marabou_ram
                # elif verifier_name == "abcrown":
                #     if benchmark_name=="acasxu":
                #         conf['config_path'] = abcrown_acasxu_config
                #     elif benchmark_name=="mnist_fc":
                #         if "x2" in onnx_name:
                #             conf['config_path'] = abcrown_mnist_x2_config
                #         else:
                #             conf['config_path'] = abcrown_mnist_x4x6_config
                #     elif benchmark_name=="tllverifybench":
                #         # pass # this config gives error when using converted tll
                #         conf['config_path'] = abcrown_tll_config
                else:
                    pass

                res, time = verifier.execute(conf)

                if res not in ["sat", "unsat"]:     # skip any other results
                    tqdm.write(f"SKIPPING: {res}: {onnx_name}~{vnnlib_name}~{verifier_name}~{i}")
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    for j in range(i, repeat):
                        insert_into_veri_db(db, benchmark_name, onnx_name, vnnlib_name, verifier_name, j, res, time)
                    signal.signal(signal.SIGINT, original_handler)
                    break

                tqdm.write(f"{res}:{time}: {onnx_name}~{vnnlib_name}~{verifier_name}~{i}")
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                insert_into_veri_db(db, benchmark_name, onnx_name, vnnlib_name, verifier_name, i, res, time)
                signal.signal(signal.SIGINT, original_handler)

    except KeyboardInterrupt:
        db.close()
        print("KeyboardInterrupt: Database closed")
        sys.exit(0)