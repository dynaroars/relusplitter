import os
import sys
import signal

import argparse

from statistics import median
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
RESULTS_ROOT = EXP_ROOT / "results" / EXP_NAME
LOGS_ROOT = EXP_ROOT / "logs" / EXP_NAME

EXP_ROOT.mkdir(parents=True, exist_ok=True)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

def release_resources():
    os.system("pkill -9 -f get_cuts")   # abc oval21

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--mode", type=str, default="all", choices=["fc", "conv", "all"])
    parser.add_argument("--verifier", type=str, default="ibp", choices=["neuralsat", "abcrown", "marabou", "nnenum"])

    args = parser.parse_args()

    benchmark = benchmarks[args.benchmark]
    benchmark_name = args.benchmark+"_16"
    instances = get_instances(benchmark)

    csv_file = RESULTS_ROOT / benchmark_name / f"{benchmark_name}_{args.verifier}.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    generated_benchmarks =[p for p in (STORAGE_ROOT/benchmark_name).iterdir() if p.is_dir()]
    assert len(generated_benchmarks) == args.seed
    generated_benchmarks_sorted = sorted(
        generated_benchmarks, 
        key=lambda p: int(p.name[-1])
    )
    

    verifier = init_verifier(args.verifier)
    

    conf = {
        'onnx_path': None,
        'vnnlib_path': None,
        'log_path': None,
        'verbosity': 1,
        'num_workers': 128,
        'config_path': None,
        'timeout': None,
        'milp': False
    }

    if not csv_file.exists():   # create the csv file with header
        print(f"Creating {csv_file}")
        f = open(csv_file, "w")
        f.write("onnx, vnnlib, ")
        f.write(",".join([f"{s}_{i} " for s in ["O","O_res","S","S_res","B","B_res"] for i in range(args.seed) ]) + "\n")
        f.close()

    with open(csv_file, "a+") as f:
        for onnx_path, vnnlib_path, timeout in tqdm(get_remaining(csv_file, instances)):
            onnx_name, vnnlib_name = onnx_path.stem, vnnlib_path.stem
            
            conf['config_path'] = get_abcrown_config(args.benchmark, onnx_name)
            conf['milp'] = get_marabou_milp(args.benchmark, onnx_name)
            conf['timeout'] = timeout
            conf['onnx_path'] = onnx_path
            conf['vnnlib_path'] = vnnlib_path


            new_onnx_name = f"{onnx_name}_Rsplitter_{vnnlib_name}.onnx"
            baseline_onnx_name = f"{onnx_name}_baseline_{vnnlib_name}.onnx"
            new_onnxs      = [ g/ "onnx" / new_onnx_name for g in generated_benchmarks]
            baseline_onnxs = [ g/ "baseline" / baseline_onnx_name for g in generated_benchmarks]
            assert all([o.exists() for o in new_onnxs]) and all([o.exists() for o in baseline_onnxs])


            veri_results_O = [] # original 
            veri_results_S = [] # splitted
            veri_results_B = [] # baseline

            veri_times_O = [] # original
            veri_times_S = []
            veri_times_B = []

            # verify the original model
            SKIP = False
            tqdm.write(f"Verifying Original model: {onnx_name}, {vnnlib_name}, {timeout}")
            for rep in range(args.seed):
                conf['log_path'] = LOGS_ROOT / benchmark_name / args.verifier / f"ORIGINAL_{onnx_name}_{vnnlib_name}_REPEAT-{rep}.log"
                res, time = verifier.execute(conf)
                release_resources()
                veri_results_O.append(res)
                veri_times_O.append(time)
                tqdm.write(f"Original: {res}, {time}")
                if res not in ["unsat"]:
                    f.write( f"{onnx_name}, {vnnlib_name}, ERROR_SKIPPED, LAST RES {res} {time}\n")
                    f.flush()
                    SKIP = True
                    break
            if SKIP:
                tqdm.write(f"Skipping {onnx_name}, {vnnlib_name}")
                continue

            # verify the splitted model
            conf['timeout'] = int(timeout * 1.5)
            tqdm.write(f"Verifying Split model: {onnx_name}, {vnnlib_name}")
            for new_onnx in new_onnxs:
                conf['onnx_path'] = new_onnx
                conf['log_path'] = LOGS_ROOT / benchmark_name / args.verifier / f"{new_onnx.stem}_seed-{new_onnx.parent.parent.name[-1]}.log"
                res, time = verifier.execute(conf)
                release_resources()
                veri_results_S.append(res)
                veri_times_S.append(time)
                tqdm.write(f"Split: {res}, {time}")


            # verify the baseline model
            tqdm.write(f"Verifying Baseline model: {onnx_name}, {vnnlib_name}")
            for baseline_onnx in baseline_onnxs:
                conf['onnx_path'] = baseline_onnx
                conf['log_path'] = LOGS_ROOT / benchmark_name / args.verifier / f"{baseline_onnx.stem}_seed-{baseline_onnx.parent.parent.name[-1]}.log"
                res, time = verifier.execute(conf)
                release_resources()
                veri_results_B.append(res)
                veri_times_B.append(time)
                tqdm.write(f"Baseline: {res}, {time}")

            f.write(
                f"{onnx_name}, {vnnlib_name}, " + 
                ",".join([str(i) for i in veri_times_O]) + "," +
                ",".join([str(i) for i in veri_results_O]) + "," +
                ",".join([str(i) for i in veri_times_S]) + "," +
                ",".join([str(i) for i in veri_results_S]) + "," +
                ",".join([str(i) for i in veri_times_B]) + "," +
                ",".join([str(i) for i in veri_results_B]) + "\n"
            )
            f.flush()
            veri_times_O = [i if isinstance(i, (float,int)) else -1 for i in veri_times_O]
            veri_times_S = [i if isinstance(i, (float,int)) else -1 for i in veri_times_S]
            veri_times_B = [i if isinstance(i, (float,int)) else -1 for i in veri_times_B]
            tqdm.write(f"Res written: O_median: {median(veri_times_O)}, S_median: {median(veri_times_S)}, B_median: {median(veri_times_B)}")
            tqdm.write(f"Res written: O_avg: {sum(veri_times_O)/len(veri_times_O)}, S_avg: {sum(veri_times_S)/len(veri_times_S)}, B_avg: {sum(veri_times_B)/len(veri_times_B)}")
