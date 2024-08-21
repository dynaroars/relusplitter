import os
import sys
import signal

from tqdm import tqdm
from pathlib import Path
from itertools import product
import multiprocessing as mp

from helpers import *
from benchmarks import benchmarks

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier


original_handler = signal.getsignal(signal.SIGINT)



onnx_root   = exp_root/'onnx'
log_root    = exp_root/'logs/relusplitter'
verify_log_root = exp_root/'logs/verification'
db_root     = exp_root/'dbs'
db_root.mkdir(parents=True, exist_ok=True)

# split_strat = ["reluS+", "reluS-", "random"]
# split_masks = ["stable+", "stable-", "unstable"]
split_strat = ["reluS+", "reluS-", "random"]
split_masks = ["stable+", "stable-", "unstable"]
split_idx   = 0
# num_splits = {
#     acasxu: [],
#     mnist_fc: [],
#     tllverifybench: [],
# }
repeats = 1
atol, rtol = 1e-4, 1e-4

invalid_combinations    = [("reluS+", "stable-"), 
                           ("reluS-", "stable+"), 
                           ("reluS+", "stable"), 
                           ("reluS-", "stable"), 
                           ("reluS+", "all"), 
                           ("reluS-", "all")]

valid_strat_n_mask      = [pair for pair in product(split_strat, split_masks) if pair not in invalid_combinations]


if __name__ == "__main__":
    try:
        benchmark           = benchmarks[sys.argv[1]]
        verifier_name       = sys.argv[2]
        n_splits_ratio      = float(sys.argv[3])
        n_splits_increment  = int(sys.argv[4])
    except Exception as e:
        print(f"Usage: python {sys.argv[0]} <benchmark> <verifier> <n_splits_ratio> <n_splits_increment>")
        sys.exit(1)

    benchmark_name = benchmark["name"]
    benchmark_path = benchmark["path"]
    verifier = init_verifier(verifier_name)
    instances = get_selected_instances(exp_root,benchmark)
    timeout = benchmark["timeout"]+30
    onnx_dir = onnx_root / benchmark_name   # for the splitted onnx
    log_dir = log_root / benchmark_name     # for relusplitter logs
    # db = get_exp_db(db_root/f'exp_{benchmark_name}.db')

    veri_log_dir = exp_root/f'logs/verification/{benchmark_name}'

    veri_log_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


    # filter instances
    instances_x_splitable = []
    for onnx_fname, vnnlib_fname in tqdm(instances):
        onnx_path = benchmark_path / f"onnx/{onnx_fname}.onnx"
        vnnlib_path = benchmark_path / f"vnnlib/{vnnlib_fname}.vnnlib"
        summary = ReluSplitter.info_s(onnx_path, vnnlib_path)     # types and counts of neurons at each layer []
        instances_x_splitable.append((onnx_path, vnnlib_path, summary[0]))
    
    for split_strat, split_mask in tqdm(valid_strat_n_mask):
        cutoff = int(len(instances_x_splitable) * n_splits_ratio)   # idx of the cutoff

        instances_x_splitable.sort(key = lambda x: x[2][split_mask])
        instances_2_use = instances_x_splitable[cutoff:]
        max_n_splits = instances_2_use[0][2][split_mask]
        for onnx_path, vnnlib_path, _ in instances_2_use:
            veri_log_path = veri_log_dir / f"{verifier_name}~{onnx_path.stem}~{vnnlib_path.stem}~orginal.log"
            conf = {
                        "onnx_path": onnx_path,
                        "vnnlib_path": vnnlib_path,
                        "timeout": timeout,
                        "log_path": veri_log_path,
                    }
            if verifier_name == "marabou":
                conf['num_workers'] = marabou_cpu
                conf['ram']         = marabou_ram
            # elif verifier_name == "abcrown":
            #     if benchmark_name=="acasxu":
            #         conf['config_path'] = abcrown_acasxu_config
            #     elif benchmark_name=="mnist_fc":
            #         if "x2" in onnx_fname:
            #             conf['config_path'] = abcrown_mnist_x2_config
            #         else:
            #             conf['config_path'] = abcrown_mnist_x4x6_config
            #     elif benchmark_name=="tllverifybench":
            #         # pass # this config gives error when using converted tll
            #         conf['config_path'] = abcrown_tll_config
            else:
                pass
            # print(veri_log_path)
            # res, time = verifier.execute(conf)
            # print(f"Verification of {onnx_path.stem}~{vnnlib_path.stem}~orginal done in {time} seconds")
            # print(res, time)

            print("===============================")
            for n_splits in range(n_splits_increment, max_n_splits, n_splits_increment):
                seed = 0
                while seed < repeats:
                    # split instances
                    failed, fname = run_splitter(   onnx_path,
                                                    vnnlib_path,
                                                    onnx_dir,
                                                    log_dir,
                                                    split_idx,
                                                    (split_strat, split_mask),
                                                    n_splits,
                                                    seed,
                                                    atol,
                                                    rtol
                                                    )
                    if failed:
                        continue
                    # verify instances
                    veri_log_path = veri_log_dir / f"{verifier_name}~{onnx_path.stem}~{vnnlib_path.stem}~{split_idx}~{split_strat}~{split_mask}~{n_splits}~{seed}.log"
                    conf['onnx_path'] = onnx_dir/f"{fname}.onnx"
                    conf['log_path'] = veri_log_path
                    # conf = {
                    #     "onnx_path": onnx_dir/f"{fname}.onnx",
                    #     "vnnlib_path": vnnlib_path,
                    #     "timeout": timeout,
                    #     "log_path": veri_log_path,
                    # }
                    # if verifier_name == "marabou":
                    #     conf['num_workers'] = marabou_cpu
                    #     conf['ram']         = marabou_ram
                    # elif verifier_name == "abcrown":
                    #     if benchmark_name=="acasxu":
                    #         conf['config_path'] = abcrown_acasxu_config
                    #     elif benchmark_name=="mnist_fc":
                    #         if "x2" in onnx_fname:
                    #             conf['config_path'] = abcrown_mnist_x2_config
                    #         else:
                    #             conf['config_path'] = abcrown_mnist_x4x6_config
                    #     elif benchmark_name=="tllverifybench":
                    #         # pass # this config gives error when using converted tll
                    #         conf['config_path'] = abcrown_tll_config
                    # else:
                    #     pass
                    print(veri_log_path)
                    res, time = verifier.execute(conf)
                    print(f"Verification of {onnx_path.stem}~{vnnlib_path.stem}~{split_idx}~{split_strat}~{split_mask}~{n_splits}~{seed} done in {time} seconds")
                    print(res, time)
                    seed += 1
                    



