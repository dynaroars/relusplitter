import os
import sys
import signal

from tqdm import tqdm
from pathlib import Path
from itertools import product
import multiprocessing as mp

from helpers import *
from settings import *

from relu_splitter.utils import logger
from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier


original_handler = signal.getsignal(signal.SIGINT)



if __name__=="__main__":
    try:
        option = sys.argv[1]
        benchmark = benchmarks[sys.argv[2]]
        verifier = init_verifier(sys.argv[3])
    except:
        print("Usage: python3 run.py <benchmark> <verifier>")
        sys.exit(1)

    if option == "info":
        instances = get_instances(benchmark)
        instance_x_counts = []
        with mp.Pool(num_cores) as pool:
            temp = pool.starmap(ReluSplitter.info_s, instances)
            temp = [x[0] for x in temp]
        instance_x_counts = [(onnx_path, vnnlib_path, summary) for (onnx_path, vnnlib_path), summary in zip(instances, temp)]
        # for onnx_path, vnnlib_path in tqdm(instances):
        #     summary = ReluSplitter.info_s( onnx_path, vnnlib_path )
        #     instance_x_counts.append( (onnx_path, vnnlib_path, summary[0]) )
        
        idx_25p = int(0.25*len(instance_x_counts))
        idx_50p = int(0.50*len(instance_x_counts))
        idx_75p = int(0.75*len(instance_x_counts))


        for mask in ["stable+", "stable-", "unstable"]:
            total = sum([x[2][mask] for x in instance_x_counts])
            instance_x_counts.sort(key=lambda x: x[2][mask])
            print(f"Mask: {mask}")
            print(f"25th percentile: {instance_x_counts[idx_25p][2][mask]}")
            print(f"50th percentile: {instance_x_counts[idx_50p][2][mask]}")
            print(f"75th percentile: {instance_x_counts[idx_75p][2][mask]}")
            print(f"Average: {total/len(instance_x_counts)}")
        
    elif option == "exp1":
        # Experiment 1: Split all stable neurons
        split_idx = 0
        strategy, mask = "random", "stable"
        # strategy, mask = "random", "unstable_n_stable+"
        min_splits, max_splits = 1, 99999999
        seed = 1
        stol = 1e-4
        atol = 1e-4

        instances = get_instances(benchmark)
        timeout = benchmark["timeout"]
        extra_time = 30
        log_dir = exp_root / "logs" / "exp1" / benchmark["name"] 
        log_dir_veri  = log_dir / "veri" / verifier.name
        log_dir_split = log_dir / "split"
        output_dir = exp_root/ "onnx" / "exp1" / benchmark["name"]

        log_dir_veri.mkdir(parents=True, exist_ok=True)
        log_dir_split.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_csv = exp_root / "results" / "exp1" / f"{benchmark['name']}-{verifier.name}-{mask}~{strategy}~{seed}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv = open(output_csv, "w")
        csv.write("onnx,vnnlib,strategy,mask,n_splits,original_r,original_t,splitted_r,splitted_t,baseline_r,baseline_t\n")

        for onnx_path, vnnlib_path in tqdm(get_instances(benchmark)):
            # init veri
            log_path = log_dir_veri / f"{onnx_path.stem}~{vnnlib_path.stem}~original.log"
            conf = get_verification_config(verifier, benchmark, onnx_path, vnnlib_path, log_path, timeout)
            r,t = verifier.execute(conf)
            if r not in ["sat", "unsat"]:
                tqdm.write(f"Original instance is {r} for {onnx_path.stem}~{vnnlib_path.stem} cannot be verified, skipping")
                # csv.write(f"{onnx_path.stem},{vnnlib_path.stem},SKIPPED,{r},{t},-1,-1,-1,-1\n")
                csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},SKIPPED,{r},{t},-1,-1,-1,-1\n")
                continue
            tqdm.write("====================================")
            tqdm.write(f"Original instance: {onnx_path.stem}~{vnnlib_path.stem}~{verifier.name}")
            tqdm.write(f"Original : {r}, {t}")

            n_splits = ReluSplitter.info_s(onnx_path, vnnlib_path)[0][mask]
            ret, splitted_onnx_path,splitter_log_path = run_splitter(onnx_path, vnnlib_path, output_dir, log_dir_split, split_idx, (strategy, mask), min_splits, max_splits, seed, atol, rtol)
            if ret != 0:
                tqdm.write(f"Splitting failed for {onnx_path.stem}~{vnnlib_path.stem}, log:{splitter_log_path}")
                csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},SPLIT_FAILED,{r},{t},-1,-1,-1,-1, \n")
                continue
                    
            log_path = log_dir_veri / f"{splitted_onnx_path.stem}.log"
            conf = get_verification_config(verifier, benchmark, splitted_onnx_path, vnnlib_path, log_path, timeout + extra_time)
            sr,st = verifier.execute(conf)
            tqdm.write(f"Splitted {n_splits}: {sr}, {st}")

            _, baseline_onnx_path = get_splitter_baseline(onnx_path, output_dir, n_splits, split_idx)
            log_path = log_dir_veri / f"{onnx_path.stem}~{vnnlib_path.stem}~baseline.log"
            conf = get_verification_config(verifier, benchmark, baseline_onnx_path, vnnlib_path, log_path, timeout)
            br,bt = verifier.execute(conf)
            tqdm.write(f"Baseline {n_splits}: {br}, {bt}")

            tqdm.write(f"{splitter_log_path}")
            tqdm.write(f"{splitted_onnx_path}")
            tqdm.write(f"{baseline_onnx_path}")
            csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},{n_splits},{r},{t},{sr},{st},{br},{bt}\n")
            csv.flush()

        csv.close()
        print(f"Results saved at {output_csv}")


    elif option == "exp2":
        # we splitted all stable,
        # now try stable+ and stable- neurons
        # and unstable on different strategies?
        split_idx = 0
        strategy = ["random", "reluS+", "reluS-"]
        masks = ["stable+", "stable-", "unstable"]
        min_splits, max_splits = 1, 99999999

        # ====
        for strategy, mask, in product(strategy, masks):
            invalid_combinations = [("stable+", "reluS-"), ("stable-", "reluS+"), ("unstable", "reluS+"), ("unstable", "reluS-")]
            if (strategy, mask) in invalid_combinations:
                continue
            split_idx = 0
            min_splits, max_splits = 1, 99999999
            seed = 1
            stol = 1e-4
            atol = 1e-4

            instances = get_instances(benchmark)
            timeout = benchmark["timeout"]
            extra_time = 30
            log_dir = exp_root / "logs" / option / benchmark["name"] 
            log_dir_veri  = log_dir / "veri" / verifier.name
            log_dir_split = log_dir / "split"
            output_dir = exp_root/ "onnx" / option / benchmark["name"]

            log_dir_veri.mkdir(parents=True, exist_ok=True)
            log_dir_split.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_csv = exp_root / "results" / option / f"{benchmark['name']}-{verifier.name}-{mask}~{strategy}~{seed}.csv"
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            csv = open(output_csv, "w")
            csv.write("onnx,vnnlib,strategy,mask,n_splits,original_r,original_t,splitted_r,splitted_t,baseline_r,baseline_t\n")

            for onnx_path, vnnlib_path in tqdm(get_instances(benchmark)):
                # init veri
                log_path = log_dir_veri / f"{onnx_path.stem}~{vnnlib_path.stem}~original.log"
                conf = get_verification_config(verifier, benchmark, onnx_path, vnnlib_path, log_path, timeout)
                r,t = verifier.execute(conf)
                if r not in ["sat", "unsat"]:
                    tqdm.write(f"Original instance is {r} for {onnx_path.stem}~{vnnlib_path.stem} cannot be verified, skipping")
                    # csv.write(f"{onnx_path.stem},{vnnlib_path.stem},SKIPPED,{r},{t},-1,-1,-1,-1\n")
                    csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},SKIPPED,{r},{t},-1,-1,-1,-1\n")
                    continue
                tqdm.write("====================================")
                tqdm.write(f"Original instance: {onnx_path.stem}~{vnnlib_path.stem}~{verifier.name}")
                tqdm.write(f"Original : {r}, {t}")

                n_splits = ReluSplitter.info_s(onnx_path, vnnlib_path)[0][mask]
                ret, splitted_onnx_path, splitter_log_path = run_splitter(onnx_path, vnnlib_path, output_dir, log_dir_split, split_idx, (strategy, mask), min_splits, max_splits, seed, atol, rtol)
                if ret != 0:
                    tqdm.write(f"Splitting failed for {onnx_path.stem}~{vnnlib_path.stem}, log:{splitter_log_path}")
                    csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},SPLIT_FAILED,{r},{t},-1,-1,-1,-1, \n")
                    continue

                log_path = log_dir_veri / f"{splitted_onnx_path.stem}.log"
                conf = get_verification_config(verifier, benchmark, splitted_onnx_path, vnnlib_path, log_path, timeout + extra_time)
                sr,st = verifier.execute(conf)
                tqdm.write(f"Splitted {n_splits}: {sr}, {st}")

                _, baseline_onnx_path = get_splitter_baseline(onnx_path, output_dir, n_splits, split_idx)
                log_path = log_dir_veri / f"{onnx_path.stem}~{vnnlib_path.stem}~baseline.log"
                conf = get_verification_config(verifier, benchmark, baseline_onnx_path, vnnlib_path, log_path, timeout)
                br,bt = verifier.execute(conf)
                tqdm.write(f"Baseline {n_splits}: {br}, {bt}")

                csv.write(f"{onnx_path.stem},{vnnlib_path.stem},{strategy},{mask},{n_splits},{r},{t},{sr},{st},{br},{bt}\n")
                csv.flush()

            csv.close()
            print(f"Results saved at {output_csv}")

    elif option == "exp3":
        # iterative split for instances that are not so good with exp1
        pass