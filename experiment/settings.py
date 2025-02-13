from pathlib import Path
import os
from multiprocessing import cpu_count

num_cores = 12

tool_root   = Path(os.environ["TOOL_ROOT"])
abc_conf_root = tool_root / "experiment" / "configs"

def get_marabou_milp(benchmark_name, onnx_name):
    if benchmark_name == "acasxu":
        return False
    # if benchmark_name == "mnist_fc":
    #     return True
    
    return True


def get_abcrown_config(benchmark_name, onnx_name):
    if benchmark_name == "acasxu":
        return abc_conf_root / "acasxu.yaml"
    elif benchmark_name == "mnist_fc":
        # if "x2" in onnx_name:
        #     return abc_conf_root / "mnistfc_small.yaml"
        # else:
        #     return abc_conf_root / "mnistfc_large.yaml"
        return abc_conf_root / "mnistfc_both_split.yaml"
    elif benchmark_name == "reach_prob":
        if "gcas" in onnx_name:
            return abc_conf_root / "reach_probability_gcas.yaml"
        else:
            return abc_conf_root / "reach_probability.yaml"
    elif benchmark_name == "rl_benchmarks":
        return abc_conf_root / "rl_benchmarks.yaml"
    elif benchmark_name == "collins_rul":
        return abc_conf_root / "collins-rul-cnn.yaml"
    elif benchmark_name == "metaroom":
        return abc_conf_root / "metaroom.yaml"
    elif benchmark_name == "cifar2020":
        return abc_conf_root / "cifar2020_2_255.yaml"
    elif benchmark_name == "oval21":
        return abc_conf_root / "oval22.yaml"
    elif benchmark_name == "resnet_a":
        return abc_conf_root / "resnet_A.yaml"
    elif benchmark_name == "resnet_b":
        return abc_conf_root / "resnet_B.yaml"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


acasxu = {
    "name"      : "acasxu",   
    "path"      : Path(tool_root/'data'/'acasxu_converted'),
    "timeout"   : 120,
}
mnist_fc = {
    "name"      : "mnist_fc",
    "path"      : Path(tool_root/'data'/'mnist_fc'),
    "timeout"   : 300,
}
reach_prob = {
    "name"      : "reach_prob_density",
    "path"      : Path(tool_root/'data'/'reach_prob_density'),
    "timeout"   : 600,
}
rl_benchmarks = {
    'name'      : "rl_benchmarks",
    'path'      : Path(tool_root/'data'/'rl_benchmarks'),
    'timeout'   : 600,
}
metaroom = {
    'name'      : "metaroom",
    'path'      : Path(tool_root/'data'/'metaroom'),
    'timeout'   : 600,
}
cifar2020 = {
    'name'      : "cifar2020",
    'path'      : Path(tool_root/'data'/'cifar2020'),
    'timeout'   : 600,
}
oval21 = {
    'name'      : "oval21",
    'path'      : Path(tool_root/'data'/'oval21'),
    'timeout'   : 600,
}
collins_rul = {
    "name"      : "collins_rul_cnn",
    "path"      : Path(tool_root/'data'/'collins_rul_cnn'),
    "timeout"   : 300,
}
resnet_a = {
    "name"      : "sri_resnet_a",
    "path"      : Path(tool_root/'data'/'sri_resnet_a'),
    "timeout"   : 300,
}
resnet_b = {
    "name"      : "sri_resnet_b",
    "path"      : Path(tool_root/'data'/'sri_resnet_b'),
    "timeout"   : 300,
}



benchmarks = {
    'acasxu': acasxu,
    'mnist_fc': mnist_fc,
    'reach_prob': reach_prob,
    'rl_benchmarks': rl_benchmarks,
    'collins_rul': collins_rul,
    'metaroom': metaroom,
    'cifar2020': cifar2020,
    'oval21': oval21,
    'resnet_a': resnet_a,
    'resnet_b': resnet_b,
}

# FC
fc_benchmarks = [
    'acasxu',
    'mnist_fc',
    'reach_prob',
    'rl_benchmarks',
]
# CONV
conv_benchmarks = [
    'collins_rul'
]
# FC + CONV
mixed_benchmarks = [
    'metaroom',
    'cifar2020',
    'oval21',
]




def get_instances(benchmark):
    instances = []
    path = benchmark["path"]
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx, vnnlib, timeout = line.strip().split(',')
            try:
                instances.append((path / onnx, path / vnnlib, int(timeout)))
            except:
                instances.append((path / onnx, path / vnnlib, int(float(timeout))))
    return instances

def get_remaining(log_fname, benchmark_instances):
    with open(log_fname, "r") as f:
        lines = f.readlines()

    print(f"last line: {lines[-1]}")

    if len(lines) == 1:
        print(f"LINE 1 of log: {log_fname}")
        print(lines)
        return benchmark_instances
    if len(lines) == benchmark_instances:
        print("all DONE")
        return []
    
    last_completed = lines[-1].strip().split(",")
    last_onnx, last_vnnlib = [i.strip() for i in last_completed[:2]]

    instance_onnx, instance_vnnlib, _ = benchmark_instances[len(lines) - 2]
    instance_onnx, instance_vnnlib = Path(instance_onnx).stem, Path(instance_vnnlib).stem

    assert last_onnx == instance_onnx and last_vnnlib == instance_vnnlib, f"last_completed: {last_onnx} {last_vnnlib} != instance: {instance_onnx} {instance_vnnlib}"

    remaining = benchmark_instances[len(lines) - 1:]
    print(f"last_completed: {last_onnx} {last_vnnlib}")
    print(f"resuming from: {instance_onnx} {instance_vnnlib}, remaining: {len(remaining)}/{len(benchmark_instances)}")

    return remaining