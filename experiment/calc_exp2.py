import sys
import os
from pathlib import Path
from statistics import mean, median, stdev 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from itertools import product

instance_offset=0.1
res_offset = {
                "unsat":0,
                "sat"  :0.4
            }
tool_root = Path(os.environ["TOOL_ROOT"])
res_root = tool_root / "experiment" / "results"
fig_path = res_root / "figs"

fig_path.mkdir(parents=True, exist_ok=True)

class stats_DP():
    dps = []
    def __init__(self, line):
        # onnx, vnnlib, stable, stable+, stable-, unstable, total
        cols = line.split(",")
        self.onnx, self.vnnlib = cols[0], cols[1]
        self.size_incr = float(cols[2])
        self.dps.append(self)
    @classmethod
    def get_stats(cls, onnx, vnnlib):
        return [dp for dp in cls.dps if dp.onnx == onnx and dp.vnnlib == vnnlib][0]
    @classmethod
    def get_size_incr(cls, onnx, vnnlib):
        return cls.get_stats(onnx, vnnlib).size_incr



class DP():
    def __init__(self, line):
        cols = line.split(",")
        self.line = line
        self.valid = True
        self.onnx, self.vnnlib = cols[0], cols[1]

        if "ERROR_SKIPPED" in line:
            self.valid = False
            return
        # onnx, vnnlib, O_0 ,O_1 ,O_2 ,O_3 ,O_4 ,O_res_0 ,O_res_1 ,O_res_2 ,O_res_3 ,O_res_4 ,S_0 ,S_1 ,S_2 ,S_3 ,S_4 ,S_res_0 ,S_res_1 ,S_res_2 ,S_res_3 ,S_res_4 ,B_0 ,B_1 ,B_2 ,B_3 ,B_4 ,B_res_0 ,B_res_1 ,B_res_2 ,B_res_3 ,B_res_4 
        o_runs = [(float(cols[2]), cols[7]), (float(cols[3]), cols[8]), (float(cols[4]), cols[9]), (float(cols[5]), cols[10]), (float(cols[6]), cols[11])]
        s_runs = [(float(cols[12]), cols[17]), (float(cols[13]), cols[18]), (float(cols[14]), cols[19]), (float(cols[15]), cols[20]), (float(cols[16]), cols[21])]
        b_runs = [(float(cols[22]), cols[27]), (float(cols[23]), cols[28]), (float(cols[24]), cols[29]), (float(cols[25]), cols[30]), (float(cols[26]), cols[31])]
        
        o_runs.sort(key=lambda x: x[0])
        s_runs.sort(key=lambda x: x[0])
        b_runs.sort(key=lambda x: x[0])

        ors, srs, brs = [r[1] for r in o_runs], [r[1] for r in s_runs], [r[1] for r in b_runs]
        ots, sts, bts = [r[0] for r in o_runs], [r[0] for r in s_runs], [r[0] for r in b_runs]

        if "error" in ors+srs+brs or -1 in ots+sts+bts:
            self.valid = False
            return
        # add res checking
        if "sat" in ors+srs+brs and "unsat" not in ors+srs+brs:
            self.ground_truth = "sat"
        elif "unsat" in ors+srs+brs and "sat" not in ors+srs+brs:
            self.ground_truth = "unsat"
        else:
            self.valid = False

        self.ot = o_runs[2][0]
        self.st = s_runs[2][0]
        self.bt = b_runs[2][0]
        self.o_res = o_runs[2][1]
        self.s_res = s_runs[2][1]
        self.b_res = b_runs[2][1]

def load_dps(fp):
    with open(fp, "r") as f:
        lines = f.readlines()
    res = []
    for line in lines[1:]:
        temp = DP(line)
        res.append(temp)
        if not temp.valid:
            # print(f"INVALID: {temp.line.strip()}")
            pass

    print("{} loaded, {} DPs, {} valid".format(fp.stem, len(res), len([r for r in res if r.valid])))
    return res

def triangle_graph(dps, fname, timeout=300, tick_offset=0.2, fontsize=8, line_width=1, yscale='log'):
    onnxs = list(set([dp.onnx for dp in dps]))
    onnxs.sort()
    onnx_map = {onnx:i+tick_offset for i,onnx in enumerate(onnxs)}

    timeout_o, timeout_s, y_max = timeout, timeout*1.5, timeout*2

    xmin, xmax = 0, len(onnx_map)
    ymin, ymax = 0, y_max

    for dp in dps:
        x = onnx_map[dp.onnx] + res_offset[dp.ground_truth]
        xo,xs,xb = x+0*instance_offset, x+1*instance_offset, x+2*instance_offset
        marker = '.' if dp.ground_truth=='unsat' else '^'
        plt.scatter(xo, dp.ot, color="green", label="Original Instances(UNSAT)", marker=marker)
        plt.scatter(xs, dp.st, color="red", label="Splitted Instances(UNSAT)", marker=marker)
        plt.scatter(xb, dp.bt, color="cyan", label="Baseline Instances(UNSAT)", marker=marker)

        plt.plot((xo,xb),(dp.ot, dp.bt), linestyle='--', color='gray', linewidth=line_width)
        plt.plot((xo,xs),(dp.ot, dp.st), linestyle='--', color='gray', linewidth=line_width)
        plt.plot((xb,xs),(dp.bt, dp.st), linestyle='--', color='gray', linewidth=line_width)
    
    plt.yscale(yscale)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    plt.plot([xmin, xmax], [timeout_o, timeout_o], linestyle="--", color='blue')
    plt.text(xmax, timeout_o, f"{timeout_o:.1f}", color='blue', va='center')

    plt.plot([xmin, xmax], [timeout_s, timeout_s], linestyle="--", color='red')
    plt.text(xmax, timeout_s, f"{timeout_s:.1f}", color='blue', va='center')

    for x in range(1, len(onnx_map)):
        plt.plot([x, x], [ymin, ymax], linestyle="--", color='black')

    onnx_names = [i for i in onnx_map]
    onnx_names_x = [onnx_map[i]-tick_offset+0.5 for i in onnx_map]
    plt.yticks(fontsize=fontsize)
    plt.xticks(onnx_names_x, onnx_names, rotation=0, ha='center', fontsize=fontsize)

    unsat_o = mlines.Line2D([], [], color='green', marker='.', linestyle='None', label='Original Instances (UNSAT)')
    unsat_s = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='Splitted Instances (UNSAT)')
    unsat_b = mlines.Line2D([], [], color='cyan', marker='.', linestyle='None', label='Baseline Instances (UNSAT)')
    sat_o = mlines.Line2D([], [], color='green', marker='^', linestyle='None', label='Original Instances (SAT)')
    sat_s = mlines.Line2D([], [], color='red', marker='^', linestyle='None', label='Splitted Instances (SAT)')
    sat_b = mlines.Line2D([], [], color='cyan', marker='^', linestyle='None', label='Baseline Instances (SAT)')
    timeout_o_line = mlines.Line2D([], [], color='blue', linestyle='--', label='Timeout (Original)')
    timeout_s_line = mlines.Line2D([], [], color='red', linestyle='--', label='Timeout (Splitted)')

    plt.legend(handles=[unsat_o, unsat_s, unsat_b, sat_o, sat_s, sat_b, timeout_o_line, timeout_s_line], loc='upper right', fontsize=fontsize)

    plt.savefig(fname)


def triangle_graph_mnist(dps, fname, timeout=300, tick_offset=0.2, fontsize=8, line_width=1, yscale='log'):
    onnxs = list(set([dp.onnx for dp in dps]))
    onnxs.sort()
    onnx_map = {onnx:i+tick_offset for i,onnx in enumerate(onnxs)}

    timeout_o, timeout_s, y_max = timeout, timeout*1.5, timeout*2

    xmin, xmax = 0, len(onnx_map)
    ymin, ymax = 0, y_max

    for dp in dps:
        x = onnx_map[dp.onnx] + res_offset[dp.ground_truth]
        xo,xs,xb = x+0*instance_offset, x+1*instance_offset, x+2*instance_offset
        marker = '.' if dp.ground_truth=='unsat' else '^'
        plt.scatter(xo, dp.ot, color="green", label="Original Instances(UNSAT)", marker=marker)
        plt.scatter(xs, dp.st, color="red", label="Splitted Instances(UNSAT)", marker=marker)
        plt.scatter(xb, dp.bt, color="cyan", label="Baseline Instances(UNSAT)", marker=marker)

        plt.plot((xo,xb),(dp.ot, dp.bt), linestyle='--', color='gray', linewidth=line_width)
        plt.plot((xo,xs),(dp.ot, dp.st), linestyle='--', color='gray', linewidth=line_width)
        plt.plot((xb,xs),(dp.bt, dp.st), linestyle='--', color='gray', linewidth=line_width)
    
    plt.yscale(yscale)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    # x2
    plt.plot([xmin, 1], [120, 120], linestyle="--", color='blue')
    plt.text(1, 120, f"{120:.1f}", color='blue', va='center')

    plt.plot([xmin, 1], [180, 180], linestyle="--", color='red')
    plt.text(1, 180, f"{180:.1f}", color='blue', va='center')

    # x4 x6
    plt.plot([1, xmax], [timeout_o, timeout_o], linestyle="--", color='blue')
    plt.text(xmax, timeout_o, f"{timeout_o:.1f}", color='blue', va='center')

    plt.plot([1, xmax], [timeout_s, timeout_s], linestyle="--", color='red')
    plt.text(xmax, timeout_s, f"{timeout_s:.1f}", color='blue', va='center')

    for x in range(1, len(onnx_map)):
        plt.plot([x, x], [ymin, ymax], linestyle="--", color='black')

    onnx_names = [i for i in onnx_map]
    onnx_names_x = [onnx_map[i]-tick_offset+0.5 for i in onnx_map]
    plt.yticks(fontsize=fontsize)
    plt.xticks(onnx_names_x, onnx_names, rotation=0, ha='center', fontsize=fontsize)

    unsat_o = mlines.Line2D([], [], color='green', marker='.', linestyle='None', label='Original Instances (UNSAT)')
    unsat_s = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='Splitted Instances (UNSAT)')
    unsat_b = mlines.Line2D([], [], color='cyan', marker='.', linestyle='None', label='Baseline Instances (UNSAT)')
    sat_o = mlines.Line2D([], [], color='green', marker='^', linestyle='None', label='Original Instances (SAT)')
    sat_s = mlines.Line2D([], [], color='red', marker='^', linestyle='None', label='Splitted Instances (SAT)')
    sat_b = mlines.Line2D([], [], color='cyan', marker='^', linestyle='None', label='Baseline Instances (SAT)')
    timeout_o_line = mlines.Line2D([], [], color='blue', linestyle='--', label='Timeout (Original)')
    timeout_s_line = mlines.Line2D([], [], color='red', linestyle='--', label='Timeout (Splitted)')

    plt.legend(handles=[unsat_o, unsat_s, unsat_b, sat_o, sat_s, sat_b, timeout_o_line, timeout_s_line], loc='upper right', fontsize=fontsize)

    plt.savefig(fname)

import matplotlib.pyplot as plt



def get_stats(vals):
    mean_val = mean(vals)
    median_val = median(vals)
    std_dev = stdev(vals) if len(vals) > 1 else 0  # Avoid error for single value
    min_val = min(vals)
    max_val = max(vals)
    twenty_fifth = sorted(vals)[int(len(vals)*0.25)]
    seventy_fifth = sorted(vals)[int(len(vals)*0.75)]


    return {
        "count": len(vals),
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
        "25th": twenty_fifth,
        "75th": seventy_fifth
    }
    


def get_verifier(fp):
    if "abcrown" in fp.stem:
        return "ABCROWN"
    elif "neuralsat" in fp.stem:
        return "NeuralSAT"
    elif "marabou" in fp.stem:
        return "Marabou"
    else:
        return "Unknown"

def get_benchmark(fp):
    if "reach_prob" in fp.stem:
        return "reach_prob"
    elif "mnist_fc" in fp.stem:
        return "mnist_fc"
    elif "acasxu" in fp.stem:
        return "acasxu"
    elif "oval21" in fp.stem:
        return "oval21"
    elif "cifar2020" in fp.stem:
        return "cifar2020"
    elif "resnet_b" in fp.stem:
        return "resnet_b"
    else:
        return "Unknown"

def get_tool_overhead(fp):
    tool = get_verifier(fp)
    benchmark = get_benchmark(fp)
    return  0
    if tool == "ABCROWN":
        return 3.0
    elif tool == "NeuralSAT":
        return 3.0
    elif tool == "Marabou":
        return 3.0
    else:   
        raise ValueError(f"Unknown tool: {tool}")


def table_by_onnx(raw_dps, benchmark, verifier):
    onnxs = list(set([dp.onnx for dp in raw_dps]))
    onnxs.sort()
    rows = []
    # print(" & ".join(header) + " \\\\")
    # rows.append(header)

    for onnx in onnxs:
        for ground_truth in ["unsat"]:
        # for ground_truth in ["unsat"]:
        # for ground_truth in ["sat", "unsat"]:
            dps = [dp for dp in raw_dps if dp.onnx == onnx and dp.ground_truth == ground_truth]
            if len(dps) == 0:
                rows.append([onnx, verifier, "\\tool{}", "0", "", "", "", "", "", "0"])
                rows.append([onnx, verifier, "baseline", "0", "", "", "", "", "", "0"])
                continue
            split_slowdowns = [dp.st/dp.ot for dp in dps]
            baseline_slowdowns = [dp.bt/dp.ot for dp in dps]
            split_stats = get_stats(split_slowdowns)
            baseline_stats = get_stats(baseline_slowdowns)

            split_changed_to_timeout = len([dp for dp in dps if dp.o_res!="timeout" and dp.s_res=="timeout"])
            baseline_changed_to_timeout = len([dp for dp in dps if dp.o_res!="timeout" and dp.b_res=="timeout"])

            # print(f"<<<{benchmark}>>>")
            # print([(dp.onnx, dp.vnnlib, dp.st/dp.ot) for dp in sorted(dps, key=lambda x: x.st/x.ot)][-2:]) 
            # print([(dp.onnx, dp.vnnlib, dp.bt/dp.ot) for dp in sorted(dps, key=lambda x: x.bt/x.ot)][-2:])
            # print(f"<<<{onnx}>>>")
            # print(median([stats_DP.get_size_incr(dp.onnx, dp.vnnlib) for dp in dps]))

            cols1 = [onnx, verifier, split_stats["count"], "\\tool{}", split_stats["min"],     split_stats["25th"],    split_stats["median"],      split_stats["75th"],    split_stats["max"], split_changed_to_timeout]
            cols2 = [onnx, verifier, baseline_stats["count"], "baseline", baseline_stats["min"],     baseline_stats["25th"], baseline_stats["median"],   baseline_stats["75th"], baseline_stats["max"], baseline_changed_to_timeout]
            cols1 = [f"{i:.2f} x" if isinstance(i, (float)) else i for i in cols1]
            cols2 = [f"{i:.2f} x" if isinstance(i, (float)) else i for i in cols2]
            cols1 = [f"{i}" if isinstance(i, (int)) else i for i in cols1]
            cols2 = [f"{i}" if isinstance(i, (int)) else i for i in cols2]
            rows.append(cols1)
            rows.append(cols2)
    rows.sort()
    return rows


def table(raw_dps, benchmark, verifier):
    rows = []
    # print(" & ".join(header) + " \\\\")
    # rows.append(header)

    for ground_truth in ["unsat"]:
    # for ground_truth in ["unsat"]:
    # for ground_truth in ["sat", "unsat"]:
        dps = [dp for dp in raw_dps if dp.ground_truth == ground_truth]
        if len(dps) == 0:
            rows.append([benchmark, verifier, "\\tool{}", "0", "", "", "", "", "", "0"])
            rows.append([benchmark, verifier, "baseline", "0", "", "", "", "", "", "0"])
            continue
        split_slowdowns = [dp.st/dp.ot for dp in dps]
        baseline_slowdowns = [dp.bt/dp.ot for dp in dps]
        split_stats = get_stats(split_slowdowns)
        baseline_stats = get_stats(baseline_slowdowns)

        split_changed_to_timeout = len([dp for dp in dps if dp.o_res!="timeout" and dp.s_res=="timeout"])
        baseline_changed_to_timeout = len([dp for dp in dps if dp.o_res!="timeout" and dp.b_res=="timeout"])



        cols1 = [benchmark, verifier, split_stats["count"], "\\tool{}", split_stats["min"],     split_stats["25th"],    split_stats["median"],      split_stats["75th"],    split_stats["max"], split_changed_to_timeout]
        cols2 = [benchmark, verifier, baseline_stats["count"], "baseline", baseline_stats["min"],     baseline_stats["25th"], baseline_stats["median"],   baseline_stats["75th"], baseline_stats["max"], baseline_changed_to_timeout]
        cols1 = [f"{i:.2f} x" if isinstance(i, (float)) else i for i in cols1]
        cols2 = [f"{i:.2f} x" if isinstance(i, (float)) else i for i in cols2]
        cols1 = [f"{i}" if isinstance(i, (int)) else i for i in cols1]
        cols2 = [f"{i}" if isinstance(i, (int)) else i for i in cols2]
        rows.append(cols1)
        rows.append(cols2)
    rows.sort()
    return rows




if __name__ == "__main__":
    res_files = [
        # res_root / "exp1/reach_prob/reach_prob_abcrown.csv",
        # res_root / "exp1/reach_prob/reach_prob_neuralsat.csv",
        # res_root / "exp1/reach_prob/reach_prob_marabou.csv",
        
        # res_root / "exp1/mnist_fc/mnist_fc_abcrown.csv",
        # res_root / "exp1/mnist_fc/mnist_fc_neuralsat.csv",
        # res_root / "exp1/mnist_fc/mnist_fc_marabou.csv",

        
        # res_root / "exp1/oval21_2l/oval21_2l_marabou.csv",


        res_root / "exp1/oval21_2l/oval21_2l_abcrown.csv",
        res_root / "exp1/oval21_2l/oval21_2l_neuralsat.csv",
        # res_root / "exp1/oval21_16/oval21_16_abcrown.csv",
        # res_root / "exp1/oval21_16/oval21_16_neuralsat.csv",
        # res_root / "exp1/oval21/oval21_abcrown.csv",
        # res_root / "exp1/oval21/oval21_neuralsat.csv",


        # res_root / "exp1/oval21/oval21_marabou.csv",
        

        # res_root / "exp1/resnet_b/resnet_b_neuralsat.csv",
        # res_root / "exp1/resnet_b/resnet_b_abcrown.csv",

        # res_root / "exp1/cifar2020/cifar2020_abcrown.csv",
        # res_root / "exp1/cifar2020/cifar2020_neuralsat.csv",
        # res_root / "exp1/cifar2020/cifar2020_marabou.csv",
    ]
    
    stats_fps = [
        "/home/lli/tools/relusplitter/experiment/mnist_fc_info.csv",
    ]

    for f in stats_fps:
        with open(f, "r") as f:
            lines = f.readlines()
        for line in lines[1:]:
            stats_DP(line)
    


    header = ["verifier", "onnx", "split/baseline", "min", "max", "mean", "median", "count", "TOed"]
    # tab = [header]
    tab = []
    for fp in res_files:
        try:
            print("=====================================")
            print(f"<<<{fp.stem}>>>")
            res = [i for i in load_dps(fp) if i.valid and i.ground_truth == "unsat" ]
            print(min([dp.ot for dp in res]))

            # check for conflicting results
            for dp in res:
                if dp.o_res == "sat":
                    assert dp.s_res != "unsat"
                    assert dp.b_res != "unsat"
                if dp.o_res == "unsat":
                    assert dp.s_res != "sat"
                    assert dp.b_res != "sat"
            # OTed

            benchmark, verifier = get_benchmark(fp), get_verifier(fp)
            threshold = 0


            filtered_dps = [dp for dp in res if dp.ot > threshold]
            split_Toed    = [dp for dp in filtered_dps if dp.o_res != "timeout" and dp.s_res == "timeout"]
            baseline_Toed = [dp for dp in filtered_dps if dp.o_res != "timeout" and dp.b_res == "timeout"]

            split_to_use = [dp for dp in filtered_dps if dp not in split_Toed]
            baseline_to_use = [dp for dp in filtered_dps if dp not in baseline_Toed]
            # split_to_use = filtered_dps
            # baseline_to_use = filtered_dps

            print(f"Threshold: {threshold}")
            tab = table_by_onnx(filtered_dps, benchmark, verifier)
            # tab = table(filtered_dps, benchmark, verifier)
            print("====")
            for row in tab:
                print(" & ".join(row) + " \\\\")

        except Exception as e:
            print(f"ERROR: {e}")
            continue