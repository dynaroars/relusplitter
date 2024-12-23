from helpers import *
from settings import *

# def check_runned(log_fname, benchmark_instances):


remaining = get_remaining("/home/lli/tools/relusplitter/experiment/results/exp4/mnist_fc~nnenum~stable~random~2~lambda.csv",
             get_instances(benchmarks["mnist_fc"]))