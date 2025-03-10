from relu_splitter.utils.read_vnnlib import *


fp = "/home/lli/tools/relusplitter/data/acasxu_converted/vnnlib/prop_7.vnnlib"

loaded_vnnlib = read_vnnlib(fp)

# output a list containing 2-tuples:
#     1. input ranges (box), list of pairs for each input variable
#     2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
#                       Each element in the list is a term in a disjunction for the specification.

# print(loaded_vnnlib[0][1][0])
# print(loaded_vnnlib[0][1][1])

print(loaded_vnnlib)
