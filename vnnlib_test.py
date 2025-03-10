from relu_splitter.spec import *


fp = "/home/lli/tools/relusplitter/data/acasxu_converted/vnnlib/prop_7.vnnlib"

loaded_vnnlib = read_vnnlib(fp)

# output a list containing 2-tuples:
#     1. input ranges (box), list of pairs for each input variable
#     2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
#                       Each element in the list is a term in a disjunction for the specification.

test = warpped_vnnlib(loaded_vnnlib[0])

print(test.serialize_postCondition())


# ; unsafe if strong left is minimial or strong right is minimal
# (assert (or
#     (and (<= Y_3 Y_0) (<= Y_3 Y_1) (<= Y_3 Y_2))
#     (and (<= Y_4 Y_0) (<= Y_4 Y_1) (<= Y_4 Y_2))
# ))
