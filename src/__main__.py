import torch
import onnx
import torch.nn as nn
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .utils.misc import check_model_equivalency


def main():
    # a = onnx.load('split_test_non.onnx')
    # b = onnx.load('split_test.onnx')
    # print(check_model_equivalency(a, b, a.graph.input[0].type.tensor_type.shape))


    a = onnx.load('ACASXU_run2a_1_3_batch_2000.onnx')
    b = onnx.load('resaved_acasxu.onnx')
    print(check_model_equivalency(a, b, (1,1,1,5)))


    # a = onnx.load('/home/lli/sroll_storage/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_4_9_batch_2000.onnx')
    # b = onnx.load('/home/lli/sroll_storage/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_3_1_batch_2000.onnx')
    # print(check_model_equivalency(a, b, (1,1,1,5)))

    print('Checking ONNX models')

if __name__ == '__main__':
    main()