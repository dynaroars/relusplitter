import torch
from torch import nn
import time

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def get_hidden_bounds(self, device):
    lower_bounds, upper_bounds = {}, {}
    # print(list(set(self.layers_requiring_bounds + self.split_nodes)))
    print("LayerReqBounds: ", self.layers_requiring_bounds)
    print("GetSplitNodes: ", self.get_split_nodes())

    for layer in list(set(self.layers_requiring_bounds + self.get_split_nodes()[0])):
    # for layer in self.nodes().mapping.values():
        print("Name: ", layer.name)
        print("Bounds:", layer.lower, layer.upper)
        lower_bounds[layer.name] = layer.lower.detach().to(device)
        upper_bounds[layer.name] = layer.upper.detach().to(device)

    return lower_bounds, upper_bounds



class SplitTest(nn.Module):
    def __init__(self):
        super(SplitTest, self).__init__()
        self.w1 = torch.tensor([[2.]])
        self.b1 = torch.tensor([-1.])
        self.w2 = torch.tensor([[1.]])
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = x.matmul(self.w1.t()) + self.b1
        z2 = z1.matmul(self.w2.t())
        return self.relu(z2)
    
class SplitTest2(nn.Module):
    def __init__(self):
        super(SplitTest2, self).__init__()
        self.w1 = torch.tensor([[2.], [-2.]])
        self.b1 = torch.tensor([-2.5, 2.5])
        self.w2 = torch.tensor([[1., -1.]])
        self.b2 = torch.tensor([1.5])
        self.w3 = torch.tensor([[1.]])
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = x.matmul(self.w1.t()) + self.b1
        z1 = self.relu(z1)
        z2 = z1.matmul(self.w2.t()) + self.b2
        z2 = self.relu(z2)
        z3 = z2.matmul(self.w3.t())
        return z3


methods = ['ibp', 'ibp+backward', 'crown', 'forward', 'forward+backward', 'crown-optimized', 'forward-optimized']
eps = 0.4
my_input = torch.tensor([[1.5]])
ptb = PerturbationLpNorm(eps=eps)
bounded_input = BoundedTensor(my_input, ptb)
repeats = 10


model_1 = SplitTest()
model_1 = BoundedModule(model_1, my_input)

model_2 = SplitTest2()
model_2 = BoundedModule(model_2, my_input)

for method in methods:
    print(f"============ Method: {method} ========")
    # print(f"Input: {my_input}")
    # print(f"eps: {eps}")

    try:
        t1 = time.time()
        for i in range(repeats):
            lb1, ub1 = model_1.compute_bounds(x=(bounded_input,), method=method)
        t2 = time.time()
        print("== M1 ==")
        # print("Time taken: ", t2 - t1)
        print(lb1, ub1, t2 - t1)

        # call this 10000 times and time it
        t1 = time.time()
        for i in range(repeats):
            lb2, ub2 = model_2.compute_bounds(x=(bounded_input,), method=method)
        t2 = time.time()
        print("== M2 ==")
        # print("Time taken: ", t2 - t1)
        print(lb2, ub2, t2 - t1)
    except Exception as e:
        print("Error, skipping")



for i in range(-400, 400):
    my_input = torch.tensor([[i/100]])
    assert model_1(my_input) == model_2(my_input)


# model_2.compute_bounds(x=(bounded_input,), method="forward")
# # model_2.compute_bounds(x=(bounded_input,), method="backward")
# lb, ub = get_hidden_bounds(model_2, 'cpu')
# print(lb)
# print(ub)
# print(dir(model_2.nodes().mapping['/input']))
# print(type(model_2.nodes().mapping['/input']))

import onnx2pytorch
from onnx2pytorch import ConvertModel
import onnx
import torch

# dump model to onnx
model = SplitTest2()
model.eval()
dummy_input = torch.randn(1, 1)
onnx_path = 'split_test.onnx'
torch.onnx.export(model, dummy_input, onnx_path)
# load onnx model
onnx_model = onnx.load(onnx_path)
# onnx_model = onnx.load("/home/lli/sroll_storage/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_2_2_batch_2000.onnx")
pytorch_model = ConvertModel(onnx_model)

print(pytorch_model)

print(dir(pytorch_model))