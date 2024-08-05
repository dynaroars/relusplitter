import onnx
from onnx import helper, numpy_helper, TensorProto

from helpers import *

import logging
from pathlib import Path

import onnx
import torch
import onnxruntime as ort
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from onnx2pytorch import ConvertModel
from src.utils.read_vnnlib import read_vnnlib








# model = onnx.load("/home/lli/tools/reluspliter/tests/mnist_fc/onnx/mnist-net_256x2.onnx")
# model = onnx.load("/home/lli/tools/reluspliter/tests/mnist_fc/onnx/mnist-net_256x4.onnx")
model = onnx.load("/home/lli/tools/relusplitter/tests/mnist_fc/onnx/mnist-net_256x6.onnx")
model = add_names_to_nodes(model)

converted_model = ConvertModel(model)

# input_ranges, spec = read_vnnlib("/home/lli/tools/reluspliter/tests/mnist_fc/vnnlib/prop_1_0.05.vnnlib")[0]
input_ranges, spec = read_vnnlib("/home/lli/tools/relusplitter/tests/mnist_fc/vnnlib/prop_2_0.03.vnnlib")[0]
# read_vnnlib("prop_2.vnnlib")
# read_vnnlib("prop_3.vnnlib")
# read_vnnlib("prop_4.vnnlib")


lb, ub = [[[[i[0] for i in input_ranges]]]], [[[[i[1] for i in input_ranges]]]]
bounded_input = new_input(torch.tensor(lb), torch.tensor(ub))

bounded_model = BoundedModule(converted_model, torch.tensor(lb))


for node in model.graph.node:
        print(f"Node name: {node.name}, Node type: {node.op_type}")
        for i, input_name in enumerate(node.input):
            print(f"  Input {i}: {input_name}")
        for i, output_name in enumerate(node.output):
            print(f"  Output {i}: {output_name}")

splitable_nodes = get_splitable_ReLU(model)
truncate_input_names = []
truncate_output_names = []
pre_relu_ouputs = []

for node in splitable_nodes:
    assert len(node.output) == 1
    assert len(node.input) == 1
    truncate_input_names.append(get_prior_node(model, node).input[0])
    truncate_output_names.append(node.output[0])
    pre_relu_ouputs.append(node.input[0])

print("Truncating model on output names:", truncate_output_names)
print(truncate_input_names)


truncated_models = [truncate_onnx_model(model, output_name) for output_name in truncate_output_names]

method = "forward+backward"

# print(f"Original.onnx")
# print(bounded_model.compute_bounds(x=(bounded_input,), method=method)[0])


# converted_model = ConvertModel( truncate_onnx_model(model, "12") )
# bounded_model = BoundedModule(converted_model, torch.tensor(lb))

for truncated_model, pre_relu_ouput, input_name, output_name in zip(truncated_models, pre_relu_ouputs, truncate_input_names, truncate_output_names):
    print(f"TruncatedModel_{output_name}.onnx")
    # Post activation output bounds
    alb, aub = (get_bounds(truncated_model, bounded_input, method=method))
    split_mask = get_unstable_ReLU_mask(alb, aub).squeeze()     # unstable ReLUs to be splitted
    n_split = split_mask.sum().item()                           # number of unstable ReLUs to be splitted
    # Input bound for the layer to be splitted
    split_lb, split_ub = get_bounds(truncate_onnx_model(model, input_name), bounded_input, method="forward+backward")
    # split loc is an random value between lb and ub
    # split loc is a point at the input interval, not the output interval
    split_loc = torch.rand_like(split_lb) * (split_ub - split_lb) + split_lb
    split_loc = split_loc.squeeze()
    # print("split_loc set to:", split_loc)

    pre_relu_layer = get_node_produce_output(model, pre_relu_ouput) # layer to be splitted
    assert pre_relu_layer.op_type == "Gemm"
    grad, offset = get_gemm_wb(model,pre_relu_layer) # weights and bias, we need this to construct the split structure

    # create the new split layer's weights and bias, first craet empty tensors
    num_out, num_in = grad.shape
    split_weights = torch.zeros((num_out + n_split, num_in))
    split_bias = torch.zeros(num_out + n_split)
    # create the merge layer weights and bias, first craet empty tensors
    merge_weights = torch.zeros((num_out, num_out + n_split))
    merge_bias = torch.zeros(num_out)
    # copy the original weights and bias to the new split layer
    print("number of ReLUs to be splitted:", n_split)
    print(split_loc.shape)
    print(split_weights.shape)
    print(split_bias.shape)
    idx = 0
    for i in range(len(split_mask)):
        w = grad[i]
        b = grad[i] @ split_loc

        
        if split_mask[i]:
            # if the neuron requires splitting
            # sum up the splitted neurons and forward it to the output
            split_weights[idx]      = w
            split_weights[idx + 1]  = -w

            split_bias[idx]         = -b
            split_bias[idx + 1]     = b

            merge_weights[i][idx] = 1
            merge_weights[i][idx + 1] = -1

            merge_bias[i] = offset[i] + b
            
            idx += 2
        else:
            # if the neuron doesnt require splitting
            # directly forward it to the output

            split_weights[idx] = grad[i]
            split_bias[idx] = offset[i]

            merge_weights[i][idx] = 1
            # merge_bias[i] = 0

            idx += 1

    print("split_weights", split_weights.shape)
    print("split_bias", split_bias.shape)
    print("merge_weights", merge_weights.shape)
    print("merge_bias", merge_bias.shape)
    
    # start making onnx model
    new_nodes = []
    new_initializers = []
    model_in = model.graph.input[0].name
    model_out = model.graph.output[0].name
    split_at_input = input_name
    merge_at_output = output_name

    visited = set([model_in])
    initializers = {i.name: i for i in model.graph.initializer}
    while model_out not in visited:
        for node in model.graph.node:
            if all((i in visited or i in initializers) for i in node.input):
                if node.input[0] == split_at_input:
                    # insert the split layer
                    split_node = onnx.helper.make_node(
                        'Gemm',
                        inputs=[split_at_input, 'split_weights', 'split_bias'],
                        outputs=['split_layer_pre_relu'],
                        name='split_layer',
                        alpha=1.0,
                        beta=1.0,
                        transB=1
                    )
                    # relu node
                    split_relu_node = onnx.helper.make_node(
                        'Relu',
                        inputs=['split_layer_pre_relu'],
                        outputs=['split_layer_post_relu'],
                        name='relu_split'
                    )

                    merge_node = onnx.helper.make_node(
                        'Gemm',
                        inputs=['split_layer_post_relu', 'merge_weights', 'merge_bias'],
                        outputs=['merge_layer_pre_relu'],
                        name='merge_layer',
                        alpha=1.0,
                        beta=1.0,
                        transB=1
                    )
                    merge_relu_node = onnx.helper.make_node(
                        'Relu',
                        inputs=['merge_layer_pre_relu'],
                        outputs=[merge_at_output],
                        name='relu_merge'
                    )
                    new_nodes.extend([split_node, split_relu_node, merge_node, merge_relu_node])

                    new_initializers.append(
                        onnx.helper.make_tensor(
                            'split_weights',
                            TensorProto.FLOAT,
                            split_weights.shape,
                            split_weights.flatten().tolist()
                        )
                    )
                    new_initializers.append(
                        onnx.helper.make_tensor(
                            'split_bias',
                            TensorProto.FLOAT,
                            split_bias.shape,
                            split_bias.flatten().tolist()
                        )
                    )
                    new_initializers.append(
                        onnx.helper.make_tensor(
                            'merge_weights',
                            TensorProto.FLOAT,
                            merge_weights.shape,
                            merge_weights.flatten().tolist()
                        )
                    )
                    new_initializers.append(
                        onnx.helper.make_tensor(
                            'merge_bias',
                            TensorProto.FLOAT,
                            merge_bias.shape,
                            merge_bias.flatten().tolist()
                        )
                    )

                    visited.add(merge_at_output)
                elif node.output[0] == merge_at_output:
                    continue
                else:
                    new_nodes.append(node)
                    new_initializers.extend([initializers[i] for i in node.input if i in initializers])
                    visited.update(node.output)

    new_graph = onnx.helper.make_graph(
        new_nodes,
        'split_merge',
        model.graph.input,
        model.graph.output,
        new_initializers
    )

    new_model = helper.make_model(new_graph, producer_name="split_merge")

    onnx.save(new_model, f"split_merge_{output_name}.onnx")

    
    # compute bound of new model and original model
    method = "forward+backward"
    print(f"Original.onnx")
    print(get_bounds(model, bounded_input, method=method))
    print(forward(model, bounded_input))
    print(f"split_merge_{output_name}.onnx")
    print(get_bounds(new_model, bounded_input, method=method))
    print(forward(new_model, bounded_input))
