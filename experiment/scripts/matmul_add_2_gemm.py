import sys

import onnx

import torch

from pathlib import Path

from relu_splitter.utils.onnx_utils import *

from relu_splitter.model import WarppedOnnxModel

from onnx2pytorch import ConvertModel


if __name__ == "__main__":
    input_onnx = Path(sys.argv[1])
    output_onnx = Path(sys.argv[2])

    model = WarppedOnnxModel(onnx.load(input_onnx))
    graph_name = model._model.graph.name
    input_names = [input.name for input in model._model.graph.input]
    input_shape = list(model.input_shapes.values())
    print(f"Input names: {input_names}")
    print(f"Input shape: {input_shape}")
    assert len(input_shape) == 1 and len(input_names) == 1
    input_name = input_names[0]
    input_shape = input_shape[0]

    nodes2merge = []
    for node in model.nodes:
        if node.op_type == "Add":
            node_input = model.get_node_inputs_no_initializers(node)
            prior = model._node_produce_output.get(node_input[0], None)
            if prior is None:
                continue
            if prior.op_type == "MatMul":
                nodes2merge.append((node, prior))
    
    gemm_nodes = []
    for node, prior in nodes2merge:
        print(f"Converting MatMul+Add to Gemm for {node.name}")
        matmul_initializers = model.get_node_initializers(prior) 
        add_initializers    = model.get_node_initializers(node)
        assert len(matmul_initializers) == 1
        assert len(add_initializers) == 1

        n_input = model.get_node_inputs_no_initializers(prior)[0]
        n_output= node.output[0]

        gemm_w = matmul_initializers[0]
        gemm_b = add_initializers[0]
        gemm_node = onnx.helper.make_node(
            'Gemm',
            inputs=[n_input, gemm_w, gemm_b],
            outputs=[n_output],
            name=model.gen_node_name("Gemm")
        )
        gemm_nodes.append(gemm_node)


    new_model = model.generate_updated_model([n for nodes in nodes2merge for n in nodes], gemm_nodes, [], graph_name=f"{graph_name}_matmul+add_2_gemm", producer_name="ReluSplitter_util")

    try:
        status, diff = check_model_closeness(model, new_model,input_shape)
        print(f"Model closeness: {status}")
        print(f"Diff: {diff}")
    except Exception as e:
        print(f"Model closeness check failed: {e}")

    new_model.save(output_onnx)

