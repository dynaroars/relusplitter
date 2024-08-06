import logging
import onnx
import torch
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

logger = logging.getLogger(__name__)


def check_model_closeness(m1, m2, input_shape, n=50, **kwargs):
    for _ in range(n):
        x = torch.randn(input_shape)
        y1,y2 = m1.forward(x), m2.forward(x)
        if not torch.allclose(y1, y2, **kwargs):
            logger.error(f"Outputs are not the same for input {x}\n{y1}\n{y2}")
            return False
    return True

def compute_model_bound(model: onnx.ModelProto, bounded_input: BoundedTensor, method="backward"):
    model = ConvertModel(model)
    model = BoundedModule(model, bounded_input)
    lb, ub = model.compute_bounds(x=(bounded_input,), method=method)
    return lb, ub


def truncate_onnx_model(onnx_model, target_output_name):
    # Load the ONNX model if it's a path, otherwise assume it's already an ONNX model object
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    
    # Create dictionaries to hold the nodes and the initializers
    truncated_nodes = []
    truncated_initializers = []
    required_initializers = set()
    required_inputs = set()
    
    # Keep track of visited nodes to avoid duplicates
    visited = set()

    def add_node_and_dependencies(node_name):
        if node_name in visited:
            return
        visited.add(node_name)

        # Find the node by its output name
        node = next((n for n in onnx_model.graph.node if node_name in n.output), None)
        if node is None:
            return

        # Add the node to the truncated list
        truncated_nodes.append(node)

        # Add the node's inputs to the required inputs set
        for input_name in node.input:
            required_inputs.add(input_name)
            # Recursively add dependencies
            add_node_and_dependencies(input_name)

    # Start from the target node and work backwards to find all required nodes
    add_node_and_dependencies(target_output_name)

    # Collect initializers for the truncated nodes
    for initializer in onnx_model.graph.initializer:
        if initializer.name in required_inputs:
            truncated_initializers.append(initializer)

    # Collect inputs for the truncated nodes
    truncated_inputs = []
    for input_tensor in onnx_model.graph.input:
        if input_tensor.name in required_inputs:
            truncated_inputs.append(input_tensor)

    # Define the output tensor for the truncated model
    output_tensor = helper.make_tensor_value_info(target_output_name, TensorProto.FLOAT, None)

    # Create the graph for the truncated model
    truncated_nodes.reverse()
    truncated_inputs.reverse()
    truncated_initializers.reverse()
    truncated_graph = helper.make_graph(
        nodes=truncated_nodes,
        name="truncated_graph",
        inputs=truncated_inputs,
        outputs=[output_tensor],
        initializer=truncated_initializers
    )

    # Create the truncated model
    truncated_model = helper.make_model(truncated_graph, producer_name="truncated_model")

    return truncated_model


