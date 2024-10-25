import logging
import onnx
import torch
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

logger = logging.getLogger(__name__)

def check_model_closeness_ort(m1, m2, input_names, input_shapes, n=50):
    import onnxruntime as ort
    sess1 = ort.InferenceSession(m1.SerializeToString())
    sess2 = ort.InferenceSession(m2.SerializeToString())
    worst_diff = 0
    for i in range(n):
        x = {name: np.random.randn(*shape).astype(np.float32) for name, shape in zip(input_names, input_shapes)}
        y1 = sess1.run(None, x)
        y2 = sess2.run(None, x)
        for y1_, y2_ in zip(y1, y2):
            worst_diff = max(worst_diff, np.abs(y1_-y2_).max())
            if not np.allclose(y1_, y2_):
                logger.error(f"Outputs are not the same for input {x}\n{y1_}\n{y2_}")
                logger.error(f"Diff: {np.abs(y1_-y2_).max()}")
                logger.error(f"{i}/{n} tests passed")
                return False, worst_diff
    return True, worst_diff


def check_model_closeness(m1, m2, input_shape, n=5, **kwargs):
    worst_diff = 0
    for i in range(n):
        x = torch.randn(input_shape)
        y1,y2 = m1.forward(x), m2.forward(x)
        worst_diff = max(worst_diff, torch.abs(y1-y2).max())
        if not torch.allclose(y1, y2, **kwargs):
            logger.error(f"Outputs are not the same for input {x}\n{y1}\n{y2}")
            logger.error(f"Diff: {torch.abs(y1-y2).max()}")
            logger.error(f"{i}/{50} tests passed")
            return False, worst_diff
    return True, worst_diff

def check_model_closeness_gpu(m1, m2, input_shape, n=5, **kwargs):
    worst_diff = 0
    for i in range(n):
        x = torch.randn(input_shape).cuda()
        y1,y2 = m1.forward_gpu(x), m2.forward_gpu(x)
        worst_diff = max(worst_diff, torch.abs(y1-y2).max())
        if not torch.allclose(y1, y2, **kwargs):
            logger.error(f"Outputs are not the same for input {x}\n{y1}\n{y2}")
            logger.error(f"Diff: {torch.abs(y1-y2).max()}")
            logger.error(f"{i}/{50} tests passed")
            return False, worst_diff
    return True, worst_diff

def compute_model_bound(model: onnx.ModelProto, bounded_input: BoundedTensor, method="backward"):
    model = ConvertModel(model)
    if "input" not in model.input_names:
        model.input_names = ["input"]
    
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


