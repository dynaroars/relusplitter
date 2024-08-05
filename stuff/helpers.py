import onnx
from onnx import helper, numpy_helper, TensorProto
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from onnx2pytorch import ConvertModel

def get_initializer(self, name):
    # Get the initializer tensor by name
    for initializer in self.graph.initializer:
        if initializer.name == name:
            return initializer

def get_gemm_wb(self, node):
    # Get the weights and bias for a GEMM node
    assert node.op_type == "Gemm"
    _, w, b = node.input
    w = torch.from_numpy( numpy_helper.to_array(get_initializer(self,w)) )
    b = torch.from_numpy( numpy_helper.to_array(get_initializer(self,b)) )
    return w,b

def get_node_produce_output(model, output_name):
    temp = []
    for node in model.graph.node:
        if output_name in node.output:
            temp.append(node)
    if len(temp) == 1:
        return temp[0]
    else:
        raise ValueError(f"Found {len(temp)} nodes producing output '{output_name}'")
    return None

def get_node_accept_input(model, input_name):
    temp = []
    for node in model.graph.node:
        if input_name in node.input:
            temp.append(node)
    if len(temp) == 1:
        return temp[0]
    else:
        raise ValueError(f"Found {len(temp)} nodes consuming input '{input_name}'")
    return None

def get_unstable_ReLU_mask(lb, ub):
    # lb and ub are two tensors of the same shape
    # find the ReLU nodes that are unstable
    eps = 1e-7
    # return (lb < eps) & (ub > eps)
    assert torch.all(lb<=ub)
    return lb>=0

def get_bounds(model, bounded_input, method="backward"):
    if not isinstance(model, ConvertModel):
        model = ConvertModel(model)
    model = BoundedModule(model, bounded_input)
    lb, ub = model.compute_bounds(x=(bounded_input,), method=method)

    return lb, ub

def forward(model, bounded_input):
    if not isinstance(model, ConvertModel):
        model = ConvertModel(model)
    model = BoundedModule(model, bounded_input)
    return model(bounded_input)

def new_input(x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
    assert torch.all(x_L <= x_U)
    return BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U))

def get_prior_node(model, node):
    single_input_ops = ["Relu"]
    assert node.op_type in single_input_ops
    assert len(node.input) == 1
    return get_node_produce_output(model, node.input[0])

def get_splitable_ReLU(model):
    splitable_nodes = []
    for node in model.graph.node:
        if node.op_type == "Relu":
            splitable_nodes.append(node)
    return splitable_nodes

def add_names_to_nodes(model):
    for i, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f"{node.op_type}_{i}"
            print(f"Assigned name '{node.name}' to node of type '{node.op_type}'")

    return model

def truncate_onnx_model(onnx_model, target_ooutput_name):
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
    add_node_and_dependencies(target_ooutput_name)

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
    output_tensor = helper.make_tensor_value_info(target_ooutput_name, TensorProto.FLOAT, None)

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