import logging
from pathlib import Path
from typing import Union, Tuple

import onnx
import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from onnx import helper, numpy_helper, TensorProto
from onnx2pytorch import ConvertModel

from copy import copy, deepcopy

from ..utils.onnx_utils import truncate_onnx_model, compute_model_bound

SINGLE_INPUT_OPS = ["Relu"]
default_logger = logging.getLogger(__name__)

class WarppedOnnxModel():
    def __init__(self, model: onnx.ModelProto, force_rename = False, logger=default_logger)-> None:
        self.logger = logger
        self.force_rename = force_rename

        self._sanity_check(model)

        self._model = model
        self._initializers = [initializer for initializer in model.graph.initializer]
        self._initializers_mapping = {initializer.name: initializer for initializer in model.graph.initializer}
        self._nodes = [node for node in model.graph.node]
        self._nodes_mapping = {node.name: node for node in model.graph.node}
        # self.node_accepts_input = {node.input[0]: node for node in model.graph.node}
        self._node_produce_output = {output: node for node in model.graph.node for output in node.output}

    @property
    def nodes(self):
        return copy(self._nodes)

    def info(self):
        s = "\n"
        for node in self._nodes:
            s += f"\t\tNode name: {node.name}, Node type: {node.op_type}\n"
            s += f"\t\t\tInputs: {node.input}\n"
            s += f"\t\t\tOutputs: {node.output}\n"
        self.logger.info(s)



    def _sanity_check(self, model: onnx.ModelProto):

        # check for duplicate node name or no node name
        node_names = [node.name for node in model.graph.node]
        has_duplicate_node_name = (len(node_names) != len(set(node_names)))
        has_node_w_empty_name    = any((node.name == "") for node in model.graph.node)
        if has_duplicate_node_name or has_node_w_empty_name or self.force_rename:
            for i, node in enumerate(model.graph.node):
                node.name = f"{node.op_type}_{i}"
                # print(f"Assigned name '{node.name}' to node of type '{node.op_type}'")
        for node in model.graph.node:
            # each node only produce one output
            assert len(node.output) == 1, f"Node {node.name} produces more than one output"
            # allowed op_types
            # ...

    def get_prior_node(self, node):
        assert node.op_type in SINGLE_INPUT_OPS
        assert len(node.input) == 1
        return self._node_produce_output[node.input[0]]

    def get_bound_of(self, input_bound:BoundedTensor, tensor_name: str, method: str = "backward") -> Tuple[torch.Tensor, torch.Tensor]:
        trunated_model = truncate_onnx_model(self._model, tensor_name)
        return compute_model_bound(trunated_model, input_bound, method=method)
    
    def get_gemm_wb(self, node):
        # Get the weights and bias for a GEMM node
        assert node.op_type == "Gemm"
        _, w, b = node.input
        w = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[w]) )
        b = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[b]) )
        return w,b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = ConvertModel(self._model)
        model = BoundedModule(model, x)
        return model(x)

    def save(self, fname: Path):
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        onnx.save(self._model, fname)
        return fname
    
    def generate_updated_model(self, nodes_to_replace, additional_nodes, additional_initializers, 
                      graph_name=f"bruh_split_merge",
                      producer_name="ReluSplitter"):

        # maybe some assertions
        
        
        new_nodes = []
        new_initializers = []


        model_in = self._model.graph.input
        model_out = self._model.graph.output

        model_input_name = model_in[0].name
        model_output_name = model_out[0].name

        visited = set([model_input_name])
        avaliable_nodes = [n for n in self._nodes if n not in nodes_to_replace] + additional_nodes
        avaliable_initializers = {i.name:i for i in (self._initializers+additional_initializers)}
        while model_output_name not in visited:
            updated = False
            for node in avaliable_nodes:
                if node in new_nodes:
                    continue
                else:
                    if all((i in visited or i in avaliable_initializers) for i in node.input):
                        new_nodes.append(node)
                        new_initializers.extend([avaliable_initializers[i] for i in node.input if i in avaliable_initializers])
                        visited.update(node.output)
                        updated = True
            if not updated:
                raise ValueError(f"Can't find a node to add to the new model")


        new_graph = onnx.helper.make_graph(
            new_nodes,
            graph_name,
            model_in,
            model_out,
            new_initializers
        )

        new_model = helper.make_model(new_graph, producer_name=producer_name)
        return WarppedOnnxModel(new_model, force_rename=False, logger=self.logger)
                        