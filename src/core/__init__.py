import logging
from pathlib import Path

import onnx
import torch
import onnxruntime as ort
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto


class NN_mod_instance():
    def __init__(self, onnx_model:onnx.ModelProto, spec: Path, logger:logging.Logger) -> None:
        self._model = onnx_model
        self._graph = self._model.graph
        self.layers_req_bounds = set()
        # construct bounded input
        self.bounded_input = None


    def compute_all_bounds(self, method:str = "forward"):
        if self.layers_req_bounds is None:
            return []
        trucated_models = self.build_trucated_model(self.layers_req_bounds)
        bounded_models = [BoundedModule(m, self.bounded_input) for m in trucated_models]
        bounds = []
        for m in bounded_models:
            bounds.append(m.compute_bounds())
        return bounds


    def build_trucated_model(self, layers):


    # stable ReLU splitting
    @property
    def splitable_relu_nodes(self):
        if self._splitable_relu_nodes is None:
            temp = []
            for node in self._graph.node:
                if node.op_type == 'Relu':
                    temp.append(node)
            self._splitable_relu_nodes = temp
        return self._splitable_relu_nodes
    
    def gen