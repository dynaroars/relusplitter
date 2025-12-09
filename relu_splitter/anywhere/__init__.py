# standard libs
import random
from functools import reduce

# third-party libs
import onnx
import torch
import numpy as np
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

# my libs
from relu_splitter import TOOL_NAME
from relu_splitter.utils.read_vnnlib import read_vnnlib
from relu_splitter.utils.onnx_utils import check_models_closeness
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.utils.logger import default_logger
from relu_splitter.anywhere.gemm import Rsplitter_Gemm
from relu_splitter.anywhere.conv import Rsplitter_Conv


class ReluSplitter_Anywhere(Rsplitter_Gemm, Rsplitter_Conv):
    supported_op_types = ["Gemm", "Conv"]
    ignore_split_nodes = True
    supress_warnings = False

    def __init__(self, network, spec, input_shape=None, logger=default_logger):
        # gemm_node: the node to split (IT CAN BE ANY GEMM NODE IN THE MODEL)
        # model: the model containing the gemm_node
        self.onnx_path = network
        self.spec_path = spec
        self.logger = logger

        self.logger.debug("Initializing Rsplitter_Gemm...")
        self.logger.debug(f"onnx: {self.onnx_path}, spec: {self.spec_path}, input_shape: {input_shape}")
        self.init_model()
        self.init_vnnlib()


    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.model.num_inputs
        # reshape input bounds to match the model input shape
        input_lb, input_ub = input_lb.view(self.input_shape), input_ub.view(self.input_shape)
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"
        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))

        
    def init_model(self):
        self.model = WarppedOnnxModel.load(self.onnx_path)
        self.input_shape = list(self.model.input_shapes.values())[0]
        assert len(self.model.input) == 1, f"Model has more than one input {model.graph.input}"
        assert len(self.model.output) == 1, f"Model has more than one output {model.graph.output}"

    def init_seeds(self, random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def get_splittable_nodes(self):
        splittable_nodes = []
        if not self.supress_warnings and not self.ignore_split_nodes:
            self.logger.warning("ignore_split_nodes is set to False, splitting nodes that are already split might work suboptimally.")
        for node in self.model.nodes:
            if self.ignore_split_nodes and TOOL_NAME in node.name:
                continue
            if node.op_type in self.supported_op_types:
                splittable_nodes.append(node)
        return splittable_nodes
    
    def resolve_node_idx(self, op_type, idx):
        assert op_type in self.supported_op_types, f"Unsupported op_type {op_type}"
        nodes = [node for node in self.get_splittable_nodes() if node.op_type == op_type]
        assert 0 <= idx < len(nodes), f"Index {idx} out of range for op_type {op_type} with {len(nodes)} splittable nodes"
        return nodes[idx]
    
    def check_equivChk_conf(self, conf):
        return True

    def split(self, op_type, idx, conf):
        assert op_type in self.supported_op_types, f"Unsupported op_type {op_type}"
        if "seed" in conf:
            self.init_seeds(conf["seed"])

        node = self.resolve_node_idx(op_type, idx)
        if op_type == "Gemm":
            split_model, baseline_model = self.gemm_split(node, conf)
        elif op_type == "Conv":
            split_model, baseline_model = self.conv_split(node, conf)
        else:
            raise NotImplementedError(f"Splitting for op_type {op_type} is not implemented")
        
        if conf.get("equiv_chk_conf", None) is not None:
            equiv_chk_conf = conf["equiv_chk_conf"]
            closeness_results = check_models_closeness(
                self.model,
                [split_model, baseline_model],
                self.input_shape,
                device=None,
                n=equiv_chk_conf.get("n", 10),
                atol=equiv_chk_conf.get("atol", 1e-6),
                rtol=equiv_chk_conf.get("rtol", 1e-6)
            )
            (split_close, split_diff), (baseline_close, baseline_diff) = closeness_results
            
            print(f"Split model closeness: {split_close}, worst diff: {split_diff}")
            print(f"Baseline model closeness: {baseline_close}, worst diff: {baseline_diff}")

        return split_model, baseline_model

