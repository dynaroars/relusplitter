import random
import logging
import os
from pathlib import Path
from typing import Union
from functools import reduce

import onnx
import torch

from ..utils.read_vnnlib import read_vnnlib
from ..utils.onnx_utils import truncate_onnx_model, check_model_closeness
from ..model import WarppedOnnxModel
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from .default_config import default_conf

TOOL_NAME = os.environ.get("TOOL_NAME", "ReluSplitter")
default_logger = logging.getLogger(__name__)

class ReluSplitter():
    def __init__(self, network: Union[Path], spec: Union[Path, str, None], logger = default_logger, conf=default_conf) -> None:
        self.onnx_path = network
        self.spec_path = spec
        self._conf = conf
        self.logger = logger
        self.check_config()
        self.init_model()
        self.init_vnnlib()
        self.init_seeds()
        self.logger.info("ReluSplitter initialized")
        self.logger.info(f"Model: {self.onnx_path}")
        self.logger.info(f"Spec: {self.spec_path}")
        self.logger.info(f"Config: {self._conf}")


    def check_config(self):
        assert self.onnx_path.exists(), f"Model file <{self.onnx_path}> does not exist"
        assert self.spec_path.exists(), f"Spec file <{self.spec_path}> does not exist"
        assert "max_splits" in self._conf, "max_splits not found in config"
        assert self._conf["split_strategy"] in ["single", "random", "adaptive"]
        assert self._conf["split_idx"] >= 0
        assert self._conf["random_seed"] >= 0
        assert self._conf["split_mask"] in ["stable", "unstable", "all"]
        assert self._conf["atol"] >= 0
        assert self._conf["rtol"] >= 0

    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.warpped_model.num_inputs

        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, "Spec number of inputs does not match model inputs"
        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))

    def init_model(self):
        model = onnx.load(self.onnx_path)
        assert len(model.graph.input) == 1, f"Model has more than one input {model.graph.input}"
        assert len(model.graph.output) == 1, f"Model has more than one output {model.graph.output}"
        self.warpped_model = WarppedOnnxModel(model)

    def init_seeds(self):
        random_seed = self._conf.get("random_seed")
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    def get_split_loc_fn(self, nodes, **kwargs):
        gemm_node, relu_node = nodes
        split_strategy = self._conf["split_strategy"]
        lb, ub = self.warpped_model.get_bound_of(self.bounded_input, gemm_node.input[0])          # the input bound of the Gemm node, from which the split location is sampled
        if split_strategy == "single":
            split_loc = (torch.rand_like(lb) * (ub - lb) + lb).squeeze()
            return lambda: split_loc
        elif split_strategy == "random":
            return lambda: (torch.rand_like(lb) * (ub - lb) + lb).squeeze()
        elif split_strategy == "adaptive":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown split strategy {split_strategy}")

    def get_split_mask(self, nodes, reversed=False):
        mask = self._conf["split_mask"]
        max_splits = self._conf["max_splits"]
        node1, node2 = nodes
        assert node1.op_type == "Gemm" and node2.op_type == "Relu", f"Invalid nodes:{node1.op_type} -> {node2.op_type}"
        if mask == "all":
            split_mask = torch.ones(node1.output[0].type.shape[1], dtype=torch.bool)
        elif mask in ["stable", "unstable"]:
        # the input and output of the Gemm node
            input_lb, input_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.input[0])          # the input bound of the Gemm node, from which the split location is sampled
            output_lb, output_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.output[0])       # the output bound for determining the stability of the neurons
            assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
            split_mask = torch.logical_or(output_lb >= 0, output_ub < 0).squeeze()
            if mask == "unstable":
                split_mask = ~split_mask
        else:
            raise NotImplementedError(f"Unknown split mask {mask}")
        self.logger.info(f"Found {torch.sum(split_mask).item()} splitable ReLUs with mask <{mask}>")

        n_splits = torch.sum(split_mask).item()
        if n_splits > max_splits:
            self.logger.info(f"Max split set to: {max_splits}...\n Randomly selecting {max_splits} to split")
            indices = torch.nonzero(split_mask, as_tuple=True)[0]
            selected_indices = indices[torch.randperm(len(indices))[:max_splits]]
            split_mask = torch.zeros_like(split_mask, dtype=torch.bool)
            split_mask[selected_indices] = True
            n_splits = max_splits
            assert torch.sum(split_mask).item() <= max_splits, "Number of True values exceeds max_splits"
            assert torch.equal(split_mask.logical_or(split_mask), split_mask), "relu max property violated"
        return split_mask, n_splits

    def get_splitable_nodes(self):
        # parrtern 1: Gemm -> Relu
        splitable_nodes = []
        for node in self.warpped_model.nodes:
            if TOOL_NAME in node.name:
                continue
            if node.op_type == "Relu":
                prior_node = self.warpped_model.get_prior_node(node)
                if prior_node.op_type == "Gemm":
                    splitable_nodes.append( (prior_node, node) )
        return splitable_nodes
    
    def split(self, split_idx=0, max_splits=None):
        splitable_nodes = self.get_splitable_nodes()
        assert split_idx < len(splitable_nodes), f"Split location <{split_idx}> is out of bound"
        gemm_node, relu_node = splitable_nodes[split_idx][0], splitable_nodes[split_idx][1]
        assert all(attri.i == 0 for attri in gemm_node.attribute if attri.name == "TransA"), "TransA == 1 is not supported yet"

        self.logger.info("=====================================")
        self.logger.info(f"Splitting model: {self.onnx_path}")
        self.logger.info(f"Spec file: {self.spec_path}")
        self.logger.info(f"Random seed: {self._conf['random_seed']}")
        self.logger.info(f"Split strategy: {self._conf['split_strategy']}")
        self.logger.info(f"Max splits: {self._conf['max_splits']}")
        self.logger.info(f"Splitting at Gemm node: <{gemm_node.name}> && ReLU node: <{relu_node.name}>")

        split_mask, n_splits = self.get_split_mask(splitable_nodes[split_idx])
        if n_splits == 0:
            self.logger.info("No ReLUs to split, splitting the next layer instead...")
            return self.split(split_idx+1, max_splits)

        split_loc_fn = self.get_split_loc_fn(splitable_nodes[split_idx])   # kwargs here
        grad, offset = self.warpped_model.get_gemm_wb(gemm_node)

        # create new layers
        num_out, num_in = grad.shape
        split_weights = torch.zeros((num_out + n_splits, num_in))
        split_bias = torch.zeros(num_out + n_splits)
        merge_weights = torch.zeros((num_out, num_out + n_splits))
        merge_bias = torch.zeros(num_out)
        
        idx = 0
        for i in range(len(split_mask)):
            w = grad[i]
            b = grad[i] @ split_loc_fn()
            if split_mask[i]:
                split_weights[idx]      = w
                split_weights[idx+1]    = -w
                split_bias[idx]         = -b
                split_bias[idx+1]       = b
                merge_weights[i][idx]   = 1
                merge_weights[i][idx+1] = -1
                merge_bias[i] = offset[i] + b
                idx += 2
            else:
                split_weights[idx] = grad[i]
                split_bias[idx] = offset[i]
                merge_weights[i][idx] = 1
                idx += 1
        split_weights = split_weights.t()
        merge_weights = merge_weights.t()
        
        orginal_input_name = gemm_node.input[0]
        orginal_output_name = relu_node.output[0]
        split_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPre")
        split_post_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPost")
        merge_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mPre")
        merge_pose_tensor_name = orginal_output_name
        split_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sW")
        split_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sB")
        merge_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mW")
        merge_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mB")
        split_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_s")
        split_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_sR")
        merge_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_m")
        merge_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_mR")

        # create the new split layer
        split_layer = onnx.helper.make_node("Gemm",
                                            inputs=[orginal_input_name, split_w_name, split_b_name],
                                            outputs=[split_pre_tensor_name],
                                            name=split_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        split_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[split_pre_tensor_name],
                                                outputs=[split_post_tensor_name],
                                                name=split_relu_node_name)
        # create the new merge layer
        merge_layer = onnx.helper.make_node("Gemm",
                                            inputs=[split_post_tensor_name, merge_w_name, merge_b_name],
                                            outputs=[merge_pre_tensor_name],
                                            name=merge_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        merge_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[merge_pre_tensor_name],
                                                outputs=[merge_pose_tensor_name],
                                                name=merge_relu_node_name)
        
        new_nodes = [split_layer, split_layer_relu, merge_layer, merge_layer_relu]
        new_initializers = [
            onnx.helper.make_tensor(split_w_name, onnx.TensorProto.FLOAT, split_weights.shape, split_weights.flatten().tolist()),
            onnx.helper.make_tensor(split_b_name, onnx.TensorProto.FLOAT, split_bias.shape, split_bias.flatten().tolist()),
            onnx.helper.make_tensor(merge_w_name, onnx.TensorProto.FLOAT, merge_weights.shape, merge_weights.flatten().tolist()),
            onnx.helper.make_tensor(merge_b_name, onnx.TensorProto.FLOAT, merge_bias.shape, merge_bias.flatten().tolist())
        ]

        new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[gemm_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=f"{self.onnx_path.stem}_split_{split_idx}",
                                                                producer_name="ReluSplitter")
        self.logger.info("=========== Model created ===========")
        self.logger.info("=====================================")
        self.logger.info(f"Checking model closeness with atol: {self._conf['atol']} and rtol: {self._conf['rtol']}")
        input_shape = list(self.warpped_model.input_shapes.values())[0]
        equiv, diff = check_model_closeness(self.warpped_model, 
                                            new_model, 
                                            input_shape, 
                                            atol=self._conf["atol"], 
                                            rtol=self._conf["rtol"])
        if not equiv:
            self.logger.error(f"Model closeness check failed, with diff {diff}")
            raise ValueError("Model closeness check failed")
        else:
            self.logger.info(f"Model closeness check passed with worst diff {diff}")
        return new_model


    @classmethod
    def info_net_only(cls, onnx_path):
        class EmptyObject:
            pass

        logger = logging.getLogger(__name__)
        logger.info("Analyzing model")
        logger.info(f"Model: {onnx_path}")

        model = WarppedOnnxModel(onnx.load(onnx_path))
        model.info()

        fake_splitter = EmptyObject()
        fake_splitter.warpped_model = model
        splitable_nodes = cls.get_splitable_nodes(fake_splitter)

        logger.info(f"Found {len(splitable_nodes)} splitable nodes")
        for i, (prior_node, relu_node) in enumerate(splitable_nodes):
            logger.info("\n"
                        f">>> Splitable ReLU layer {i} <<<\n"
                        f"Gemm node: {prior_node.name}\n"
                        f"ReLU node: {relu_node.name}\n"
                        "=====================================")
        return splitable_nodes