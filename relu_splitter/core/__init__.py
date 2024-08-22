import random
import logging
import os
from pathlib import Path
from typing import Union
from functools import reduce

from tqdm import tqdm
import onnx
import torch

from ..utils.misc import adjust_mask_random_k, adjust_mask_first_k, find_feasible_point
from ..utils.read_vnnlib import read_vnnlib
from ..utils.onnx_utils import check_model_closeness, check_model_closeness_gpu
from ..utils.errors import NOT_ENOUGH_NEURON, INVALID_PARAMETER, MODEL_NOT_EQUIV
from ..model import WarppedOnnxModel
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from .default_config import default_config

TOOL_NAME = os.environ.get("TOOL_NAME", "ReluSplitter")
default_logger = logging.getLogger(__name__)

class ReluSplitter():
    def __init__(self, network: Path, spec: Path, logger=default_logger, conf=default_config) -> None:
        self.onnx_path = network
        self.spec_path = spec
        self._conf = conf
        self.logger = logger

        self.logger.debug("ReluSplitter initializing")
        self.logger.debug(f"Model: {self.onnx_path}")
        self.logger.debug(f"Spec: {self.spec_path}")
        self.logger.debug(f"Config: {self._conf}")

        self.check_config()
        self.init_model()
        self.init_vnnlib()
        self.init_seeds()

        self.logger.debug("ReluSplitter initialized")


    def check_config(self):
        assert self.onnx_path.exists(), f"Model file <{self.onnx_path}> does not exist"
        assert self.spec_path.exists(), f"Spec file <{self.spec_path}> does not exist"
        params = ["min_splits", "max_splits", "random_seed", "split_strategy", "split_mask", "atol", "rtol"]
        missing_params = [param for param in params if param not in self._conf]
        assert len(missing_params) == 0, f"Missing parameters in config: {missing_params}"
        assert 0 < self._conf["min_splits"] <= self._conf["max_splits"]
        assert self._conf["atol"] >= 0
        assert self._conf["rtol"] >= 0
        assert self._conf["random_seed"] >= 0
        assert self._conf["split_strategy"] in ["single", "random", "reluS+", "reluS-", "adaptive"], f"Unknown split strategy {self._conf['split_strategy']}"
        assert self._conf["split_mask"] in ["stable+", "stable-", "stable", "unstable", "all", "unstable_n_stable+"], f"Unknown split mask {self._conf['split_mask']}"
        invalid_combinations = [("reluS+", "stable-"), ("reluS-", "stable+"), ("reluS+", "stable"), ("reluS-", "stable"), ("reluS+", "all"), ("reluS-", "all"),
                                ("reluS-", "unstable_n_stable+")]
        # assert (self._conf["split_strategy"], self._conf["split_mask"]) not in invalid_combinations, f"Invalid combination of split strategy and mask"
        assert self._conf["device"] in ["cpu", "cuda"], f"Invalid device {self._conf['device']}"
        if self._conf["device"] == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"

    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.warpped_model.num_inputs
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"
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
        lb, ub = lb.squeeze(), ub.squeeze()
        if split_strategy == "single":
            split_loc = (torch.rand_like(lb) * (ub - lb) + lb)
            return lambda **kwargs: split_loc
        elif split_strategy == "random":
            return lambda **kwargs: (torch.rand_like(lb) * (ub - lb) + lb)
        elif split_strategy == "reluS+":
            def split_loc_fn(**kwargs):
                w, b = kwargs["w"], kwargs["b"]
                return find_feasible_point(lb, ub, w, b)
            return split_loc_fn
        elif split_strategy == "reluS-":
            def split_loc_fn(**kwargs):
                w, b = kwargs["w"], kwargs["b"]
                return find_feasible_point(lb, ub, -w, -b)
            return split_loc_fn
        else:
            raise INVALID_PARAMETER(f"Unknown split strategy {split_strategy}")


    def get_split_masks(self, nodes):
        node1, node2 = nodes
        assert node1.op_type == "Gemm" and node2.op_type == "Relu", f"Invalid nodes:{node1.op_type} -> {node2.op_type}"
        input_lb, input_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.input[0])          # the input bound of the Gemm node, from which the split location is sampled
        output_lb, output_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.output[0])       # the output bound for determining the stability of the neurons
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        masks = {}
        masks["all"] = torch.ones_like(output_lb, dtype=torch.bool).squeeze()
        masks["stable"] = torch.logical_or(output_lb > 0, output_ub <= 0).squeeze()
        masks["unstable"] = ~masks["stable"]
        masks["stable+"] = (output_lb > 0).squeeze()
        masks["stable-"] = (output_ub <= 0).squeeze()
        masks["unstable_n_stable+"] = torch.logical_or(masks["unstable"], masks["stable+"])
        assert torch.all(masks["stable"] == (masks["stable+"] | masks["stable-"])), "stable is not equal to stable+ AND stable-"
        assert not torch.any(masks["stable+"] & masks["stable-"]), "stable+ and stable- are not mutually exclusive"
        assert not torch.any(masks["unstable"] & masks["stable"]), "unstable and stable are not mutually exclusive"
        assert torch.all((masks["unstable"] | masks["stable"]) == masks["all"]), "The union of unstable and stable does not cover all elements"
        return masks


    def get_splittable_nodes(self, model=None):
        # parrtern 1: Gemm -> Relu
        if model is None:
            model = self.warpped_model
        splittable_nodes = []
        for node in model.nodes:
            if TOOL_NAME in node.name:
                continue
            if node.op_type == "Relu":
                prior_node = model.get_prior_node(node)
                if prior_node.op_type == "Gemm":
                    splittable_nodes.append( (prior_node, node) )
        return splittable_nodes
    

    def split(self, split_idx=0):
        splittable_nodes = self.get_splittable_nodes()
        assert split_idx < len(splittable_nodes), f"Split location <{split_idx}> is out of bound"
        gemm_node, relu_node = splittable_nodes[split_idx][0], splittable_nodes[split_idx][1]
        assert all(attri.i == 0 for attri in gemm_node.attribute if attri.name == "TransA"), "TransA == 1 is not supported yet"

        self.logger.debug("=====================================")
        self.logger.debug(f"Splitting model: {self.onnx_path} with spec: {self.spec_path}")
        self.logger.debug(f"Splitting at Gemm node: <{gemm_node.name}> && ReLU node: <{relu_node.name}>")
        self.logger.debug(f"Random seed: {self._conf['random_seed']}")
        self.logger.debug(f"Split strategy: {self._conf['split_strategy']}")
        self.logger.debug(f"Split mask: {self._conf['split_mask']}")
        self.logger.debug(f"min_splits: {self._conf['min_splits']}, max_splits: {self._conf['max_splits']}")

        split_masks = self.get_split_masks(splittable_nodes[split_idx])
        split_mask = split_masks[self._conf["split_mask"]]
        mask_size  = torch.sum(split_mask).item()
        min_splits, max_splits = self._conf["min_splits"], self._conf["max_splits"]
        
        self.logger.info(f"============= Split Mask Sumamry =============")
        self.logger.info(f"stable+: {torch.sum(split_masks['stable+'])}\t"
                            f"stable-: {torch.sum(split_masks['stable-'])}")
        self.logger.info(f"unstable: {torch.sum(split_masks['unstable'])}\t"
                            f"all: {torch.sum(split_masks['all'])}")
        

        if mask_size < min_splits:
            self.logger.error(f"Not enough ReLUs to split, found {mask_size} ReLUs, but min_splits is {min_splits}")
            raise NOT_ENOUGH_NEURON("CANNOT-SPLITE: Not enough ReLUs to split")
        elif mask_size > max_splits:
            self.logger.info(f"Randomly selecting {max_splits}/{mask_size} ReLUs to split")
            # split_mask = adjust_mask_random_k(split_mask, max_splits)
            split_mask = adjust_mask_first_k(split_mask, max_splits)
        n_splits = torch.sum(split_mask).item()  # actual number of splits
        assert min_splits <= n_splits <= max_splits, f"Number of splits {n_splits} is out of range [{min_splits}, {max_splits}]"
        self.logger.info(f"Splitting {n_splits} {self._conf['split_mask']} ReLUs")
        
        split_loc_fn = self.get_split_loc_fn(splittable_nodes[split_idx])   # kwargs here
        grad, offset = self.warpped_model.get_gemm_wb(gemm_node)    # w,b of the layer to be splitted
        # create new layers
        num_out, num_in = grad.shape
        split_weights = torch.zeros((num_out + n_splits, num_in))
        split_bias = torch.zeros(num_out + n_splits)
        merge_weights = torch.zeros((num_out, num_out + n_splits))
        merge_bias = torch.zeros(num_out)
        
        idx = 0     # index of neuron in the new split layer
        for i in tqdm(range(len(split_mask)), desc="Constructing new layers"):
            if split_mask[i]:
                split_loc = split_loc_fn(w=grad[i], b=offset[i])
                w = grad[i]
                b = grad[i] @ split_loc
                
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
        self.logger.debug(f"Checking model closeness with atol: {self._conf['atol']} and rtol: {self._conf['rtol']}")
        input_shape = list(self.warpped_model.input_shapes.values())[0]
        if self._conf["device"] == "cpu":
            equiv, diff = check_model_closeness(self.warpped_model, 
                                                new_model, 
                                                input_shape, 
                                                atol=self._conf["atol"], 
                                                rtol=self._conf["rtol"])
        else:
            equiv, diff = check_model_closeness_gpu(self.warpped_model, 
                                                new_model, 
                                                input_shape, 
                                                atol=self._conf["atol"], 
                                                rtol=self._conf["rtol"])
        if not equiv:
            self.logger.error(f"Model closeness check failed with diff {diff}")
            raise MODEL_NOT_EQUIV("SPLITE-ERROR: Model closeness check failed")
        else:
            self.logger.info(f"Model closeness check passed with worst diff {diff}")
        self.logger.info("====== DONE ======")
        return new_model


    @classmethod
    def info_net_only(cls, onnx_path):
        class EmptyObject:
            pass
        print("Analyzing model...\n")
        print(f"Model: {onnx_path}\n")
        print("Analyzing model without a vnnlib file, include a vnnlib file to get more information.\n")
        print("=====================================")
        model = WarppedOnnxModel(onnx.load(onnx_path))
        model.info()
        fake_splitter = EmptyObject()
        fake_splitter.warpped_model = model
        splittable_nodes = cls.get_splittable_nodes(fake_splitter)
        print(f"Found {len(splittable_nodes)} splittable nodes")
        print("=====================================")

        for i, (prior_node, relu_node) in enumerate(splittable_nodes):
            print(  f">>> splittable ReLU layer {i} <<<\n"
                    f"Gemm node: {prior_node.name}\n"
                    f"ReLU node: {relu_node.name}\n"
                    "=====================================")
        return splittable_nodes
    
    
    @classmethod
    def info(cls, onnx_path, spec_path):
        print("Analyzing model...\n")
        print(f"Model: {onnx_path}\n")
        print(f"Spec: {spec_path}\n")
        print("=====================================")
        relu_splitter = cls(onnx_path, spec_path, logger=default_logger, conf=default_config)
        splittable_nodes = relu_splitter.get_splittable_nodes()
        print(f"Found {len(splittable_nodes)} splittable nodes")
        print("=====================================")
        node_info = []
        for i, (prior_node, relu_node) in enumerate(splittable_nodes):
            masks = relu_splitter.get_split_masks((prior_node, relu_node))
            print(  f">>> splittable ReLU layer {i} <<<\n"
                    f"Gemm node: {prior_node.name}\n"
                    f"ReLU node: {relu_node.name}\n"
                    f"======== Neuron composition ========\n"
                    f"stable+: {torch.sum(masks['stable+'])}\n"
                    f"stable-: {torch.sum(masks['stable-'])}\n"
                    f"unstable: {torch.sum(masks['unstable'])}\n"
                    f"Total: {torch.sum(masks['all'])}\n"
                    "=====================================")
            counts = {k: torch.sum(v).item() for k, v in masks.items()}
            node_info.append(counts)
        return node_info
            

    @classmethod
    def info_s(cls, onnx_path, spec_path):
        relu_splitter = cls(onnx_path, spec_path, logger=default_logger, conf=default_config)
        splittable_nodes = relu_splitter.get_splittable_nodes()
        node_info = []
        for i, (prior_node, relu_node) in enumerate(splittable_nodes):
            masks = relu_splitter.get_split_masks((prior_node, relu_node))
            counts = {k: torch.sum(v).item() for k, v in masks.items()}
            node_info.append(counts)
        return node_info

    @classmethod
    def get_baseline_split(cls, onnx_path, n_splits, split_idx=0, atol=1e-4, rtol=1e-4):
        device="cuda"if torch.cuda.is_available() else "cpu"

        model = WarppedOnnxModel(onnx.load(onnx_path))
        splittable_nodes = cls.get_splittable_nodes(None, model)
        assert split_idx < len(splittable_nodes), f"Split location <{split_idx}> is out of bound"
        gemm_node, relu_node = splittable_nodes[split_idx][0], splittable_nodes[split_idx][1]
        grad, offset = model.get_gemm_wb(gemm_node)
        # randomly select n_splits neurons to split
        split_mask = torch.ones(grad.shape[0], dtype=torch.bool)
        split_mask = adjust_mask_random_k(split_mask, n_splits)

        grad, offset = model.get_gemm_wb(gemm_node)    # w,b of the layer to be splitted

        # create new layers
        num_out, num_in = grad.shape
        split_weights = torch.zeros((num_out + n_splits, num_in))
        split_bias = torch.zeros(num_out + n_splits)
        merge_weights = torch.zeros((num_out, num_out + n_splits))
        merge_bias = torch.zeros(num_out)
        
        idx = 0     # index of neuron in the new split layer
        for i in range(len(split_mask)):
            if split_mask[i]:
                w = grad[i]/2
                b = offset[i]/2
                
                split_weights[idx]      = w
                split_weights[idx+1]    = w
                split_bias[idx]         = b
                split_bias[idx+1]       = b
                merge_weights[i][idx]   = 1
                merge_weights[i][idx+1] = 1
                merge_bias[i] = 0
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
        split_pre_tensor_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPre")
        split_post_tensor_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPost")
        merge_pre_tensor_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_mPre")
        merge_pose_tensor_name = orginal_output_name
        split_w_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_sW")
        split_b_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_sB")
        merge_w_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_mW")
        merge_b_name = model.gen_tensor_name(prefix=f"{TOOL_NAME}_mB")
        split_node_name      = model.gen_node_name(prefix=f"{TOOL_NAME}_s")
        split_relu_node_name = model.gen_node_name(prefix=f"{TOOL_NAME}_sR")
        merge_node_name      = model.gen_node_name(prefix=f"{TOOL_NAME}_m")
        merge_relu_node_name = model.gen_node_name(prefix=f"{TOOL_NAME}_mR")

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

        new_model = model.generate_updated_model(  nodes_to_replace=[gemm_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=f"{onnx_path.stem}_split_{split_idx}",
                                                                producer_name="ReluSplitter")
        # self.logger.info("=========== Model created ===========")
        # self.logger.debug(f"Checking model closeness with atol: {self._conf['atol']} and rtol: {self._conf['rtol']}")
        input_shape = list(model.input_shapes.values())[0]
        if device == "cpu":
            equiv, diff = check_model_closeness(model, 
                                                new_model, 
                                                input_shape, 
                                                atol=atol, 
                                                rtol=rtol)
        else:
            equiv, diff = check_model_closeness_gpu(model, 
                                                new_model, 
                                                input_shape, 
                                                atol=atol, 
                                                rtol=rtol)
        if not equiv:
            # self.logger.error(f"Model closeness check failed with diff {diff}")
            raise MODEL_NOT_EQUIV("SPLITE-ERROR: Model closeness check failed")
        else:
            pass
            # self.logger.info(f"Model closeness check passed with worst diff {diff}")
        # self.logger.info("====== DONE ======")
        return new_model