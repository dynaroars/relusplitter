from .common import *
from .fc_split import RSplitter_fc
from .conv_split import RSplitter_conv

TOOL_NAME = os.environ.get("TOOL_NAME", "ReluSplitter")
default_logger = logging.getLogger(__name__)

class ReluSplitter(RSplitter_fc, RSplitter_conv):

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
        assert self._conf["scale_factor"][0] > 0 and self._conf["scale_factor"][1] < 0
        assert self._conf["atol"] >= 0
        assert self._conf["rtol"] >= 0
        assert self._conf["random_seed"] >= 0
        assert self._conf["split_strategy"] in ["single", "random", "reluS+", "reluS-", "adaptive"], f"Unknown split strategy {self._conf['split_strategy']}"
        assert self._conf["split_mask"] in ["stable+", "stable-", "stable", "unstable", "all", "unstable_n_stable+"], f"Unknown split mask {self._conf['split_mask']}"
        invalid_combinations = [("reluS+", "stable-"), ("reluS-", "stable+"), ("reluS+", "stable"), ("reluS-", "stable"), ("reluS+", "all"), ("reluS-", "all"),
                                ("reluS-", "unstable_n_stable+")]
        assert (self._conf["split_strategy"], self._conf["split_mask"]) not in invalid_combinations, f"Invalid combination of split strategy and mask"
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



