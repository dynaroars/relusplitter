from .common import *
from .fc_split import RSplitter_fc
from .conv_split import RSplitter_conv

class ReluSplitter(RSplitter_fc, RSplitter_conv):

    def __init__(self, network: Path, spec: Path, input_shape=None, logger=default_logger, conf=default_config) -> None:
        self.onnx_path = network
        self.spec_path = spec
        self._conf = conf
        self.logger = logger

        self.logger.debug("ReluSplitter initializing")
        self.logger.debug(f"Model: {self.onnx_path}")
        self.logger.debug(f"Spec: {self.spec_path}")
        self.logger.debug(f"Config: {self._conf}")

        # self.check_config()
        self.init_model()
        self.init_vnnlib()
        self.init_seeds()

        if input_shape is not None:
            self.input_shape = input_shape

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
        if self.input_shape is not None:
            input_lb, input_ub = input_lb.view(self.input_shape), input_ub.view(self.input_shape)
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"
        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))

    def init_model(self):
        model = onnx.load(self.onnx_path)
        self.warpped_model = WarppedOnnxModel(model)
        self.input_shape = list(self.warpped_model.input_shapes.values())[0]
        assert len(model.graph.input) == 1, f"Model has more than one input {model.graph.input}"
        assert len(model.graph.output) == 1, f"Model has more than one output {model.graph.output}"

    def init_seeds(self):
        random_seed = self._conf.get("random_seed")
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    def split(self, idx, conf):
        splittable_nodes = self.get_splittable_nodes()
        assert idx < len(splittable_nodes), f"Split location <{idx}> is out of bound"
        n1, n2 = splittable_nodes[idx][0], splittable_nodes[idx][1]
        assert n2.op_type == "Relu", f"Node at split location is not a ReLU node"
        if n1.op_type == "Gemm":
            return self.split_fc(
                (n1, n2),
                n_splits=conf["n_splits"],
                split_mask=conf["split_mask"],
                fc_strategy=conf["fc_strategy"],
                scale_factors=conf["scale_factor"],
                create_baseline=conf["create_baseline"]
                )
        elif n1.op_type == "Conv":
            return self.split_conv(
                (n1, n2), 
                n_splits=conf["n_splits"], 
                split_mask=conf["split_mask"], 
                conv_strategy=conf["conv_strategy"], 
                scale_factors=conf["scale_factor"],
                create_baseline=conf["create_baseline"]
                )


    def get_splittable_nodes(self, model=None):
        # parrtern 1: Gemm -> Relu
        # parrtern 2: Conv2d -> Relu
        if model is None:
            model = self.warpped_model
        splittable_nodes = []
        for node in model.nodes:
            # skip the nodes generated by the RSplitter
            if TOOL_NAME in node.name:
                continue
            if node.op_type == "Relu":
                prior_node = model.get_prior_node(node)
                # pattern 1
                if prior_node.op_type == "Gemm":
                    splittable_nodes.append( (prior_node, node) )
                # pattern 2
                elif prior_node.op_type == "Conv":
                    splittable_nodes.append( (prior_node, node) )
                else:
                    logger.debug(f"Relu found after non-splittable node: {prior_node.op_type}")
                    pass
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
                    f"{prior_node.op_type} node: {prior_node.name}\n"
                    f"{relu_node.op_type} node: {relu_node.name}\n"
                    "=====================================")
        return splittable_nodes
    
    
    @classmethod
    def info(cls, onnx_path, spec_path, input_shape):
        print("Analyzing model...\n")
        print(f"Model: {onnx_path}\n")
        print(f"Spec: {spec_path}\n")
        print("=====================================")
        relu_splitter = cls(onnx_path, spec_path, logger=default_logger, conf=default_config, input_shape=input_shape)
        splittable_nodes = relu_splitter.get_splittable_nodes()
        print(f"Found {len(splittable_nodes)} splittable nodes")
        print("=====================================")
        for i, (prior_node, relu_node) in enumerate(splittable_nodes):
            print(  f">>> splittable ReLU layer {i} <<<\n"
                    f"{prior_node.op_type} node: {prior_node.name}\n"
                    f"{relu_node.op_type} node: {relu_node.name}\n"
                    f"======== Neuron composition ========\n")

            if prior_node.op_type == "Gemm":
                # use ibp or it might OOM for some model
                bounds = relu_splitter.warpped_model.get_bound_of(relu_splitter.bounded_input, prior_node.output[0], method="ibp")
                masks = relu_splitter.fc_get_split_masks(bounds)
                # masks = relu_splitter.fc_get_split_masks((prior_node, relu_node), method="backward")
                print(  f"stable+: {torch.sum(masks['stable+'])}\n"
                        f"stable-: {torch.sum(masks['stable-'])}\n"
                        f"unstable: {torch.sum(masks['unstable'])}\n"
                        f"Total: {torch.sum(masks['all'])}\n")
            elif prior_node.op_type == "Conv":
                bounds = relu_splitter.warpped_model.get_bound_of(relu_splitter.bounded_input, prior_node.output[0], method="ibp")
                masks = relu_splitter.conv_get_split_masks(bounds)
                for i in range(len(masks)):
                    counts = {k : torch.sum(v).item() for k, v in masks[i].items()}
                    print(f"kernel {i}: {counts}")
            else:
                self.logger.debug(f"None splittable node: {prior_node.op_type} in info...")
            print("=====================================")
            

    @classmethod
    def info_s(cls, onnx_path, spec_path):
        relu_splitter = cls(onnx_path, spec_path, logger=default_logger, conf=default_config)
        splittable_nodes = relu_splitter.get_splittable_nodes()
        node_info = []
        for i, (prior_node, relu_node) in enumerate(splittable_nodes):
            bounds = relu_splitter.warpped_model.get_bound_of(relu_splitter.bounded_input, prior_node.output[0], method="ibp")
            masks = relu_splitter.fc_get_split_masks(bounds)
            counts = {k: torch.sum(v).item() for k, v in masks.items()}
            node_info.append(counts)
        return node_info



