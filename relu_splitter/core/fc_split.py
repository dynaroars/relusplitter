from .common import *

class RSplitter_fc():

    def get_split_loc_fn_fc(self, nodes, **kwargs):
        # come back fix this later, Theres no need to pick from the input interval, just pick from the output interval
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

    # def fc_get_split_masks(self, nodes, method="backward"):
    #     node1, node2 = nodes
    #     assert node1.op_type == "Gemm" and node2.op_type == "Relu", f"Invalid nodes:{node1.op_type} -> {node2.op_type}"
    #     input_lb, input_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.input[0], method=method)          # the input bound of the Gemm node, from which the split location is sampled
    #     output_lb, output_ub = self.warpped_model.get_bound_of(self.bounded_input, node1.output[0], method=method)       # the output bound for determining the stability of the neurons
    #     assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
    #     masks = {}
    #     masks["all"] = torch.ones_like(output_lb, dtype=torch.bool).squeeze()
    #     masks["stable"] = torch.logical_or(output_lb > 0, output_ub <= 0).squeeze()
    #     masks["unstable"] = ~masks["stable"]
    #     masks["stable+"] = (output_lb > 0).squeeze()
    #     masks["stable-"] = (output_ub <= 0).squeeze()
    #     masks["unstable_n_stable+"] = torch.logical_or(masks["unstable"], masks["stable+"])
    #     assert torch.all(masks["stable"] == (masks["stable+"] | masks["stable-"])), "stable is not equal to stable+ AND stable-"
    #     assert not torch.any(masks["stable+"] & masks["stable-"]), "stable+ and stable- are not mutually exclusive"
    #     assert not torch.any(masks["unstable"] & masks["stable"]), "unstable and stable are not mutually exclusive"
    #     assert torch.all((masks["unstable"] | masks["stable"]) == masks["all"]), "The union of unstable and stable does not cover all elements"
    #     return masks

    def fc_get_split_masks(self, bounds):
        output_lb, output_ub = bounds
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

    def split_fc(self, nodes_to_split):
        gemm_node, relu_node = nodes_to_split
        assert all(attri.i == 0 for attri in gemm_node.attribute if attri.name == "TransA"), "TransA == 1 is not supported yet"

        self.logger.debug("=====================================")
        self.logger.debug(f"Splitting model: {self.onnx_path} with spec: {self.spec_path}")
        self.logger.debug(f"Splitting at Gemm node: <{gemm_node.name}> && ReLU node: <{relu_node.name}>")
        self.logger.debug(f"Random seed: {self._conf['random_seed']}")
        self.logger.debug(f"Split strategy: {self._conf['split_strategy']}")
        self.logger.debug(f"Split mask: {self._conf['split_mask']}")
        self.logger.debug(f"min_splits: {self._conf['min_splits']}, max_splits: {self._conf['max_splits']}")

        bounds = self.warpped_model.get_bound_of(self.bounded_input, gemm_node.output[0], method="backward")
        split_masks = self.fc_get_split_masks(bounds)
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
        
        split_loc_fn = self.get_split_loc_fn_fc( (nodes_to_split) )   # kwargs here
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
                scale_ratio_pos, scale_ratio_neg = self._conf["scale_factor"]
                # split_weights[idx]      = w
                # split_weights[idx+1]    = -w
                # split_bias[idx]         = -b
                # split_bias[idx+1]       = b

                split_weights[idx]      = scale_ratio_pos * w
                split_weights[idx+1]    = scale_ratio_neg * w
                split_bias[idx]         = -scale_ratio_pos * b
                split_bias[idx+1]       = -scale_ratio_neg * b

                merge_weights[i][idx]   = 1/scale_ratio_pos
                merge_weights[i][idx+1] = 1/scale_ratio_neg
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
                                                                graph_name=f"{self.onnx_path.stem}_split",
                                                                # graph_name=f"{self.onnx_path.stem}_split_{split_idx}",
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
    def get_baseline_split_fc(cls, onnx_path, n_splits, split_idx=0, atol=1e-4, rtol=1e-4):
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