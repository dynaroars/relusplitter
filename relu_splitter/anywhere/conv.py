import random

import torch
import onnx
from tqdm import tqdm


from relu_splitter import TOOL_NAME

class Rsplitter_Conv():
    def conv_split(self, conv_node, conf):
        split_activation = conf["split_activation"].lower()

        assert conv_node.op_type == "Conv", f"Node to split is not a Conv node: {conv_node.op_type}"
        assert split_activation in ["relu", "leakyrelu", "prelu"], f"Unsupported split activation: {split_activation}"

        split_model, baseline_model = self._conv_split(conv_node, conf)

        return split_model, baseline_model
    

    #----------------------- Conv -----------------------#
    #----------------------- Conv -----------------------#
    def _conv_split(self, conv_node, conf):
        bounding_method_tight = conf.get("bounding_method_tight", "backward")
        bounding_method_loose = conf.get("bounding_method_loose", "ibp")
        create_baseline = conf.get("create_baseline", False)

        split_activation = conf.get("split_activation", "relu").lower()
        additional_activation_params = conf.get("additional_activation_params", {})

        param_selection_conf = conf.get("param_conf", {})

        # Tight bounds shape: torch.Size([1, 8, 16, 16])
        tight_bounds = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method=bounding_method_tight)
        loose_bounds = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method=bounding_method_loose)
        tight_bounds = (tight_bounds[0].squeeze(0), tight_bounds[1].squeeze(0))  # remove batch dim
        loose_bounds = (loose_bounds[0].squeeze(0), loose_bounds[1].squeeze(0))  # remove batch dim

        split_idxs = self._decide_split_idxs_conv(tight_bounds, loose_bounds, conf)
        split_dict = self._decide_split_params_conv(tight_bounds, loose_bounds, split_idxs, param_selection_conf)
        activation_params = {
            "leakyrelu_alpha": self._decide_leakyrelu_alpha_conv(additional_activation_params),
            "prelu_slope": self._decide_prelu_slope_conv(additional_activation_params),
        }
        new_model = self._split_general_conv(conv_node ,split_dict, split_activation, activation_params)

        baseline_model = None
        if create_baseline:
            baseline_idxs = []
            baseline_dict = self._decide_split_params_conv(tight_bounds, loose_bounds, baseline_idxs, param_selection_conf)
            baseline_model = self._split_general_conv(conv_node ,baseline_dict, split_activation, activation_params)

        return new_model, baseline_model
    
    def _decide_split_idxs_conv(self, tight_bounds, loose_bounds, conf):
        sorting_strat = conf.get("sorting_strat", "random")
        n_splits = conf.get("n_splits", 0)
        seed = conf.get("seed", None)
        if seed is not None: # create a seperate random generator to avoid messing up global random state
            rnd = random.Random(seed)
        else:
            rnd = random

        layer_width = tight_bounds[0].shape[0]  # number of kernels
        idxs = list(range(layer_width))

        assert 0 <= n_splits <= layer_width, f"n_splits {n_splits} must be between 0 and layer width {layer_width}, got {n_splits}"
        if sorting_strat == "random":
            rnd.shuffle(idxs)
        else:
            raise NotImplementedError(f"sorting_strat {sorting_strat} is not implemented yet")
        
        split_idxs = idxs[:n_splits]
        self.logger.info(f"Split idxs: {split_idxs}")
        return split_idxs

    def _decide_split_params_conv(self, tight_bounds, loose_bounds, split_idxs, param_selection_conf):
        conv_tau_strat = param_selection_conf.get("conv_tau_strat", "naive_sliding_window")
        conv_scale_strat = param_selection_conf.get("conv_scale_strat", "fixed")
        if conv_scale_strat == "fixed":
            fixed_scale = param_selection_conf.get("conv_fixed_scale", (1.0, -1.0))
        if conv_scale_strat == "random":
            s_pos_range = param_selection_conf.get("conv_random_scale_range", (0.1, 100.0))
            s_neg_range = param_selection_conf.get("conv_random_scale_range", (-0.1, -100.0))

        stable_tau_strat = param_selection_conf.get("stable_tau_strat", "random").lower() # random, BigTau, SmallTau
        stable_tau_margin = param_selection_conf.get("stable_tau_margin", (10.0, 50.0))
        stable_scale_strat = param_selection_conf.get("stable_scale_strat", "fixed").lower()
        if stable_scale_strat == "fixed":
            stable_fixed_scales = param_selection_conf.get("stable_fixed_scales", (1.0, -1.0))
        if stable_scale_strat == "random":
            stable_s_pos_range = param_selection_conf.get("stable_s_pos_range", (0.1, 100.0))
            stable_s_neg_range = param_selection_conf.get("stable_s_neg_range", (-100.0, -0.1))

        n_kernels = tight_bounds[0].shape[0]

        split_dict = {}
        for idx in range(n_kernels):
            tight_kernel_lb, tight_kernel_ub = tight_bounds[0][idx], tight_bounds[1][idx]
            loose_kernel_lb, loose_kernel_ub = loose_bounds[0][idx], loose_bounds[1][idx]
            if idx in split_idxs:   # require split
                # taus
                if conv_tau_strat == "naive_sliding_window":
                    # find the val to sat most intervals using sliding line sweep
                    evnts = []
                    for i in range(tight_kernel_lb.numel()):
                        evnts.append( (tight_kernel_lb.view(-1)[i].item(), 1) )
                        evnts.append( (tight_kernel_ub.view(-1)[i].item(), -1))
                    evnts.sort(key=lambda x: x[0])
                    active_intervals = 0
                    max_active_intervals = -1
                    start, end = None, None
                    curr_start = evnts[0][0]
                    for val, evnt in evnts:
                        active_intervals += evnt
                        if active_intervals > max_active_intervals:
                            max_active_intervals = active_intervals
                            start = curr_start
                            end = val
                        curr_start = val
                    tau = random.uniform(start, end)
                else:
                    raise NotImplementedError(f"conv_tau_strat {conv_tau_strat} is not implemented yet")
                # scales
                if conv_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scale
                elif conv_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"conv_scale_strat {conv_scale_strat} is not implemented yet")
            else:   # no split
                tau = 10.0  # default tau for no split
                s_pos, s_neg = 1.0, -1.0
            split_dict[idx] = (tau, (s_pos, s_neg))
        return split_dict


    
    def _decide_leakyrelu_alpha_conv(self, additional_activation_conf):
        return additional_activation_conf.get("leakyrelu_alpha", 0.01)
    
    def _decide_prelu_slope_conv(self, additional_activation_conf):
        return additional_activation_conf.get("prelu_slope", 0.25)
    
    def _split_general_conv(self, conv_node, split_dict, split_activation, params):
        if split_activation == "relu":
            split_model = self._split_ReLU_conv(conv_node, split_dict)
        elif split_activation == "leakyrelu":
            split_model = self._split_LeakyReLU_conv(conv_node, split_dict, params["leakyrelu_alpha"])
        elif split_activation == "prelu":
            split_model = self._split_PReLU_conv(conv_node, split_dict, params["prelu_slope"])
        else:
            raise NotImplementedError(f"split_activation {split_activation} is not implemented yet")
        return split_model
    
    def _split_ReLU_conv(self, node_to_split, split_dict):


        ori_dilations = node_to_split.attribute[1].ints
        if ori_dilations == []:
            ori_dilations = [1,1]
        ori_kernel_shape = node_to_split.attribute[2].ints
        ori_pads = node_to_split.attribute[3].ints
        ori_strides = node_to_split.attribute[4].ints

        ori_w, ori_b = self.model.get_conv_wb(node_to_split)
        assert ori_w.dim() == 4, f"Conv weight dimension is not 4: {ori_w.dim()}"
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

        split_layer_kernel_count = 2 * ori_oC

        conv1_w = torch.zeros((split_layer_kernel_count, ori_iC, ori_kH, ori_kW))
        conv1_b = torch.zeros((split_layer_kernel_count,))
        conv2_w = torch.zeros((ori_oC, split_layer_kernel_count, 1, 1))
        conv2_b = torch.zeros((ori_oC,))

        free_kernels = [i for i in range(split_layer_kernel_count)]
        random.shuffle(free_kernels)

        for i in tqdm(range(ori_oC), desc="Creating split Conv-ReLU-Conv layers"):
            idx1, idx2 = free_kernels.pop(), free_kernels.pop()
            tau, (s_pos, s_neg) = split_dict[i]
            # conv_1
            conv1_w[idx1] = s_pos * ori_w[i]
            conv1_w[idx2] = s_neg * ori_w[i]
            conv1_b[idx1] = -s_pos * (tau-ori_b[i])
            conv1_b[idx2] = -s_neg * (tau-ori_b[i])
            # conv_2
            conv2_w[i, idx1, 0, 0] = 1.0 / s_pos
            conv2_w[i, idx2, 0, 0] = 1.0 / s_neg
            conv2_b[i] = tau

        input_tname = node_to_split.input[0]
        conv_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1")   # intermediate output after gemm_1
        relu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_ReLU")     # intermediate output after ReLU
        conv_2_output_tname  = node_to_split.output[0]                                        # final output

        conv_1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1w")
        conv_1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1b")
        conv_2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2w")
        conv_2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2b")

        conv_1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv1")
        relu_nname   = self.model.gen_node_name(prefix=f"{TOOL_NAME}_ReLU")
        conv_2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv2")

        conv_1_node = onnx.helper.make_node(
            "Conv",
            inputs=[input_tname, conv_1_w_tname, conv_1_b_tname],
            outputs=[conv_1_output_tname],
            name=conv_1_nname,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        relu_node = onnx.helper.make_node(
            "Relu",
            inputs=[conv_1_output_tname],
            outputs=[relu_output_tname],
            name=relu_nname
        )
        conv_2_node = onnx.helper.make_node(
            "Conv",
            inputs=[relu_output_tname, conv_2_w_tname, conv_2_b_tname],
            outputs=[conv_2_output_tname],
            name=conv_2_nname,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )

        new_nodes = [conv_1_node, relu_node, conv_2_node]
        new_initializers = [
            onnx.helper.make_tensor(conv_1_w_tname, onnx.TensorProto.FLOAT, conv1_w.shape, conv1_w.flatten().tolist()),
            onnx.helper.make_tensor(conv_1_b_tname, onnx.TensorProto.FLOAT, conv1_b.shape, conv1_b.flatten().tolist()),
            onnx.helper.make_tensor(conv_2_w_tname, onnx.TensorProto.FLOAT, conv2_w.shape, conv2_w.flatten().tolist()),
            onnx.helper.make_tensor(conv_2_b_tname, onnx.TensorProto.FLOAT, conv2_b.shape, conv2_b.flatten().tolist())
        ]
        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_ConvSplit",
            producer_name=TOOL_NAME
        )
        return new_model
            
