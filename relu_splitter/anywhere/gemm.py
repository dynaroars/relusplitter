import random

import torch
import onnx
from tqdm import tqdm


from relu_splitter import TOOL_NAME

class Rsplitter_Gemm():

    def gemm_split(self, gemm_node, conf):
        split_activation = conf["split_activation"].lower()

        assert gemm_node.op_type == "Gemm", f"Node to split is not a Gemm node: {gemm_node.op_type}"
        assert split_activation in ["relu", "leakyrelu", "prelu", "thresholdedrelu"], f"Unsupported split activation: {split_activation}"

        if split_activation == "relu":
            split_model, baseline_model = self.split_ReLU(gemm_node, conf)
        # elif split_activation == "leakyrelu":
        #     split_model, baseline_model = self.split_LeakyReLU(nodes, conf)
        elif split_activation == "prelu":
            split_model, baseline_model = self.split_PReLU(gemm_node, conf)
        # elif split_activation == "thresholdedrelu":
        #     split_model, baseline_model = self.split_ThresholdedReLU(nodes, conf)
        else:
            raise NotImplementedError(f"Splitting for activation {split_activation} is not implemented")
        return split_model, baseline_model

    # shared general implementations for gemm splitting
    def gemm_get_split_masks_general(self, bounds, cutoff=0):
        # cutoff is the kink point, for ReLU is 0, for threasholded this value can be different
        output_lb, output_ub = bounds
        masks = {}
        masks["all"] = torch.ones_like(output_lb, dtype=torch.bool).squeeze()
        masks["stable"] = torch.logical_or(output_lb > cutoff, output_ub <= cutoff).squeeze()
        masks["unstable"] = ~masks["stable"]
        masks["stable+"] = (output_lb > cutoff).squeeze()
        masks["stable-"] = (output_ub <= cutoff).squeeze()
        masks["unstable_n_stable+"] = torch.logical_or(masks["unstable"], masks["stable+"])
        assert torch.all(masks["stable"] == (masks["stable+"] | masks["stable-"])), "stable is not equal to stable+ AND stable-"
        assert not torch.any(masks["stable+"] & masks["stable-"]), "stable+ and stable- are not mutually exclusive"
        assert not torch.any(masks["unstable"] & masks["stable"]), "unstable and stable are not mutually exclusive"
        assert torch.all((masks["unstable"] | masks["stable"]) == masks["all"]), "The union of unstable and stable does not cover all elements"
        return masks


    #----------------------- ReLU -----------------------#
    #----------------------- ReLU -----------------------#
    def split_ReLU(self, nodes, conf):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY
        bounding_method_tight = conf.get("bounding_method_tight", "backward")
        bounding_method_loose = conf.get("bounding_method_loose", "ibp")
        create_baseline = conf.get("create_baseline", False)
        candidate_selection_conf = conf.get("candidate_selection_conf", {})
        param_selection_conf = conf.get("param_conf", {})

        tight_bounds = self.model.get_bound_of(self.bounded_input, nodes.output[0], method=bounding_method_tight)
        loose_bounds = self.model.get_bound_of(self.bounded_input, nodes.output[0], method=bounding_method_loose)

        split_idxs = self._decide_split_idxs_ReLU(tight_bounds, loose_bounds, candidate_selection_conf)
        split_dict = self._decide_split_params_ReLU(tight_bounds, loose_bounds, split_idxs, param_selection_conf)
        new_model = self._split_ReLU(nodes ,split_dict)

        baseline_model = None
        if create_baseline:
            raise NotImplementedError("yo")
            baseline_model  = self._create_baseline_ReLU(split_dict)

        return new_model, baseline_model
    
    # responsible for selecting neurons to split
    def _decide_split_idxs_ReLU(self, tight_bounds, loose_bounds, candidate_selection_conf):
        sorting_strat = candidate_selection_conf.get("sorting_strat", "random")
        n_splits = candidate_selection_conf.get("n_splits", 0)

        layer_width = tight_bounds[0].shape[1]
        idxs = list(range(layer_width))

        assert 0 <= n_splits <= layer_width, f"n_splits {n_splits} must be between 0 and layer width {layer_width}, got {n_splits}"
        if sorting_strat == "random":
            random.shuffle(idxs)
        else:
            raise NotImplementedError(f"sorting_strat {sorting_strat} is not implemented yet")

        return idxs[:n_splits]

    #     split_mask = candidate_selection_conf.get("split_mask", "stable").lower()
    #     n_splits = candidate_selection_conf.get("n_splits", None)
    #     strict = candidate_selection_conf.get("strict", False)
    #     strat = candidate_selection_conf.get("strat", "random").lower()

    #     masks = self.gemm_get_split_masks_general(bounds, cutoff=0)
    #     self.logger.info(f"============= Split Mask Sumamry =============")
    #     self.logger.info(f"stable+: {torch.sum(masks['stable+'])}\t"
    #                         f"stable-: {torch.sum(masks['stable-'])}")
    #     self.logger.info(f"unstable: {torch.sum(masks['unstable'])}\t"
    #                         f"all: {torch.sum(masks['all'])}")
        
    #     mask = masks[split_mask]
    #     mask_size = torch.sum(mask).item()

    #     if strict:
    #         assert n_splits is not None, "n_splits must be specified with strict=True"
    #         assert n_splits <= mask_size, f"n_splits {n_splits} exceeds the available neurons to split {mask_size}"

    #     if n_splits is None:
    #         n_splits = mask_size
    #         self.logger.info(f"n_splits is not specified, set to maximum available neurons to split: {n_splits}")

    #     if n_splits < mask_size:
    #         self.logger.info(f"n_splits < mask_size ({n_splits} < {mask_size}), applying strat {strat} to select neurons to split")
    #         if strat == "random":
    #             split_idxs = torch.nonzero(mask, as_tuple=False).squeeze().tolist()
    #             split_idxs = random.sample(split_idxs, n_splits)
    #             return split_idxs
    #         elif strat == "max_interval":
    #             output_lb, output_ub = bounds
    #             unstable_idxs = torch.nonzero(mask, as_tuple=False).squeeze().tolist()
    #             interval_sizes = [output_ub[i] - output_lb[i] for i in unstable_idxs]
    #             sorted_intervals = sorted(zip(unstable_idxs, interval_sizes), key=lambda x: x[1], reverse=True)
    #             split_idxs = [idx for idx, size in sorted_intervals[:n_splits]]
    #             return split_idxs
    #         else:
    #             raise NotImplementedError(f"strat {strat} is not implemented yet")
    #     else:
    #         self.logger.info(f"n_splits >= mask_size ({n_splits} <= {mask_size}), splitting all available neurons (Strat not applied)")
    #         return torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

            
    # responsible for identifying the split parameters
    def _decide_split_params_ReLU(self, tight_bounds, loose_bounds, split_idxs, split_conf):
        # if idx in split_idxs, use tight_bounds to decide tau
        # if idx not in split_idxs, use loose_bounds to decide tau
        split_tau_strat = split_conf.get("split_tau_strat", "midpoint").lower()
        split_scale_strat = split_conf.get("split_scale_strat", "fixed").lower()
        if split_scale_strat == "fixed":
            fixed_scales = split_conf.get("fixed_scales", (1.0, -1.0))
        if split_scale_strat == "random":
            s_pos_range = split_conf.get("s_pos_range", (0.1, 100.0))
            s_neg_range = split_conf.get("s_neg_range", (-100.0, -0.1))

        stable_tau_strat = split_conf.get("stable_tau_strat", "random").lower() # random, BigTau, SmallTau
        stable_tau_margin = split_conf.get("stable_tau_margin", (10.0, 50.0))
        stable_scale_strat = split_conf.get("stable_scale_strat", "fixed").lower()
        if stable_scale_strat == "fixed":
            stable_fixed_scales = split_conf.get("stable_fixed_scales", (1.0, -1.0))
        if stable_scale_strat == "random":
            stable_s_pos_range = split_conf.get("stable_s_pos_range", (0.1, 100.0))
            stable_s_neg_range = split_conf.get("stable_s_neg_range", (-100.0, -0.1))


        split_dict = {}
        layer_width = tight_bounds[0].shape[1]
        for i in range(layer_width):
            if i in split_idxs: # destabilized neuron
                lb, ub = tight_bounds[0][0,i].item(), tight_bounds[1][0,i].item()
                if split_tau_strat == "random":
                    tau = random.uniform(lb, ub)
                elif split_tau_strat == "midpoint":
                    tau = (lb+ub) / 2.0
                else:
                    raise NotImplementedError(f"split_tau_strat {split_tau_strat} is not implemented yet")
                # decide scales
                if split_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scales
                elif split_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"split_scale_strat {split_scale_strat} is not implemented yet")
                split_dict[i] = (tau, (s_pos, s_neg))
                
            else:   # Not destabilized neuron
                lb, ub = loose_bounds[0][0,i].item(), loose_bounds[1][0,i].item()
                _bigtau_tmp = abs(min(0, lb))
                bigtau = random.uniform(_bigtau_tmp + stable_tau_margin[0], _bigtau_tmp + stable_tau_margin[1])
                _smalltau_tmp = abs(max(0, ub))
                smalltau = -random.uniform(_smalltau_tmp + stable_tau_margin[0], _smalltau_tmp + stable_tau_margin[1])
                rand_tau = random.choice([bigtau, smalltau])
                if stable_tau_strat == "random":
                    tau = rand_tau
                elif stable_tau_strat == "BigTau":
                    tau = bigtau
                elif stable_tau_strat == "SmallTau":
                    tau = smalltau
                else:
                    raise NotImplementedError(f"stable_tau_strat {stable_tau_strat} is not implemented yet")
                # decide scales
                if stable_scale_strat == "fixed":
                    s_pos, s_neg = stable_fixed_scales
                elif stable_scale_strat == "random":
                    s_pos = random.uniform(stable_s_pos_range[0], stable_s_pos_range[1])
                    s_neg = random.uniform(stable_s_neg_range[0], stable_s_neg_range[1])
                else:
                    raise NotImplementedError(f"stable_scale_strat {stable_scale_strat} is not implemented yet")
                split_dict[i] = (tau, (s_pos, s_neg))
        return split_dict


                    
                
                
                


        # tau_strat = split_conf.get("tau_strat", "random").lower()
        # scale_strat = split_conf.get("scale_strat", "fixed").lower()
        # if scale_strat == "fixed":
        #     fixed_scales = split_conf.get("fixed_scales", (1.0, -1.0))
        # if scale_strat == "random":
        #     s_pos_range = split_conf.get("s_pos_range", (0.1, 100.0))
        #     s_neg_range = split_conf.get("s_neg_range", (-100.0, -0.1))
        
        # split_dict = {}
        # for i in idxs:
        #     # decide tau
        #     if tau_strat == "random":
        #         # print(bounds[0][i])
        #         # print(bounds[1][i])
        #         tau = random.uniform(bounds[0][0,i].item(), bounds[1][0,i].item())
        #     elif tau_strat == "midpoint":
        #         tau = (bounds[0][i].item() + bounds[1][i].item()) / 2.0
        #     else:
        #         raise NotImplementedError(f"tau_strat {tau_strat} is not implemented yet")
        #     # decide scales
        #     if scale_strat == "fixed":
        #         s_pos, s_neg = fixed_scales
        #     elif scale_strat == "random":
        #         s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
        #         s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
        #     else:
        #         raise NotImplementedError(f"scale_strat {scale_strat} is not implemented yet")
        #     split_dict[i] = (tau, (s_pos, s_neg))
        # return split_dict


    # actually split
    def _split_ReLU(self, node_to_split,split_dict):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        n_splits = len(split_dict)

        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        # create new layers: gemm_1 -> relu -> gemm_2
        num_out, num_in = w_o.shape
        gemm_1_w = torch.zeros((num_out + n_splits, num_in))
        gemm_1_b = torch.zeros((num_out + n_splits,))
        gemm_2_w = torch.zeros((num_out, num_out + n_splits))
        gemm_2_b = torch.zeros(num_out)

        idx = 0
        for i in tqdm(range(num_out), desc="Constructing new ReLU layers"):
            if i in split_dict:
                tau, (s_pos, s_neg) = split_dict[i]
                # gemm_1
                gemm_1_w[idx]       = s_pos * w_o[i]
                gemm_1_w[idx + 1]   = s_neg * w_o[i]
                gemm_1_b[idx]       = -s_pos * (tau - b_o[i])
                gemm_1_b[idx + 1]   = -s_neg * (tau - b_o[i])
                # gemm_2
                gemm_2_w[i][idx]       = 1/s_pos
                gemm_2_w[i][idx + 1]   = 1/s_neg
                gemm_2_b[i]            = tau
                
                idx += 2
            else:
                # (Future work) can use some random ratio here. or even random offset
                ratio = 1.0
                # gemm_1
                gemm_1_w[idx]     = w_o[i] * ratio
                gemm_1_b[idx]     = b_o[i]                
                # gemm_2
                gemm_2_w[i][idx]  = 1.0 / ratio
                gemm_2_b[i]       = 0.0
                idx += 1
        gemm_1_w = gemm_1_w.T
        gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        relu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_ReLU")    # intermediate output after relu
        gemm_2_output_tname  = node_to_split.output[0]                                   # final output after gemm_2, should be same as original output

        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")

        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        relu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_ReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")

        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)
        relu_node = onnx.helper.make_node(
            "Relu",
            inputs=[gemm_1_output_tname],
            outputs=[relu_output_tname],
            name=relu_nname)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[relu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)

        new_nodes = [gemm_1_node, relu_node, gemm_2_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
        ]

        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit",
            producer_name=TOOL_NAME
        )
        return new_model

    def _create_baseline_ReLU(self, split_dict):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass


    #----------------------- LeakyReLU -----------------------#
    #----------------------- LeakyReLU -----------------------#
    def decide_LeakyReLU(self, leaky_conf):
        # need return a single alpha value
        self.logger.warn("using static alpha=0.01 for LeakyReLU splitting")
        return 0.0

    def split_LeakyReLU(self, nodes, conf):
        gemm_node, activation_node = nodes
        bounding_method = conf.get("bounding_method", "backward")
        create_baseline = conf.get("create_baseline", False)
        candidate_selection_conf = conf.get("candidate_selection_conf", {})
        param_selection_conf = conf.get("param_conf", {})

        bounds = self.model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method)
        split_idxs = self._decide_split_idxs_LeakyReLU(bounds, candidate_selection_conf)
        split_dict = self._decide_split_params_LeakyReLU(bounds, split_idxs, param_selection_conf)
        split_alpha = self._decide_alpha_LeakyReLU(bounds, split_dict, conf)
        new_model = self._split_LeakyReLU(nodes ,split_dict, split_alpha)

        baseline_model = None
        if create_baseline:
            raise NotImplementedError("yo")
            baseline_model  = self._create_baseline_LeakyReLU(split_dict, alpha)
        return new_model, baseline_model

    def _decide_split_idxs_LeakyReLU(self, bounds, candidate_selection_conf):
        return self._decide_split_idxs_ReLU(bounds, candidate_selection_conf)

    def _decide_split_params_LeakyReLU(self, bounds, idxs, split_conf):
        return self._decide_split_params_ReLU(bounds, idxs, split_conf)

    def _decide_alpha_LeakyReLU(self, bounds, split_dict, extra_conf):
        # for now, just return a randome value between 0.01 and 0.1
        return random.uniform(0.01, 0.1)

    def _split_LeakyReLU(self, nodes, split_dict, alpha = 0.01):
        node_to_split, activation_node = nodes

        # split_dict: {idx: (tau, (s_pos, s_neg))}
        n_splits = len(split_dict)

        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        # adhoc
        assert activation_node.op_type == "LeakyRelu", f"Activation node is not LeakyReLU: {activation_node.op_type}"
        activation_alpha = activation_node.attribute[0].f
        print(f"Activation node alpha: {activation_alpha}")
        # alpha * new_activation_alpha == activation_alpha.f
        if activation_node.op_type == "LeakyRelu":
            last_leakyrelu_alpha = activation_alpha / alpha
        elif activation_node.op_type == "PRelu":
            last_prelu_slope = activation_alpha / alpha
        elif activation_node.op_type == "Relu":
            pass
        else:
            raise NotImplementedError(f"Activation node type {activation_node.op_type} not supported")
        print( f"Split LeakyReLU alpha: {alpha}, last alpha {last_leakyrelu_alpha}, product: {alpha * last_leakyrelu_alpha}" )
        print( f"Last LeakyReLU alpha: {activation_alpha}" )
        # end adhoc

        # create new layers: gemm_1 -> leakyrelu -> gemm_2
        num_out, num_in = w_o.shape
        gemm_1_w = torch.zeros((num_out + n_splits, num_in))
        gemm_1_b = torch.zeros((num_out + n_splits,))
        gemm_2_w = torch.zeros((num_out, num_out + n_splits))
        gemm_2_b = torch.zeros(num_out)

        idx = 0
        for i in tqdm(range(num_out), desc="Constructing new LeakyReLU layers"):
            if i in split_dict:
                tau, (s_pos, s_neg) = split_dict[i]
                # TODO: big big issue here. Post 1st leaky this is 100% original output (strait line)
                # and for non-split neuron, the negative half is scaled by alpha (kinky line)
                # they cannot share alpha.
                # split neurons has equivalent alpha = 2nd leaky alpha
                # non-split neurons has equivalent alpha = original original alpha = 1st leaky alpha * 2nd leaky alpha
                # gemm_1
                gemm_1_w[idx]       = s_pos * w_o[i]
                gemm_1_w[idx + 1]   = s_neg * w_o[i]
                gemm_1_b[idx]       = -s_pos * (tau - b_o[i])
                gemm_1_b[idx + 1]   = -s_neg * (tau - b_o[i])
                # gemm_2
                alpha_scaling = 1.0 / (1.0 + alpha)
                gemm_2_w[i][idx]       = 1/s_pos * alpha_scaling
                gemm_2_w[i][idx + 1]   = 1/s_neg * alpha_scaling
                gemm_2_b[i]            = tau
                
                idx += 2
            else:
                # (Future work) can use some random ratio here. or even random offset
                ratio = 1.0
                # gemm_1
                gemm_1_w[idx]     = w_o[i] * ratio
                gemm_1_b[idx]     = b_o[i]                
                # gemm_2
                gemm_2_w[i][idx]  = 1.0 / ratio
                gemm_2_b[i]       = 0.0
                idx += 1
        gemm_1_w = gemm_1_w.T
        gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        leakyrelu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_LeakyReLU")    # intermediate output after leakyrelu
        gemm_2_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2")              # final output after gemm_2, should be same as original output after gemm_2  
        last_activation_output_tname = activation_node.output[0]                          # final output after last activation, should be same as original output  
        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")    
        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        leakyrelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")
        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)
        leakyrelu_node = onnx.helper.make_node(
            "LeakyRelu",
            inputs=[gemm_1_output_tname],
            outputs=[leakyrelu_output_tname],
            name=leakyrelu_nname,
            alpha=alpha)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[leakyrelu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)
        
        last_activation_node = None
        if activation_node.op_type == "LeakyRelu":
            last_activation_node = onnx.helper.make_node(
                "LeakyRelu",
                inputs=[gemm_2_output_tname],
                outputs=[last_activation_output_tname],
                name=self.model.gen_node_name(prefix=f"{TOOL_NAME}_LastLeakyReLU"),
                alpha=last_leakyrelu_alpha)
        else:
            raise NotImplementedError(f"Activation node type {activation_node.op_type} not supported")

        new_nodes = [gemm_1_node, leakyrelu_node, gemm_2_node, last_activation_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
        ]
        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split, activation_node],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit",
            producer_name=TOOL_NAME
        )
        return new_model
        

    def _create_baseline_LeakyReLU(self, split_dict, alpha=0.01):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass

    #----------------------- PReLU -----------------------#
    #----------------------- PReLU -----------------------#
    def split_PReLU(self, nodes, conf):
        gemm_node, activation_node = nodes
        bounding_method = conf.get("bounding_method", "backward")
        create_baseline = conf.get("create_baseline", False)
        candidate_selection_conf = conf.get("candidate_selection_conf", {})
        param_selection_conf = conf.get("param_conf", {})

        bounds = self.model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method)
        split_idxs = self._decide_split_idxs_PReLU(bounds, candidate_selection_conf)
        split_dict = self._decide_split_params_PReLU(bounds, split_idxs, param_selection_conf)
        split_slope = self._decide_slope_PReLU(bounds, split_dict, conf)
        new_model = self._split_PReLU(gemm_node ,split_dict, split_slope)

        baseline_model = None
        if create_baseline:
            raise NotImplementedError("yo")
            baseline_model  = self._create_baseline_PReLU(split_dict, slope)
        return new_model, baseline_model
    
    def _decide_split_idxs_PReLU(self, bounds, candidate_selection_conf):
        return self._decide_split_idxs_ReLU(bounds, candidate_selection_conf)
    
    def _decide_split_params_PReLU(self, bounds, idxs, split_conf):
        return self._decide_split_params_ReLU(bounds, idxs, split_conf)
    
    def _decide_slope_PReLU(self, bounds, split_dict, extra_conf):
        # for now, just return a torch tensor of fixed slop 0.01
        slope_strategy = extra_conf.get("slope_strat", "single").lower()
        if slope_strategy == "single":          # all neurons share the same slope
            random_slope = random.uniform(0.01, 0.9)
            slope = torch.tensor(random_slope)
            return slope
        elif slope_strategy == "per_neuron":    # each neuron has its own slope
            layer_width = bounds[0].shape[1]
            slope = torch.tensor([0.234 * layer_width])
            slope = torch.randn(layer_width) * 0.9 + 0.01
            return slope.abs()
        else:
            raise NotImplementedError(f"slope_strategy {slope_strategy} is not implemented yet")
        
    
    def _split_PReLU(self, node_to_split, split_dict, slope):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        n_splits = len(split_dict)

        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        # create new layers: gemm_1 -> prelu -> gemm_2
        num_out, num_in = w_o.shape
        gemm_1_w = torch.zeros((num_out + n_splits, num_in))
        gemm_1_b = torch.zeros((num_out + n_splits,))
        gemm_2_w = torch.zeros((num_out, num_out + n_splits))
        gemm_2_b = torch.zeros(num_out)

        idx = 0
        actual_slope = torch.zeros((num_out + n_splits,))
        for i in tqdm(range(num_out), desc="Constructing new PReLU layers"):
            if isinstance(slope, float) or slope.ndim == 0:
                    s = slope
            else:
                s = slope[i]

            if i in split_dict:
                tau, (s_pos, s_neg) = split_dict[i]
                # gemm_1
                gemm_1_w[idx]       = s_pos * w_o[i]
                gemm_1_w[idx + 1]   = s_neg * w_o[i]
                gemm_1_b[idx]       = -s_pos * (tau - b_o[i])
                gemm_1_b[idx + 1]   = -s_neg * (tau - b_o[i])
                # gemm_2
                slope_scaling = 1.0 / (s + 1.0)
                gemm_2_w[i][idx]       = 1/s_pos * slope_scaling
                gemm_2_w[i][idx + 1]   = 1/s_neg * slope_scaling
                gemm_2_b[i]            = tau

                actual_slope[idx] = s
                actual_slope[idx + 1] = s
                
                idx += 2
            else:
                # (Future work) can use some random ratio here. or even random offset
                ratio = 1.0
                # gemm_1
                gemm_1_w[idx]     = w_o[i] * ratio
                gemm_1_b[idx]     = b_o[i]                
                # gemm_2
                gemm_2_w[i][idx]  = 1.0 / ratio
                gemm_2_b[i]       = 0.0

                actual_slope[idx] = s

                idx += 1
        gemm_1_w = gemm_1_w.T
        gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        prelu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_PReLU")    # intermediate output after prelu
        gemm_2_output_tname  = node_to_split.output[0]                                   # final output after gemm_2, should be same as original output after gemm_2
        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        prelu_slope_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_PReLUslope")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")
        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        prelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_PReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")
        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)
        prelu_node = onnx.helper.make_node(
            "PRelu",
            inputs=[gemm_1_output_tname, prelu_slope_tname],
            outputs=[prelu_output_tname],
            name=prelu_nname)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[prelu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=0,
            transA=0)
        new_nodes = [gemm_1_node, prelu_node, gemm_2_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(prelu_slope_tname, onnx.TensorProto.FLOAT, actual_slope.shape, actual_slope.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
        ]
        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit",
            producer_name=TOOL_NAME
        )
        return new_model


    def _create_baseline_PReLU(self, node_to_split,split_dict, slope):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass


    #----------------------- ThresholdedReLU -----------------------#
    #----------------------- ThresholdedReLU -----------------------#
    def split_ThresholdedReLU(self, split_strat, tau_strat, scale_strat, alpha=1.0, create_baseline=False):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY


        split_mask = None
        taus = None
        scales = None
        split_dict = {}


        if create_baseline:
            split_model     = self._split_ThresholdedReLU(split_dict)
            baseline_model  = self._create_baseline_ThresholdedReLU(split_dict)
            return split_model, baseline_model

        new_model = self._split_ThresholdedReLU(split_dict)
        return new_model
    def _split_ThresholdedReLU(self, split_dict, alpha=1.0):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass
    def _create_baseline_ThresholdedReLU(self, split_dict, alpha=1.0):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass


