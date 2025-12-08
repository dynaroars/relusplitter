import random

import torch


class Rsplitter_Gemm():
    split_activations = ["ReLU", "LeakyReLU", "PReLU", "ThresholdedReLU"]

    def gemm_split(self, gemm_node, conf):
        assert gemm_node.op_type == "Gemm", f"Node to split is not a Gemm node: {gemm_node.op_type}"
        assert conf.get("split_activation", None) in self.split_activations, f"Unsupported split activation {conf.get('split_activation', None)}"

        split_activation = conf["split_activation"]
        if split_activation == "ReLU":
            split_model, baseline_model = self.split_ReLU(gemm_node, conf)
        elif split_activation == "LeakyReLU":
            split_model, baseline_model = self.split_LeakyReLU(gemm_node, conf)
        elif split_activation == "PReLU":
            split_model, baseline_model = self.split_PReLU(gemm_node, conf)
        elif split_activation == "ThresholdedReLU":
            split_model, baseline_model = self.split_ThresholdedReLU(gemm_node, conf)
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
    
    def gemm_get_taus(self, bounds, idxs, tau_strat, extra_offset=0):
        if tau_strat == "random":
            taus = [random.uniform(bounds[0][i] + extra_offset, bounds[1][i] + extra_offset) for i in idxs]
        raise NotImplementedError(f"tau_strat {tau_strat} is not implemented yet")
    
    def gemm_get_scales(self, bounds, idxs, scale_strat):
        if scale_strat == "random":
            scales = [ (random.uniform(0.1, 100), random.uniform(-100, -0.1)) for i in idxs ]
        elif scale_strat == "static":
            scales = [ (1,-1) for i in idxs ]
        else:
            raise NotImplementedError(f"scale_strat {scale_strat} is not implemented yet")
        return scales



    #----------------------- ReLU -----------------------#
    #----------------------- ReLU -----------------------#
    def split_ReLU(self, gemm_node, conf):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY
        # TODO: resume here
        bounding_method = conf["bounding_method"]
        split_strat = conf["split_strat"]
        tau_strat = conf["tau_strat"]
        scale_strat = conf["scale_strat"]
        create_baseline = conf["create_baseline"]

        bounds = self.model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method)
        split_masks = self.gemm_get_split_masks_general(bounds, cutoff=0)
        taus = self.get_taus_ReLU(bounds, split_mask, tau_strat)
        scales = None
        split_dict = {}


        if create_baseline:
            split_model     = self._split_ReLU(split_dict)
            baseline_model  = self._create_baseline_ReLU(split_dict)
            return split_model, baseline_model

        new_model = self._split_ReLU(split_dict)
        return new_model


    def _split_ReLU(self, split_dict):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass
    def _create_baseline_ReLU(self, split_dict):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass


    #----------------------- LeakyReLU -----------------------#
    #----------------------- LeakyReLU -----------------------#
    def decide_LeakyReLU(self, leaky_conf):
        # need return a single alpha value
        self.logger.warn("using static alpha=0.01 for LeakyReLU splitting")
        return 0.01

    def split_LeakyReLU(self, split_strat, tau_strat, scale_strat, leaky_conf, create_baseline=False):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY

        split_mask = None
        taus = None
        scales = None
        alpha = self.decide_alpha_LeakyReLU(leaky_conf)
        split_dict = {} # split_dict: {idx: (tau, (s_pos, s_neg))}

        if create_baseline:
            split_model     = self._split_LeakyReLU(split_dict)
            baseline_model  = self._create_baseline_LeakyReLU(split_dict)
            return split_model, baseline_model

        new_model = self._split_LeakyReLU(split_dict)
        return new_model

    def _split_LeakyReLU(self, split_dict, alpha=0.01):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass

    def _create_baseline_LeakyReLU(self, split_dict, alpha=0.01):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass

    #----------------------- PReLU -----------------------#
    #----------------------- PReLU -----------------------#
    def split_PReLU(self, split_strat, tau_strat, scale_strat, slope=0.01, create_baseline=False):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY


        split_mask = None
        taus = None
        scales = None
        split_dict = {}


        if create_baseline:
            split_model     = self._split_PReLU(split_dict)
            baseline_model  = self._create_baseline_PReLU(split_dict)
            return split_model, baseline_model

        new_model = self._split_PReLU(split_dict)
        return new_model
    
    def _split_PReLU(self, split_dict, slope=0.01):
        # split_dict: {idx: (tau, (s_pos, s_neg))}
        pass

    def _create_baseline_PReLU(self, split_dict, slope=0.01):
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


