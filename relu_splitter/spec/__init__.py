default_logger = logging.getLogger(__name__)


class warpped_vnnlib():
    def __init__(self, vnnlib_rep, input_shape = None， loger=default_logger):
        self.logger = logger
        self.input_shape = input_shape
        
        input_bounds, output_bounds = vnnlib_rep[0]

        self.input_lb = [i[0] for i in input_bounds]
        self.input_ub = [i[1] for i in input_bounds]
        assert all([ lb <= ub for lb, ub in zip(self.input_lb, self.input_ub)]), "Input lower bound is greater than upper bound"

        self.spec_num_inputs = reduce(lambda x, y: x*y, self.input_lb.shape)
        if self.input_shape is not None:
            self.specified_num_inputs = reduce(lambda x, y: x*y, self.input_shape)
            assert self.spec_num_inputs == reduce(lambda x, y: x*y, self.input_shape), f"Spec number of inputs does not match model inputs {self.spec_num_inputs} != {reduce(lambda x, y: x*y, self.input_shape)}"
        else:
            self.specified_num_inputs = None
            self.logger.warning("Input shape is not specified, not checking # input in vnnlib file...")

        self.output_specs = 

        self.bounded_tensor = None

    @property
    def bounded_tensor(self):
        if self._bounded_tensor is None:
            # TODO: convert to nhwc
            self._bounded_tensor = BoundedTensor(torch.tensor([self.input_lb]), PerturbationLpNorm(x_L=torch.tensor([self.input_lb]), x_U=torch.tensor([self.input_ub])))
        return self._bounded_tensor


    def serialize(self):
        pass

    def save(self, fname:Path):
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        with open(fname, "w") as f:
            f.write(self.serialize())
        return fname



    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.warpped_model.num_inputs
        # reshape input bounds to match the model input shape
        input_lb, input_ub = input_lb.view(self.input_shape), input_ub.view(self.input_shape)
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"
        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))