import pdb
import  os.path
import  numpy as np


class Polynomial:

    def __init__(self, batchsz, k_shot, k_qry):
        """
        :param batchsz: task num
        :param k_shot: number of samples for fine-tuning
        :param k_qry:
        :param imgsz:
        """
        self.batch_size = batchsz
        self.param_ranges = [   [-2.0, 3.0],
                                [-2.0, 3.0],
                                [-2.0, 3.0]]
        self.input_range = [-5.0, 5.0]
        self.dim_input = 2
        self.dim_output = 1

        self.num_samples_per_class = k_shot + k_qry 

    def next(self):
        params = []
        for p in self.param_ranges:
            params.append(np.random.uniform(p[0], p[1], [self.batch_size]))
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, self.dim_input])
            outputs[func] = np.expand_dims((params[0][func] * init_inputs[func,:,0]**2.0) + (params[1][func] * init_inputs[func,:,1]) + params[2][func], axis=1)
        return init_inputs, outputs


