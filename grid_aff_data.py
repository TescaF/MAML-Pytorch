import scipy.io
import pdb
import sys
import numpy as np
import pickle
from numpy.random import RandomState
import os.path

class Affordances:
    def __init__(self, inputs, train, exclude, samples, batchsz, k_shot, k_qry, dim_out):
        self.aff_ds = "/u/tesca/data/downsized/"
        self.rand = RandomState(222)
        self.sample_size = samples

        with open(os.path.expanduser("~") + "/data/obj_keys.pkl", 'rb') as handle:
            keys = pickle.load(handle)
        categories = list(sorted(set([k.split("_")[0] for k in keys])))
        if train:
            valid_keys = [k for k in keys if not k.startswith(categories[exclude])]
        else:
            valid_keys = [k for k in keys if k.startswith(categories[exclude])]
        self.valid_keys = list(sorted(set(valid_keys)))
        self.valid_objs = list(sorted(set([k.split("_00")[0] for k in self.valid_keys])))
        self.valid_categories = [c for c in categories if c is not categories[exclude]]

        self.exclude = exclude
        print("Categories: " + str(categories))
        print(("Excluding" if train else "Testing on") + " category '" + str(categories[exclude]) + "'")

        self.batch_size = batchsz
        self.ds_dim = dim_out
        self.num_objs_per_batch = k_shot + k_qry

        self.large_categories = []
        obj_counts = [o.split("_")[0] for o in self.valid_objs]
        objs = list(sorted(set(obj_counts)))
        for o in objs:
            if obj_counts.count(o) >= self.num_objs_per_batch:
                self.large_categories.append(o)
        print(self.large_categories)

    def select_keys(self):
        pos_keys, valid_affs = [],[]
        c = self.rand.choice(len(self.large_categories), self.batch_size, replace=True)
        # Each "batch" is an object class
        for t in range(self.batch_size):
            # Get set of negative examples for img classification
            p_keys = []
            pos_objs = sorted([o for o in self.valid_objs if o.startswith(self.large_categories[c[t]])])
            k = self.rand.choice(len(pos_objs), self.num_objs_per_batch, replace=False)
            v_affs = []
            for n in range(self.num_objs_per_batch):
                sample_keys = sorted([key for key in self.valid_keys if key.startswith(pos_objs[k[n]])])
                sk = self.rand.choice(len(sample_keys), self.sample_size, replace=False)
                for s in range(self.sample_size):
                    p_keys.append(sample_keys[sk[s]])
                    affs = scipy.io.loadmat(self.aff_ds + p_keys[-1] + "_downsize_" + str(self.ds_dim) + ".mat")['gt_label']
                    for a in range(affs.shape[0]-1):
                        valid = (affs[a+1].sum() > 0)
                        if len(v_affs) <= a:
                            v_affs.append(valid)
                        else:
                            v_affs[a] = v_affs[a] and valid
            pos_keys.append(p_keys)
            aff_idxs = [k for k in range(len(v_affs)) if v_affs[k] == True]
            a = self.rand.choice(len(aff_idxs))
            valid_affs.append(aff_idxs[a]+1)
        return pos_keys, valid_affs

    def next(self):
        pos_keys, aff_idxs = self.select_keys()
        outputs = np.zeros([self.batch_size, self.num_objs_per_batch * self.sample_size, self.ds_dim, self.ds_dim])
        # Each "batch" is an object class
        for t in range(self.batch_size):
            output_list = []
            # Number of objects per class
            for c in range(self.num_objs_per_batch):
                # Number of images per object
                dims = []
                pos = pos_keys[t][c*self.sample_size:]
                for n in range(self.sample_size):
                    affs = scipy.io.loadmat(self.aff_ds + pos[n] + "_downsize_" + str(self.ds_dim) + ".mat")['gt_label'][aff_idxs[t]]
                    output_list.append(affs)
            outputs[t] = np.stack(output_list)
        return pos_keys, outputs
