import math
import pdb
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from copy import deepcopy


class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'reweight':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(param[0]))
                # gain=1 according to cbfinn's implementation
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param[:2]))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','softmax','polar']:
                continue
            else:
                raise NotImplementedError


    def apply_polar_tf(self, x):
        '''d = int(x.shape[-1]/2)
        tf = torch.zeros_like(x).cuda()
        for i in range(x.shape[-2]):
            for j in range(x.shape[-1]):
                dist = x.shape[-1] * np.sqrt((i-d)**2 + (j-d)**2) / (0.5*np.sqrt(2*(x.shape[-1]**2)))
                if dist == 0:
                    continue
                r = int(np.round(dist-np.sqrt(2)))
                #r = int(np.floor(x.shape[-1] * np.log(np.sqrt((i-d)**2 + (j-d)**2)) / np.log(x.shape[-1])))
                a = int(np.round((x.shape[-1] / (2 * np.pi)) * (math.atan2(j-d,i-d)))%x.shape[-1])
                tf[:,r,a] = x[:,i,j]
        return tf'''
        d = int(x.shape[-1]/2)
        w = int(x.shape[-1])
        tf = torch.zeros_like(x).cuda()
        for i in range(x.shape[-1]):
            for j in range(x.shape[-2]):
                r = j/2
                a = 2*np.pi*i/w
                sx = int(np.floor(r * np.cos(a))+d)
                sy = int(np.floor(r * np.sin(a))+d)
                if sx in range(w) and sy in range(w):
                    tf[:,j,i] = x[:,sy,sx]
        return tf


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'reweight':
                tmp = 'reweight:(in:%d, out:%d)'%(param[0], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['polar','flatten', 'tanh', 'relu', 'upsample', 'reshape', 'softmax','sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def grad_hook(self, grad):
        self.grads.append(grad)

    def forward(self, x, vars=None, bn_training=True,dropout_rate=-1, hook=None, grad_hook=None, last_layer=False,debug=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        if vars is None:
            vars = self.vars
        if last_layer:
            c = -1
            idx = len(vars)-2
            bn_idx = len(self.vars_bn)
        else:
            c = 0
            idx = 0 
            bn_idx = 0
        p = 0

        self.grads = []
        for name, param in self.config[c:]:
            if debug:
                pdb.set_trace()
            if hook == p:
                hook_data = x
            if grad_hook == p:
                h = x.register_hook(self.grad_hook)

            p += 1
            if dropout_rate > 0:
                x = F.dropout(x, p=dropout_rate, training=True)
            #if torch.isnan(x).any():
            #    pdb.set_trace()
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'reweight':
                w = vars[idx]
                x = torch.mul(x, w)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'polar':
                x = self.apply_polar_tf(x)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                if param[2]:
                    x = F.linear(x, w, b)
                else:
                    x = F.linear(x, w, None)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = torch.tanh(x)
                #x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'softmax':
                x = torch.softmax(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                pdb.set_trace()
                x = F.avg_pool2d(x, param[0], param[1], param[2])
                pdb.set_trace()

            else:
                raise NotImplementedError

            if hook is not None:
                return hook_data

        # make sure variable is used properly
        if not idx == len(vars):
            pdb.set_trace()
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        if hook is None:
            return x
        else:
            return hook_data


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
