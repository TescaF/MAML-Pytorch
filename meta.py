from scipy.stats import norm
import math
import itertools
import pdb
import  torch
#from torch.utils.tensorboard import SummaryWriter
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from torch.distributions import Normal
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    learner import Learner
from    copy import deepcopy

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))
    pdb.set_trace()

def print_backward_stack(mod):
    l = mod.grad_fn
    while l is not None:
        print(l)
        if len(l.next_functions) == 0:
            return
        l = l.next_functions[0][0]


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, out_dim, loss_fn=F.cross_entropy, accs_fn=torch.eq):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config)
        self.inner_net = Learner(config)
        self.ft_net = Learner(config[:-1])
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.opt = optim.Adam(self.ft_net.parameters(), lr=self.update_lr)
        self.loss_fn = loss_fn
        self.accs_fn = accs_fn
        self.output_dim = out_dim
        self.lmb = args.lmb #lambda
        self.pos_matrix = None

    def polar_loss(self, logits, y):
        pred = []
        r = int(math.sqrt(logits.shape[-1]))
        c = float(r)/2.0
        if self.pos_matrix is None:
            self.pos_matrix = torch.stack([torch.zeros(r,r),torch.zeros(r,r)],-1).float().cuda()
            for i in range(r):
                for j in range(r):
                    a = 2*np.pi*i/r
                    sx = int(np.floor((j/2) * math.cos(a))+c)
                    sy = int(np.floor((j/2) * math.sin(a))+c)
                    if sx in range(r) and sy in range(r):
                        self.pos_matrix[j,i,0] = sy
                        self.pos_matrix[j,i,1] = sx 
        for i in range(logits.shape[0]):
            s = F.softmax(logits[i],0).reshape(r,r)
            pos = torch.mul(torch.stack([s,s],-1),self.pos_matrix)
            pred.append((torch.sum(pos.reshape((-1,2)),dim=0)-c)/c)
        return F.mse_loss(torch.stack(pred),y)

    def avg_pred(self, logits):
        pred = []
        r = int(math.sqrt(logits.shape[-1]))
        c = float(r)/2.0
        if self.pos_matrix is None:
            self.pos_matrix = torch.stack(torch.meshgrid([torch.arange(r),torch.arange(r)]),-1).float().cuda()
        for i in range(logits.shape[0]):
            s = F.softmax(logits[i],0).reshape(r,r)
            pos = torch.mul(torch.stack([s,s],-1),self.pos_matrix)
            pred.append((torch.sum(pos.reshape((-1,2)),dim=0)-c)/c)
        return torch.stack(pred)

    def avg_loss(self, logits, y):
        pred = self.avg_pred(logits)
        return torch.mean(torch.sum((pred-y)**2.0,dim=1)) #F.mse_loss(pred,y)

    def direct_update(self, x_spt, y_spt, x_qry, y_qry):
        # Update
        loss = 0
        for i in range(x_spt.shape[0]):
            logits = self.net(x_spt[i], None, bn_training=True)
            loss += self.loss_fn(logits, y_spt[i])
        total_loss = loss / x_spt.shape[0]
        self.meta_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        p = self.net.parameters()
        max_grad = 0
        for p in list(self.net.parameters()):
            max_grad = max(max_grad, p.grad.data.norm(2).item())
        self.meta_optim.step()

        # Post-update training loss
        loss = 0
        cam_vals = []
        for i in range(x_spt.shape[0]):
            logits = self.net(x_spt[i], None, bn_training=True)
            loss += self.loss_fn(logits, y_spt[i])
            cam_vals.append(self.net(x_spt[i],None,bn_training=True,hook=3))
        total_loss = loss / x_spt.shape[0]

        # Post-update testing loss
        test_loss = 0
        for i in range(x_qry.shape[0]):
            logits = self.net(x_qry[i], None, bn_training=True)
            test_loss += self.loss_fn(logits, y_qry[i])
        total_test_loss = test_loss / x_qry.shape[0]

        return total_loss.item(), total_test_loss.item(), cam_vals, max_grad

    def update(self, net, update_step, data_in, y_spt, y_qry, class_tgt, spt_idx, qry_idx):
        sig = nn.Sigmoid()
        hook = len(list(net.config))-1
        test_corrects = [0 for _ in range(self.update_step + 1)]
        ft_train_corrects = [0 for _ in range(self.update_step)]
        pt_train_corrects = [0 for _ in range(self.update_step)]

        with torch.no_grad():
            logits_q = net(data_in, vars=None)[spt_idx:qry_idx] 
            loss_q = self.loss_fn(logits_q, y_qry)
            test_corrects[0] += loss_q.item()

        for k in range(update_step):
            logits_a = sig(torch.sum(net(data_in, vars=(None if k==0 else s_weights)),dim=1)-1)
            loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), class_tgt)
            ft_train_corrects[k] += loss_a.item()

            with torch.no_grad():
                input_r = net(data_in, vars=(None if k==0 else s_weights),hook=hook)[:spt_idx]
            logits_r = F.linear(input_r, (net.parameters()[-2] if k==0 else s_weights[-2]), (net.parameters()[-1] if k==0 else s_weights[-1]))
            loss_r = self.loss_fn(logits_r, y_spt)

            grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
            grad_r = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights), allow_unused=True)) 
            grad[-2] += grad_r[-2]
            grad[-1] += grad_r[-1]
            pt_train_corrects[k] += loss_r.item()

            s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

            ## Get test loss
            with torch.no_grad():
                logits_q = net(data_in, vars=s_weights)[spt_idx:qry_idx]
                loss_q = self.loss_fn(logits_q, y_qry)
                test_corrects[k+1] += loss_q.item()

        return loss_q, [test_corrects, ft_train_corrects, pt_train_corrects], s_weights

    def class_forward(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        task_num = x_spt.size(0)
        losses_q = 0
        test_loss = [0 for _ in range(self.update_step + 1)]
        ft_train_loss = [0 for _ in range(self.update_step)]
        pt_train_loss = [0 for _ in range(self.update_step)]
        idx_a = x_spt.shape[1]
        idx_b = idx_a + x_qry.shape[1]
        tgt_set = torch.cat([torch.ones(idx_b).long().cuda(), torch.zeros(n_spt.shape[1]).long().cuda()]) 

        for i in range(task_num):
            full_set = torch.cat([x_spt[i],x_qry[i],n_spt[i]])
            loss_q, losses, w = update(self.net, self.update_step, full_set, y_spt[i], y_qry[i], tgt_set, idx_a, idx_b)
            test_loss = [test_loss[j] + losses[0][j] for j in range(len(test_loss))]
            ft_train_loss = [ft_train_loss[j] + losses[1][j] for j in range(len(ft_train_loss))]
            pt_train_loss = [pt_train_loss[j] + losses[2][j] for j in range(len(pt_train_loss))]
            losses_q += loss_q

        ## Get total losses after all tasks
        total_loss =  losses_q / task_num
        self.meta_optim.zero_grad()
        total_loss.backward()

        ## Clip gradient
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.meta_optim.step()
        loss = total_loss.item()

        return loss, [np.array(test_losses)/task_num, np.array(pt_train_loss)/task_num, np.array(ft_train_loss)/task_num]

    def forward(self, x_spt, y_spt, x_qry, y_qry,print_flag=False):
        task_num = x_spt.size(0)
        losses_q, losses_s = 0, 0
        corrects = [0 for _ in range(self.update_step + 1)]
        train_corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            s_weights = self.net.parameters()
            ## Get loss before first update
            logits_q = self.net(x_qry[i], None, bn_training=True)
            with torch.no_grad():
                if self.accs_fn is None:
                    correct = self.loss_fn(logits_q, y_qry[i]).item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    del pred_q
                corrects[0] = corrects[0] + correct

            ## Update model w.r.t. task loss
            for k in range(self.update_step):
                logits_a = self.net(x_spt[i], vars=s_weights, bn_training=True)
                loss = self.loss_fn(logits_a, y_spt[i])
                train_corrects[k] = train_corrects[k] + loss.item()
                grad_a = list(torch.autograd.grad(loss, s_weights,allow_unused=True))
                for g in range(len(grad_a)):
                    if grad_a[g] is None:
                        pdb.set_trace()
                        grad_a[g] = torch.zeros_like(s_weights[g])
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, s_weights)))  
                del loss, grad_a, logits_a

                # Get accuracy after update
                logits_q = self.net(x_qry[i], s_weights, bn_training=True)
                with torch.no_grad():
                    if self.accs_fn is None:
                        correct = self.loss_fn(logits_q, y_qry[i]).item()
                    else:
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = self.accs_fn(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        del pred_q
                    corrects[k + 1] = corrects[k + 1] + correct

            ## Get post-tuning losses for task and AL objectives
            losses_q += self.loss_fn(logits_q, y_qry[i])
            del s_weights,logits_q
            torch.cuda.empty_cache()

        ## Get total losses after all tasks
        total_loss =  losses_q / task_num #) + (losses_q/losses_s)
        ## Optimize parameters
        self.meta_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        p = self.net.parameters()
        max_grad = 0
        for p in list(self.net.parameters()):
            max_grad = max(max_grad, p.grad.data.norm(2).item())
        self.meta_optim.step()
        loss = total_loss.item()
        del total_loss, losses_q

        accs = np.array(corrects) / task_num
        return accs, loss, np.array(train_corrects)/task_num, max_grad

    def finetuning(self, x_spt, y_spt, x_qry, y_qry, debug=False):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        hook = len(list(self.net.parameters()))-1

        net = deepcopy(self.net)
        fast_weights = list(net.parameters())

        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            if self.accs_fn is None:
                correct = self.loss_fn(logits_q,y_qry).item()
            else:
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = self.accs_fn(pred_q, y_qry).sum().item()/querysz
            corrects[0] = corrects[0] + correct

        ## Update model w.r.t. task loss
        for k in range(self.update_step_test):
            logits_a = self.net(x_spt, vars=fast_weights, bn_training=True)
            loss = self.loss_fn(logits_a, y_spt)
            grad_a = list(torch.autograd.grad(loss, fast_weights,allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    grad_a[g] = torch.zeros_like(fast_weights[g])
            fast_weights = list(map(lambda p: p[1] - (self.update_lr * p[0]), zip(grad_a, fast_weights)))
            del loss, logits_a, grad_a

            # Get accuracy after update
            with torch.no_grad():
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                if self.accs_fn is None:
                    correct = self.loss_fn(logits_q, y_qry).item()
                else:
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = self.accs_fn(pred_q, y_qry).sum().item()/querysz  # convert to numpy
                    del pred_q
                corrects[k + 1] = corrects[k + 1] + correct
        cam_vals1 = net(x_spt,vars=fast_weights,bn_training=True,hook=hook-2,debug=debug)

        del net
        accs = np.array(corrects) 
        return accs,[cam_vals1], logits_q

    def class_tune4(self, n_spt, x_spt, y_spt, x_qry, y_qry,debug=False):

        net = deepcopy(self.net)
        s_weights = net.parameters()

        sig = nn.Sigmoid()
        hook = len(list(self.net.config))-1
        test_corrects = [0 for _ in range(self.update_step_test + 1)]
        pt_corrects = [0 for _ in range(self.update_step_test + 1)]
        ft_corrects = [0 for _ in range(self.update_step_test + 1)]

        full_set = torch.cat([x_spt,x_qry,n_spt])
        tgt_set = torch.cat([torch.ones(x_spt.shape[0] + x_qry.shape[0]).long().cuda(), torch.zeros(n_spt.shape[0]).long().cuda()]) #.reshape(-1,1)
        idx_a = x_spt.shape[0]
        idx_b = idx_a + x_qry.shape[0]

        with torch.no_grad():
            #logits_q = net(x_qry, vars=net.parameters()) #, post_hook=hook)
            logits_q = net(full_set, vars=net.parameters())[idx_a:idx_b] #, post_hook=hook)
            loss_q1 = self.loss_fn(logits_q, y_qry)
            test_corrects[0] += loss_q1.item()

            logits_a = sig(torch.sum(net(full_set, vars=net.parameters()),dim=1) - 1)
            loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), tgt_set)
            ft_corrects[0] += loss_a.item()

            ## Get location loss
            #logits_r = net(x_spt, vars=net.parameters()) #, post_hook=hook)
            logits_r = net(full_set, vars=net.parameters())[:idx_a] #, post_hook=hook)
            loss_r = self.loss_fn(logits_r, y_spt)
            pt_corrects[0] += loss_r.item()

        '''k = 0
        while k < self.update_step or sum_grad > 1:
            ## Get classification loss
            logits_a = sig(net(full_set, vars=(None if k==0 else s_weights)) - 1)
            loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), tgt_set)
            grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights)))
            s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
            sum_grad = 0
            for g in grad:
                sum_grad += g.norm(2).item()
            k += 1'''

        for k in range(self.update_step_test):
            #logits_a = sig(net(full_set, vars=s_weights) - 1)
            logits_a = sig(torch.sum(net(full_set, vars=(None if k==0 else s_weights)),dim=1) - 1)
            #logits_a = sig(net(full_set, vars=(None if k==0 else s_weights)) - 1)
            loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), tgt_set)
            ft_corrects[k+1] += loss_a.item()
            #grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
            #grad = list(torch.autograd.grad(loss_total, (net.parameters() if k==0 else s_weights))) 
            #s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

            ## Get location loss
            #logits_r = net(full_set, vars=s_weights)[:x_spt.shape[0]]
            with torch.no_grad():
                input_r = net(full_set, vars=(None if k==0 else s_weights),hook=hook)[:idx_a]
            logits_r = F.linear(input_r, (net.parameters()[-2] if k==0 else s_weights[-2]), (net.parameters()[-1] if k==0 else s_weights[-1]))
            #logits_r = net(full_set, vars=(None if k==0 else s_weights))[:x_spt.shape[0]]
            loss_r = self.loss_fn(logits_r, y_spt)
            pt_corrects[k+1] += loss_r.item()

            grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
            grad2 = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights), allow_unused=True)) 
            grad[-2] += grad2[-2]
            grad[-1] += grad2[-1]
            #s_weights[-2:] = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights[-2:])))
            s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

            ## Get test loss
            with torch.no_grad():
                logits_q = net(full_set, vars=s_weights)[idx_a:idx_b] #, post_hook=hook)
                #logits_q = net(x_qry, vars=s_weights) #, post_hook=hook)
                loss_q = self.loss_fn(logits_q, y_qry)
                test_corrects[k+1] += loss_q.item()
            del loss_r, logits_r

        '''spt_logits = net(x_spt, vars=s_weights, bn_training=True, post_hook=hook)
        spt_pred = self.avg_pred(spt_logits)
        qry_logits = net(x_qry, vars=s_weights, bn_training=True, post_hook=hook)
        qry_pred = self.avg_pred(qry_logits)'''
        all_logits = net(full_set, vars=s_weights) #, post_hook=hook)
        loss = self.loss_fn(all_logits[:idx_a], y_spt).item()
        qry_pred = self.avg_pred(all_logits[idx_a:idx_b])

        #cam_vals1 = net(x_spt,vars=s_weights,bn_training=True) #,post_hook=hook,debug=debug)
        cam_vals2 = torch.mm(net(full_set,vars=s_weights,bn_training=True,hook=hook,debug=debug)[idx_a:idx_b], s_weights[-2])
        del all_logits, s_weights
        return loss, [cam_vals2],  [np.array(test_corrects),np.array(ft_corrects),np.array(pt_corrects)], qry_pred

def main():
    pass


if __name__ == '__main__':
    main()
