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
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
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
        return F.mse_loss(pred,y)

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

    def class_forward(self, n_spt, x_spt, y_spt, x_qry, y_qry,debug=False):
        task_num = x_spt.size(0)
        losses_q, losses_s = 0, 0
        test_corrects = [0 for _ in range(self.update_step + 1)]
        train_corrects = [0 for _ in range(self.update_step)]
        loss_fn = nn.BCEWithLogitsLoss()
        hook = len(list(self.net.parameters()))-1

        for i in range(task_num):
            s_weights = self.net.parameters()
            with torch.no_grad():
                logits_q = self.net(x_qry[i], vars=s_weights, bn_training=True, hook=hook)
                loss_q = self.loss_fn(logits_q, y_qry[i])
                test_corrects[0] += loss_q.item()

            for k in range(self.update_step):
                ## Get classification loss
                logits_a = self.net(x_spt[i], vars=s_weights, bn_training=True)
                logits_b = self.net(x_qry[i], vars=s_weights, bn_training=True)
                logits_c = self.net(n_spt[i], vars=s_weights, bn_training=True)
                lossa = loss_fn(logits_a, torch.ones(x_spt.shape[1]).float().cuda().unsqueeze(1))/x_spt.shape[1]
                lossb = loss_fn(logits_b, torch.ones(x_qry.shape[1]).float().cuda().unsqueeze(1))/x_qry.shape[1]
                lossc = loss_fn(logits_c, torch.zeros(n_spt.shape[1]).float().cuda().unsqueeze(1))/n_spt.shape[1]

                ## Get location loss
                logits_r = self.net(x_spt[i], vars=s_weights, bn_training=True, hook=hook)
                lossr = self.loss_fn(logits_r, y_spt[i])
                train_corrects[k] += lossr.item()

                ## Update weights
                grad_a = list(torch.autograd.grad(lossa+lossb+lossc+(self.lmb*lossr), s_weights,allow_unused=True))
                for g in range(len(grad_a)):
                    if grad_a[g] is None:
                        pdb.set_trace()
                        grad_a[g] = torch.zeros_like(s_weights[g])
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, s_weights)))  

                ## Get test loss
                with torch.no_grad():
                    logits_q = self.net(x_qry[i], vars=s_weights, bn_training=True, hook=hook)
                    loss_q = self.loss_fn(logits_q, y_qry[i])
                    test_corrects[k+1] += loss_q.item()
                del lossa, lossb, lossc, lossr, grad_a, logits_a, logits_b, logits_c, logits_r

            ## Get post-tuning losses for location objectives
            logits_q = self.net(x_qry[i], vars=s_weights, bn_training=True, hook=hook)
            losses_q += self.loss_fn(logits_q, y_qry[i])
            del s_weights,logits_q
            torch.cuda.empty_cache()

        ## Get total losses after all tasks
        total_loss =  losses_q / task_num #) + (losses_q/losses_s)
        self.meta_optim.zero_grad()
        total_loss.backward()
        ## Clip gradient
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        p = self.net.parameters()
        max_grad = 0
        for p in list(self.net.parameters())[:-2]:
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.data.norm(2).item())
        self.meta_optim.step()
        loss = total_loss.item()
        del total_loss, losses_q

        accs = np.array(test_corrects) / task_num
        return accs, loss, np.array(train_corrects)/task_num, max_grad

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

    def class_finetuning(self, n_spt, x_spt, y_spt, x_qry, y_qry, debug=False):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        loss_fn = nn.BCEWithLogitsLoss()

        hook = len(list(self.net.parameters()))-1
        net = deepcopy(self.net)
        fast_weights = list(net.parameters())
        with torch.no_grad():
            logits_q = net(x_qry, vars=fast_weights, bn_training=True, hook=hook)
            loss_q = self.loss_fn(logits_q, y_qry)
            corrects[0] += loss_q.item()
        for k in range(self.update_step_test):
            ## Update model w.r.t. classification loss
            logits_a = net(x_spt, vars=fast_weights, bn_training=True)
            logits_b = net(x_qry, vars=fast_weights, bn_training=True)
            logits_c = net(n_spt, vars=fast_weights, bn_training=True)
            logits_r = net(x_spt, vars=fast_weights, bn_training=True, hook=hook)
            lossa = loss_fn(logits_a, torch.ones(x_spt.shape[0]).float().cuda().unsqueeze(1))/x_spt.shape[0]
            lossb = loss_fn(logits_b, torch.ones(x_qry.shape[0]).float().cuda().unsqueeze(1))/x_qry.shape[0]
            lossc = loss_fn(logits_c, torch.zeros(n_spt.shape[0]).float().cuda().unsqueeze(1))/n_spt.shape[0]
            lossr = self.loss_fn(logits_r, y_spt)
            '''lossa = F.cross_entropy(logits_a, torch.ones(x_spt.shape[0]).long().cuda())
            lossb = F.cross_entropy(logits_b, torch.ones(x_qry.shape[0]).long().cuda())
            lossc = F.cross_entropy(logits_c, torch.zeros(n_spt.shape[0]).long().cuda())'''
            grad_a = list(torch.autograd.grad(lossa+lossb+lossc+(3*lossr), fast_weights,allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    pdb.set_trace()
                    grad_a[g] = torch.zeros_like(s_weights[g])
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, fast_weights)))  
            with torch.no_grad():
                logits_q = net(x_qry, vars=fast_weights, bn_training=True, hook=hook)
                loss_q = self.loss_fn(logits_q, y_qry)
                corrects[k+1] += loss_q.item()
            del lossa, lossb, lossc, lossr, grad_a, logits_a, logits_b, logits_c, logits_r

            ## Update model w.r.t. location loss
            '''logits_a = net(x_spt, vars=fast_weights, bn_training=True, hook=7)
            loss = self.avg_loss(logits_a, y_spt)
            grad_a = list(torch.autograd.grad(loss, fast_weights[4:6],allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    grad_a[g] = torch.zeros_like(fast_weights[g])
            t = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, fast_weights[4:6])))
            fast_weights = fast_weights[:4] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, fast_weights[4:6]))) + fast_weights[6:]
            with torch.no_grad():
                logits_q = net(x_qry, vars=fast_weights, bn_training=True, hook=7)
                loss_q = self.avg_loss(logits_q, y_qry)
                corrects[k+1] += loss_q.item()'''
        cam_vals1 = net(x_qry,vars=fast_weights,bn_training=True,hook=hook-2,debug=debug)
        #del net,loss, loss_q, grad_a, logits_a
        del net

        accs = np.array(corrects) 
        pred = self.avg_pred(logits_q)
        return accs,[cam_vals1], pred

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

    def class_tune2(self, n_spt, x_spt, y_spt, x_qry, y_qry,debug=False):
        task_num = x_spt.size(0)
        train_corrects = [0 for _ in range(self.update_step_test)]
        loss_fn = nn.BCEWithLogitsLoss()
        hook = len(list(self.net.parameters()))-1

        net = deepcopy(self.net)
        s_weights = net.parameters()
        for m in range(self.update_step_test):
            losses_q = 0
            with torch.no_grad():
                logits_q = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
                loss_q = self.loss_fn(logits_q, y_spt)
                train_corrects[m] += loss_q.item()

            for k in range(self.update_step):
                ## Get classification loss
                logits_a = net(x_spt, vars=s_weights, bn_training=True)
                logits_b = net(x_qry, vars=s_weights, bn_training=True)
                logits_c = net(n_spt, vars=s_weights, bn_training=True)
                lossa = loss_fn(logits_a, torch.ones(x_spt.shape[0]).float().cuda().unsqueeze(1))/x_spt.shape[0]
                lossb = loss_fn(logits_b, torch.ones(x_qry.shape[0]).float().cuda().unsqueeze(1))/x_qry.shape[0]
                lossc = loss_fn(logits_c, torch.zeros(n_spt.shape[0]).float().cuda().unsqueeze(1))/n_spt.shape[0]

                ## Get location loss
                logits_r = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
                lossr = self.loss_fn(logits_r, y_spt)

                ## Update weights
                grad_a = list(torch.autograd.grad(lossa+lossb+lossc+(self.lmb*lossr), s_weights,allow_unused=True))
                for g in range(len(grad_a)):
                    if grad_a[g] is None:
                        pdb.set_trace()
                        grad_a[g] = torch.zeros_like(s_weights[g])
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, s_weights)))  
                del lossa, lossb, lossc, lossr, grad_a, logits_a, logits_b, logits_c, logits_r

            ## Get post-tuning losses for location objectives
            logits_q = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
            losses_q += self.loss_fn(logits_q, y_spt)
            #pred_q = self.avg_pred(logits_q)
            torch.cuda.empty_cache()

            ## Get total losses after all tasks
            total_loss =  losses_q / task_num #) + (losses_q/losses_s)
            grad_a = list(torch.autograd.grad(total_loss, s_weights,allow_unused=True))
            s_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad_a[:-2], s_weights[:-2]))) + s_weights[-2:]  
            loss = total_loss.item()
            del total_loss, losses_q, logits_q

        spt_logits = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
        spt_pred = self.avg_pred(spt_logits)
        loss = self.loss_fn(spt_logits, y_spt).item()
        qry_logits = net(x_qry, vars=s_weights, bn_training=True, hook=hook)
        cam_vals1 = net(x_spt,vars=s_weights,bn_training=True,hook=hook-2,debug=debug)
        cam_vals2 = net(x_qry,vars=s_weights,bn_training=True,hook=hook-2,debug=debug)
        pred = self.avg_pred(qry_logits)
        del qry_logits, spt_logits, s_weights
        return loss, [cam_vals2],  np.array(train_corrects)/task_num, pred
        #return loss, [torch.cat([cam_vals1,cam_vals2])],  np.array(train_corrects)/task_num, pred

    def class_tune3(self, n_spt, x_spt, y_spt, x_qry, y_qry,debug=False,sim=True):
        task_num = x_spt.size(0)
        test_corrects = [0 for _ in range(self.update_step_test + 1)]
        train_corrects = [0 for _ in range(self.update_step_test)]
        loss_fn = nn.BCEWithLogitsLoss()
        hook = len(list(self.net.parameters()))-1

        net = deepcopy(self.net)
        s_weights = net.parameters()
        step_optim = optim.SGD(s_weights, self.update_lr)
        grads = torch.zeros([x_qry.shape[0],196]).cuda()

        with torch.no_grad():
            logits_q = net(x_qry, vars=s_weights, bn_training=True, hook=hook)
            loss_q = self.loss_fn(logits_q, y_qry)
            test_corrects[0] += loss_q.item()

        for k in range(self.update_step_test):
            ## Get classification loss
            #class_in = torch.cat([x_spt, x_qry])
            logits_a = net(x_qry, vars=s_weights, bn_training=True, grad_hook=hook-5)
            #logits_b = net(x_qry, vars=s_weights, bn_training=True)
            logits_c = net(n_spt, vars=s_weights, bn_training=True)
            lossa = loss_fn(logits_a, torch.ones(x_qry.shape[0]).float().cuda().unsqueeze(1))/x_qry.shape[0]
            #lossb = loss_fn(logits_b, torch.ones(x_qry.shape[0]).float().cuda().unsqueeze(1))/x_qry.shape[0]
            lossc = loss_fn(logits_c, torch.zeros(n_spt.shape[0]).float().cuda().unsqueeze(1))/n_spt.shape[0]

            ## Get location loss
            logits_r = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
            lossr = self.loss_fn(logits_r, y_spt)
            train_corrects[k] += lossr.item()

            ## Update weights
            step_optim.zero_grad()
            step_loss = lossa + lossc #+ (self.lmb * lossr)
            step_loss.backward()
            pdb.set_trace()
            grads += net.grads
            step_optim.step()
            '''grad_a = list(torch.autograd.grad(lossa+lossb+lossc+(self.lmb*lossr), s_weights,,allow_unused=True))
            pdb.set_trace()
            #grad_a = list(torch.autograd.grad(lossa+lossb+lossc+(self.lmb*lossr), s_weights,allow_unused=True))
            for g in range(len(grad_a)):
                if grad_a[g] is None:
                    pdb.set_trace()
                    grad_a[g] = torch.zeros_like(s_weights[g])
            s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_a, s_weights)))  '''


            with torch.no_grad():
                logits_q = net(x_qry, vars=s_weights, bn_training=True, hook=hook)
                loss_q = self.loss_fn(logits_q, y_qry)
                test_corrects[k+1] += loss_q.item()
            del lossa, lossc, lossr, logits_a, logits_c, logits_r
            #del lossa, lossb, lossc, lossr, grad_a, logits_a, logits_b, logits_c, logits_r

        spt_logits = net(x_spt, vars=s_weights, bn_training=True, hook=hook)
        spt_pred = self.avg_pred(spt_logits)
        loss = self.loss_fn(spt_logits, y_spt).item()
        qry_logits = net(x_qry, vars=s_weights, bn_training=True, hook=hook)
        cam_vals1 = net(x_spt,vars=s_weights,bn_training=True,hook=hook-2,debug=debug)
        cam_vals2 = net(x_qry,vars=s_weights,bn_training=True,hook=hook-2,debug=debug)
        cam_vals3 = torch.mul(cam_vals2,F.relu(grads))
        pred = self.avg_pred(qry_logits)
        del qry_logits, spt_logits, s_weights
        if sim:
            return loss, [cam_vals2],  np.array(test_corrects), pred
        else:
            return loss, [cam_vals3],  np.array(train_corrects), pred

def main():
    pass


if __name__ == '__main__':
    main()
