import  pdb
import  numpy as np
import  math
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    learner import Learner
from    copy import deepcopy

class Meta(nn.Module):
    def __init__(self, args, config, loss_fn=F.cross_entropy):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.feat_lr = args.feat_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.loss_fn = loss_fn
        self.pos_matrix = None
        self.pos_only = (args.pos_only == 1)

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
        #pred = self.avg_pred(logits)
        #return torch.mean(torch.sum((pred-y)**2.0,dim=1))
        pred = []
        r = int(math.sqrt(logits.shape[-1]))
        c = float(r)/2.0
        if self.pos_matrix is None:
            self.pos_matrix = torch.stack(torch.meshgrid([torch.arange(r),torch.arange(r)]),-1).float().cuda()
            self.dist_matrix = torch.zeros((r*2,r*2)).float().cuda()
            for i in range(r*2):
                for j in range(r*2):
                    self.dist_matrix[i,j] = math.sqrt((i-r)**2 + (j-r)**2)
        losses = []
        for i in range(logits.shape[0]):
            s = F.softmax(logits[i],0).reshape(r,r)
            xr = int(np.clip(torch.floor((y[i,0] * c) + c).item(), 0, r))
            yr = int(np.clip(torch.floor((y[i,1] * c) + c).item(), 0, r))
            start_idx = [r - xr, r - yr]
            end_idx = [start_idx[0] + r, start_idx[1] + r]
            dists = self.dist_matrix[start_idx[0]:end_idx[0],start_idx[1]:end_idx[1]]
            losses.append(torch.sum(dists * s))
        return np.sum(np.array(losses))/logits.shape[0]

    def update(self, net, update_step, data_in, y_spt, y_qry, class_tgt, spt_idx, qry_idx):
        sig = nn.Sigmoid()
        hook = len(list(net.config))-1
        test_losses = [0 for _ in range(update_step + 1)]
        ft_train_losses = [0 for _ in range(update_step)]
        pt_train_losses = [0 for _ in range(update_step)]
        x_spt, x_qry, n_spt = data_in
        all_input = torch.cat(data_in)

        with torch.no_grad():
            logits_q = net(all_input, vars=None)[spt_idx:qry_idx,2:]
            loss_q = F.mse_loss(logits_q, y_qry)
            test_losses[0] += loss_q.item()

        for k in range(update_step):
            if self.pos_only:
                logits_r = net(all_input, vars=(None if k==0 else s_weights))[:spt_idx,2:]
                loss_r = F.mse_loss(logits_r, y_spt)
                pt_train_losses[k] += loss_r.item()
                grad = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
            else:
                ''' g7:
                logits_a = net(all_input, vars=(None if k==0 else s_weights))[:,:2]
                loss_a = F.cross_entropy(logits_a, class_tgt) 
                ft_train_losses[k] += loss_a.item()
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

                logits_r = net(all_input, vars=s_weights)[:spt_idx,2:]
                loss_r = F.mse_loss(logits_r, y_spt)
                grad = list(torch.autograd.grad(loss_r, s_weights)) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights)))

                pt_train_losses[k] += loss_r.item()
                with torch.no_grad():
                    logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                    loss_q = F.mse_loss(logits_q, y_qry)
                    test_losses[k+1] += loss_q.item()
                continue'''

                logits_a = net(all_input, vars=(None if k==0 else s_weights))[:,:2]
                loss_a = F.cross_entropy(logits_a, class_tgt) 
                ft_train_losses[k] += loss_a.item()
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                # g9 is weighting update lr for features x 10
                s_weights = list(map(lambda p: p[1] - self.feat_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

                with torch.no_grad():
                    part_input = net(all_input, vars=s_weights,hook=len(net.config)-1).detach()
                logits_r = F.linear(part_input, s_weights[-2], s_weights[-1])[:spt_idx,2:]
                loss_r = F.mse_loss(logits_r, y_spt)
                grad = list(torch.autograd.grad(loss_r, s_weights,allow_unused=True)) 
                s_weights[-1] = s_weights[-1] - self.update_lr * grad[-1]
                s_weights[-2] = s_weights[-2] - self.update_lr * grad[-2]

                pt_train_losses[k] += loss_r.item()
                with torch.no_grad():
                    logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                    loss_q = F.mse_loss(logits_q, y_qry)
                    test_losses[k+1] += loss_q.item()
                continue

            '''if mode == "g9":
                logits_a = net(all_input, vars=(None if k==0 else s_weights))[:,:2]
                loss_a = F.cross_entropy(logits_a, class_tgt) 
                ft_train_losses[k] += loss_a.item()
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - 10.0 * self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

                with torch.no_grad():
                    part_input = net(all_input, vars=s_weights,hook=len(net.config)-1).detach()
                w_input = part_input[:spt_idx] * s_weights[-2][1,:].detach()
                logits_r = F.linear(w_input, s_weights[-2], s_weights[-1])[:spt_idx,2:]
                loss_r = F.mse_loss(logits_r, y_spt)
                grad = list(torch.autograd.grad(loss_r, s_weights,allow_unused=True)) 
                s_weights[-1] = s_weights[-1] - self.update_lr * grad[-1]
                s_weights[-2] = s_weights[-2] - self.update_lr * grad[-2]

                pt_train_losses[k] += loss_r.item()
                with torch.no_grad():
                    logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                    loss_q = F.mse_loss(logits_q, y_qry)
                    test_losses[k+1] += loss_q.item()
                continue'''


            with torch.no_grad():
                logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                loss_q = F.mse_loss(logits_q, y_qry)
                test_losses[k+1] += loss_q.item()

        logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
        loss_q = F.mse_loss(logits_q, y_qry)
        return loss_q, [test_losses, ft_train_losses, pt_train_losses], s_weights

    def forward(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        task_num = x_spt.size(0)
        losses_q = 0
        test_losses = [0 for _ in range(self.update_step + 1)]
        ft_train_loss = [0 for _ in range(self.update_step)]
        pt_train_loss = [0 for _ in range(self.update_step)]
        idx_a = x_spt.shape[1]
        idx_b = idx_a + x_qry.shape[1]
        tgt_set = torch.cat([torch.ones(idx_b).long().cuda(), torch.zeros(n_spt.shape[1]).long().cuda()]) 

        for i in range(task_num):
            full_set = [x_spt[i],x_qry[i],n_spt[i]]
            loss_q, losses, w = self.update(self.net, self.update_step, full_set, y_spt[i], y_qry[i], tgt_set, idx_a, idx_b)
            test_losses = [test_losses[j] + losses[0][j] for j in range(len(test_losses))]
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

    def tune(self, n_spt, x_spt, y_spt, x_qry, y_qry):
        net = deepcopy(self.net)
        opt = optim.Adam(net.parameters(), lr=self.meta_lr)
        hook = len(list(net.config))-1
        test_losses = [0 for _ in range(self.update_step_test + 1)]
        ft_train_loss = [0 for _ in range(self.update_step_test)]
        pt_train_loss = [0 for _ in range(self.update_step_test)]
        idx_a = x_spt.shape[0]
        idx_b = idx_a + x_qry.shape[0]
        full_set = [x_spt,x_qry,n_spt]
        tgt_set = torch.cat([torch.ones(idx_b).long().cuda(), torch.zeros(n_spt.shape[0]).long().cuda()]) 

        test_loss, losses, w = self.update(net, self.update_step_test, full_set, y_spt, y_qry, tgt_set, idx_a, idx_b)
        test_losses = losses[0]
        ft_train_loss = losses[1]
        pt_train_loss = losses[2]
        '''for i in range(self.update_step_test):
            test_loss, losses, w = self.update(net, self.update_step_test, full_set, y_spt, y_qry, tgt_set, idx_a, idx_b)
            opt.zero_grad()
            test_loss.backward()
            test_losses = [test_losses[j] + losses[0][j] for j in range(len(test_losses))]
            ft_train_loss = [ft_train_loss[j] + losses[1][j] for j in range(len(ft_train_loss))]
            pt_train_loss = [pt_train_loss[j] + losses[2][j] for j in range(len(pt_train_loss))]'''

        logits = net(torch.cat(full_set), vars=w)[:,2:]
        spt_logits = logits[:idx_a]
        qry_logits = logits[idx_a:idx_b]

        #cam_vals = net(x_qry,vars=w,hook=len(net.config)-4) * w[-2][1].expand((x_qry.shape[0],196))
        #cam_vals = torch.mm(net(x_qry,vars=w,hook=len(net.config)-4), w[-4])
        #out_vals = net(x_qry,vars=w,hook=len(net.config)-3)
        return [np.array(test_losses), np.array(pt_train_loss), np.array(ft_train_loss)], qry_logits

def main():
    pass

if __name__ == '__main__':
    main()
