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

    def adadelta_update(self, net, update_step, data_in, y_spt, y_qry, class_tgt, spt_idx, qry_idx):
        test_losses = [0 for _ in range(update_step + 1)]
        ft_train_losses = [0 for _ in range(update_step)]
        pt_train_losses = [0 for _ in range(update_step)]
        x_spt, x_qry = data_in
        #x_spt, x_qry, n_spt = data_in
        all_input = torch.cat(data_in)
        eps = 1e-6
        rho = self.update_lr

        with torch.no_grad():
            logits_q = net(all_input, vars=None)[spt_idx:qry_idx,2:]
            loss_q = F.mse_loss(logits_q, y_qry)
            test_losses[0] += loss_q.item()

        squares, deltas = [], []
        s_weights = deepcopy(list(net.parameters()))
        for p in s_weights:
            squares.append(torch.zeros_like(p))
            deltas.append(torch.zeros_like(p))

        for k in range(update_step):
            logits_r = net(all_input, vars=(None if k==0 else s_weights))[:spt_idx,2:]
            loss_r = F.mse_loss(logits_r, y_spt)
            pt_train_losses[k] += loss_r.item()
            grads = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights))) 
        
            for p in range(len(grads)):
                grad = grads[p]
                squares[p].mul_(rho).addcmul_(1-rho, grad, grad)
                std = squares[p].add_(eps).sqrt_()
                curr_delta = deltas[p].add(eps).sqrt_().div_(std).mul_(grad)
                if k == 0:
                    s_weights[p] = net.parameters()[p] - curr_delta
                else:
                    s_weights[p] = s_weights[p] - curr_delta
                deltas[p].mul_(rho).addcmul_(1-rho, curr_delta, curr_delta)

            with torch.no_grad():
                logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                loss_q = F.mse_loss(logits_q, y_qry)
                test_losses[k+1] += loss_q.item()

        logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
        loss_q = F.mse_loss(logits_q, y_qry)
        pdb.set_trace()
        return loss_q, [test_losses, ft_train_losses, pt_train_losses], s_weights

    def update(self, net, update_step, data_in, y_spt, y_qry, class_tgt, spt_idx, qry_idx):
        sig = nn.Sigmoid()
        #sw = deepcopy(net.parameters())
        #opt = optim.Adam(sw, lr=self.update_lr)
        test_losses = [0 for _ in range(update_step + 1)]
        ft_train_losses = [0 for _ in range(update_step)]
        pt_train_losses = [0 for _ in range(update_step)]
        x_spt, x_qry = data_in
        #x_spt, x_qry, n_spt = data_in
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

                '''opt.zero_grad()
                loss_r = F.mse_loss(logits_r, y_spt)
                loss_r.backward()
                opt.step()'''

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
            full_set = [x_spt[i],x_qry[i]] #,n_spt[i]]
            loss_q, losses, w = self.adadelta_update(self.net, self.update_step, full_set, y_spt[i], y_qry[i], tgt_set, idx_a, idx_b)
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
        test_losses = [0 for _ in range(self.update_step_test + 1)]
        ft_train_loss = [0 for _ in range(self.update_step_test)]
        pt_train_loss = [0 for _ in range(self.update_step_test)]
        idx_a = x_spt.shape[0]
        idx_b = idx_a + x_qry.shape[0]
        full_set = [x_spt,x_qry] #,n_spt]
        tgt_set = torch.cat([torch.ones(idx_b).long().cuda(), torch.zeros(n_spt.shape[0]).long().cuda()]) 

        test_loss, losses, w = self.adadelta_update(net, self.update_step_test, full_set, y_spt, y_qry, tgt_set, idx_a, idx_b)
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
