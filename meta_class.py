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
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.loss_fn = loss_fn
        self.pos_matrix = None

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
        mode = "g9"

        if mode == "g4":
            with torch.no_grad():
                n_idx = int(data_in.shape[0]/2)
                logits_q = sig(torch.sum(net(torch.cat([data_in[spt_idx:qry_idx],data_in[n_idx + spt_idx:]]), vars=None),dim=1)-1)
                loss_q = F.cross_entropy(torch.stack([1-logits_q,logits_q],dim=1), torch.cat([class_tgt[spt_idx:qry_idx],class_tgt[n_idx + spt_idx:]]))
                test_losses[0] += loss_q.item()
        elif mode == "g6" or mode == "g7" or mode == "g8" or mode == "g9":
            with torch.no_grad():
                logits_q = net(all_input, vars=None)[spt_idx:qry_idx,2:]
                loss_q = F.mse_loss(logits_q, y_qry)
                test_losses[0] += loss_q.item()
        else:
            with torch.no_grad():
                #logits_q = net(x_qry, vars=None)
                #loss_q = self.loss_fn(logits_q, y_qry)
                ## v4
                #with torch.no_grad():
                #    class_w = list(net.parameters())[-2][1].expand((all_input.shape[0],196)).detach()
                #logits_q = net(all_input,vars=None,hook=len(net.config)-3) * class_w
                ## end v4
                logits_q = net(all_input, vars=None,hook=len(net.config)-3)
                loss_q = self.loss_fn(logits_q[spt_idx:qry_idx], y_qry)
                test_losses[0] += loss_q.item()

        l = []
        for k in range(update_step):
            #logits_a = sig(torch.sum(net(data_in, vars=(None if k==0 else s_weights)),dim=1)-1)
            #loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), class_tgt)
            #ft_train_losses[k] += loss_a.item()

            if mode == "g0":
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

            if mode == "g1":
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
                with torch.no_grad():
                    input_r = net(data_in, vars=s_weights,hook=hook)[:spt_idx]
                logits_r = F.linear(input_r, s_weights[-2], s_weights[-1])
                loss_r = self.loss_fn(logits_r, y_spt)
                grad_r = list(torch.autograd.grad(loss_r, s_weights, allow_unused=True))[-2:]
                s_weights = s_weights[:-2] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_r, s_weights[-2:])))
                pt_train_losses[k] += loss_r.item()
                #if k == 4:
                #    print(str([torch.norm(g).item() for g in grad]))

            if mode == "g2":
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                logits_r = net(data_in, vars=(None if k==0 else s_weights))[:spt_idx]
                loss_r = self.loss_fn(logits_r, y_spt)
                grad_r = list(torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights), allow_unused=True))[-2:]
                grad[-2] += grad_r[-2]
                grad[-1] += grad_r[-1]
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
                pt_train_losses[k] += loss_r.item()

            if mode == "g3":
                logits_r = net(x_spt, vars=(None if k==0 else s_weights))
                loss_r = self.loss_fn(logits_r, y_spt)
                l.append(round(loss_r.item(),4))
                grad_r = torch.autograd.grad(loss_r, (net.parameters() if k==0 else s_weights))
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_r, (net.parameters() if k==0 else s_weights))))
                pt_train_losses[k] += loss_r.item()

            if mode == "g4":
                n_idx = int(data_in.shape[0]/2)
                logits_a = sig(torch.sum(net(torch.cat([data_in[:spt_idx],data_in[n_idx:n_idx + spt_idx]]), vars=(None if k==0 else s_weights)),dim=1)-1)
                loss_a = F.cross_entropy(torch.stack([1-logits_a,logits_a],dim=1), torch.cat([class_tgt[:spt_idx],class_tgt[n_idx:n_idx + spt_idx]]))
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
                ft_train_losses[k] += loss_a.item()
                with torch.no_grad():
                    logits_q = sig(torch.sum(net(torch.cat([data_in[spt_idx:qry_idx],data_in[n_idx + spt_idx:]]), vars=(None if k==0 else s_weights)),dim=1)-1)
                    loss_q = F.cross_entropy(torch.stack([1-logits_q,logits_q],dim=1), torch.cat([class_tgt[spt_idx:qry_idx],class_tgt[n_idx + spt_idx:]]))
                    test_losses[k+1] += loss_q.item()
                continue

            if mode == "g5":
                logits_a = net(all_input, vars=(None if k==0 else s_weights))
                loss_a = F.cross_entropy(logits_a, class_tgt)
                l.append(round(loss_a.item(),4))
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
                ft_train_losses[k] += loss_a.item()

                ## v4
                #with torch.no_grad():
                #    class_w = s_weights[-2][1].expand((all_input.shape[0],196)).detach()
                #logits_r = net(all_input,vars=s_weights,hook=len(net.config)-3) * class_w
                ## end v4
                with torch.no_grad():
                    input_r = net(all_input, vars=s_weights,hook=len(net.config)-4)[:spt_idx]
                logits_r = F.linear(input_r, s_weights[-6], s_weights[-5])
                loss_r = self.loss_fn(logits_r, y_spt)
                grad_r = list(torch.autograd.grad(loss_r, s_weights, allow_unused=True))
                pdb.set_trace()
                s_weights = s_weights[:-2] + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_r, s_weights[-2:])))
                '''logits_r = net(all_input, vars=s_weights,hook=len(net.config)-3)
                loss_r = self.loss_fn(logits_r[:spt_idx], y_spt)
                grad = list(torch.autograd.grad(loss_r, s_weights,allow_unused=True))
                grad = [g for g in grad if g is not None]
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, s_weights[:len(grad)]))) + s_weights[len(grad):]'''
                pt_train_losses[k] += loss_r.item()

            if mode == "g6":
                logits_a = net(all_input, vars=(None if k==0 else s_weights))[:,:2]
                logits_r = net(all_input, vars=(None if k==0 else s_weights))[:spt_idx,2:]
                loss_a = F.cross_entropy(logits_a, class_tgt) * 2.0
                loss_r = F.mse_loss(logits_r, y_spt)
                grad = list(torch.autograd.grad(loss_a+loss_r, (net.parameters() if k==0 else s_weights))) 
                s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))
                ft_train_losses[k] += loss_a.item()
                pt_train_losses[k] += loss_r.item()
                with torch.no_grad():
                    logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
                    loss_q = F.mse_loss(logits_q, y_qry)
                    test_losses[k+1] += loss_q.item()
                continue
            if mode == "g7":
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
                continue

            if mode == "g8":
                logits_a = net(all_input, vars=(None if k==0 else s_weights))[:,:2]
                loss_a = F.cross_entropy(logits_a, class_tgt) 
                ft_train_losses[k] += loss_a.item()
                grad = list(torch.autograd.grad(loss_a, (net.parameters() if k==0 else s_weights))) 
                # g9 is weighting update lr for features x 10
                s_weights = list(map(lambda p: p[1] - 10.0 * self.update_lr * p[0], zip(grad, (net.parameters() if k==0 else s_weights))))

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

            if mode == "g9":
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
                continue


            with torch.no_grad():
                #logits_q = net(x_qry, vars=s_weights)
                ## v4
                #with torch.no_grad():
                #    class_w = s_weights[-2][1].expand((all_input.shape[0],196)).detach()
                #logits_q = net(all_input,vars=s_weights,hook=len(net.config)-3) * class_w
                ## end v4
                logits_q = net(all_input, vars=s_weights,hook=len(net.config)-3)
                loss_q = self.loss_fn(logits_q[spt_idx:qry_idx], y_qry)
                test_losses[k+1] += loss_q.item()
                #logits_r = net(all_input, vars=s_weights,hook=len(net.config)-3)
                #loss_r = self.loss_fn(logits_r[:spt_idx], y_spt)

        if mode == "g4":
            logits_q = sig(torch.sum(net(torch.cat([data_in[spt_idx:qry_idx],data_in[n_idx + spt_idx:]]), vars=s_weights),dim=1)-1)
            loss_q = F.cross_entropy(torch.stack([1-logits_q,logits_q],dim=1), torch.cat([class_tgt[spt_idx:qry_idx],class_tgt[n_idx + spt_idx:]]))
        elif mode == "g6" or mode == "g7" or mode == "g8" or mode == "g9":
            logits_q = net(all_input, vars=s_weights)[spt_idx:qry_idx,2:]
            loss_q = F.mse_loss(logits_q, y_qry)
        else:
            #logits_q = net(x_qry, vars=s_weights)
            #loss_q = self.loss_fn(logits_q, y_qry)
            ## v4
            #with torch.no_grad():
            #    class_w = s_weights[-2][1].expand((all_input.shape[0],196)).detach()
            #logits_q = net(all_input,vars=s_weights,hook=len(net.config)-3) * class_w
            ## end v4
            logits_q = net(all_input, vars=s_weights,hook=len(net.config)-3)
            loss_q = self.loss_fn(logits_q[spt_idx:qry_idx], y_qry)
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

        for i in range(self.update_step_test):
            test_loss, losses, w = self.update(net, self.update_step_test, full_set, y_spt, y_qry, tgt_set, idx_a, idx_b)
            opt.zero_grad()
            test_loss.backward()
        test_losses = [test_losses[j] + losses[0][j] for j in range(len(test_losses))]
        ft_train_loss = [ft_train_loss[j] + losses[1][j] for j in range(len(ft_train_loss))]
        pt_train_loss = [pt_train_loss[j] + losses[2][j] for j in range(len(pt_train_loss))]

        spt_logits = net(x_spt, vars=w,hook=len(net.config)-3)
        qry_logits = net(x_qry, vars=w,hook=len(net.config)-3)
        train_loss = self.loss_fn(spt_logits, y_spt).item()
        qry_pred = self.avg_pred(qry_logits)

        cam_vals = net(x_qry,vars=w,hook=len(net.config)-4) * w[-2][1].expand((x_qry.shape[0],196))
        #cam_vals = torch.mm(net(x_qry,vars=w,hook=len(net.config)-4), w[-4])
        out_vals = net(x_qry,vars=w,hook=len(net.config)-3)
        return [train_loss,test_loss], [cam_vals,out_vals],  [np.array(test_losses),np.array(pt_train_loss),np.array(ft_train_loss)], qry_pred

def main():
    pass

if __name__ == '__main__':
    main()
