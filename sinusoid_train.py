import pdb
import  torch, os
import  numpy as np
from sinusoid import Sinusoid
import  argparse

from    meta import Meta

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    dim_hidden = [40,40]
    dim_input = 1
    dim_output = 1
    config = [
        ('linear', [dim_hidden[0], dim_input]),
        #('linear', [dim_input, dim_hidden[0]]),
        ('relu', [True]),
        ('bn', [dim_hidden[0]])]

    for i in range(1, len(dim_hidden)):
        config += [
            ('linear', [dim_hidden[i], dim_hidden[i-1]]),
            ('relu', [True]),
            ('bn', [dim_hidden[i]])]

    config += [
        ('linear', [dim_output, dim_hidden[-1]]),
        ('relu', [True]),
        ('bn', [dim_output])]

    device = torch.device('cpu')
    #device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = Sinusoid(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry)

    # epoch: number of training batches
    for step in range(args.epoch):

        batch_x, batch_y, amp, phase = db_train.next()
        #inputa = batch_x[:, :args.k_spt, :]
        inputa = batch_x[:, :args.k_spt, :]
        inputb = batch_x[:,args.k_spt:, :]
        labela = batch_y[:, :args.k_spt, :]
        labelb = batch_y[:,args.k_spt:, :]

        #x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                     torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 500 == 0:
            accs = []
            for _ in range(1000//args.task_num):
                # test
                batch_x, batch_y, amp, phase = db_train.next()
                inputa = batch_x[:, :args.k_spt, :]
                inputb = batch_x[:,args.k_spt:, :]
                labela = batch_y[:, :args.k_spt, :]
                labelb = batch_y[:,args.k_spt:, :]

                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                             torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    #x_spt_one = x_spt_one.unsqueeze(0)
                    #y_spt_one = y_spt_one.unsqueeze(0)
                    #x_qry_one = x_qry_one.unsqueeze(0).unsqueeze(0)
                    #y_qry_one = y_qry_one.unsqueeze(0).unsqueeze(0)
                    #pdb.set_trace()
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=2)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
