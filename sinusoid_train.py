import pdb
import  torch, os
import  numpy as np
from sinusoid import Sinusoid
import  argparse

from    meta import Meta

def main(args):

    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '.pt'

    torch.cuda.synchronize()
    torch.manual_seed(222)
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(222)
    torch.cuda.synchronize()
    np.random.seed(222)

    print(args)

    dim_hidden = [40,40]
    dim_input = 1
    dim_output = 1
    config = [
        ('fc', [dim_hidden[0], dim_input]),
        ('relu', [True])]#,
        #('bn', [dim_hidden[0]])]

    for i in range(1, len(dim_hidden)):
        config += [
            ('fc', [dim_hidden[i], dim_hidden[i-1]]),
            ('relu', [True])] #,
            #('bn', [dim_hidden[i]])]

    config += [
        ('fc', [dim_output, dim_hidden[-1]])] #,
        #('relu', [True])] #,
        #('bn', [dim_output])]

    #device = torch.device('cpu')
    device = torch.device('cuda')
    torch.cuda.synchronize()
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
        inputa = batch_x[:, :args.k_spt, :]
        labela = batch_y[:, :args.k_spt, :]
        if args.k_spt == 1:
            inputa = np.array(inputa) #np.concatenate([inputa, inputa], axis=1)     
            labela = np.array(labela) #np.concatenate([labela, labela], axis=1)     
        labelb = batch_y[:,args.k_spt:, :]
        inputb = batch_x[:,args.k_spt:, :]

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                     torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)
        if step % 500 == 0:
            preloss = ('%.10f'%accs[0].item())
            postloss = ('%.10f'%accs[-1].item())
            print('Step ' + str(step) + ': ' + preloss + ', ' + postloss)

        '''if step % 500 == 0:
            accs = []
            for _ in range(1000//args.task_num):
                batch_x, batch_y, amp, phase = db_train.next()
                inputa = batch_x[:, :args.k_spt, :]
                inputb = batch_x[:,args.k_spt:, :]
                labela = batch_y[:, :args.k_spt, :]
                labelb = batch_y[:,args.k_spt:, :]
                if args.k_spt == 1:
                    inputa = np.concatenate([inputa, inputa], axis=1)     
                    labela = np.concatenate([labela, labela], axis=1)     

                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(inputa).float().to(device), torch.from_numpy(labela).float().to(device), \
                                             torch.from_numpy(inputb).float().to(device), torch.from_numpy(labelb).float().to(device)

                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)'''
    torch.save(maml.state_dict(), save_path)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=70000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=25)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
