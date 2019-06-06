import pdb
import random
import  torch, os
import  numpy as np
from sinusoid import Sinusoid
import  argparse

from meta import Meta

def main(args):

    save_path = os.getcwd() + '/data/model_batchsz' + str(args.k_spt) + '_stepsz' + str(args.update_lr) + '.pt'

    torch.cuda.synchronize()
    torch.manual_seed(222)
    torch.cuda.synchronize()
    torch.cuda.manual_seed_all(222)
    torch.cuda.synchronize()
    np.random.seed(222)

    print(args)

    #device = torch.device('cpu')
    device = torch.device('cuda')
    torch.cuda.synchronize()

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

    mod = Meta(args, config).to(device)
    mod.load_state_dict(torch.load(save_path))
    mod.eval()

    db_train = Sinusoid(
                       batchsz=args.task_num,
                       k_shot=args.k_spt,
                       k_qry=args.k_qry)

    all_accs = []
    batch_x, batch_y, amp, phase = db_train.next()
    queries = None

    for i in range(args.task_num):
        if i % 10 == 0:
            print("Task " + str(i))
        accs = []
        c_idx = np.array(range(batch_x[i].shape[0]))
        q_set = torch.from_numpy(batch_x[i]).float().to(device), torch.from_numpy(batch_y[i]).float().to(device)
            
        while len(c_idx) > 0:
            if args.iter_qry == 1:
                s_idx = al_method_random(batch_x[i], c_idx)
                inputa = [batch_x[i, s_idx, :]]
                labela = [batch_y[i, s_idx, :]]
                if queries is None:
                    qin = np.concatenate([inputa, inputa]) #, axis=1)     
                    ql = np.concatenate([labela, labela]) #, axis=1)     
                    queries = [inputa, labela, s_idx]
                else:
                    qin = np.concatenate([queries[0], inputa]) #, axis=1)     
                    ql = np.concatenate([queries[1], labela]) #, axis=1)     
                    queries = [qin, ql, s_idx]
                c_idx = c_idx[c_idx != s_idx]

            else:
                qin = batch_x[i, :args.k_spt, :]
                ql = batch_y[i, :args.k_spt, :]
                q_set = torch.from_numpy(batch_x[i, args.k_spt:, :]).float().to(device), torch.from_numpy(batch_y[i, args.k_spt:, :]).float().to(device)
                c_idx = []
            qin, ql = torch.from_numpy(qin).float().to(device), torch.from_numpy(ql).float().to(device)
            test_acc = mod.finetunning(qin, ql, q_set[0], q_set[1])
            accs.append( test_acc[-1] )
        if args.iter_qry == 1:
            all_accs.append(accs)
        else:
            all_accs.append(test_acc)

    accs_val = np.array(all_accs).mean(axis=0).astype(np.float16)
    print('Test acc:', accs_val)

def al_method_random(data, avail_idx):
    return avail_idx[random.randint(0, len(avail_idx) - 1)]

'''def al_method_k_centers(data, avail_idx):
        used_idx = np.array(range(data.shape[1]))
        used_idx = c_idx[c_idx != s_idx]
                        dists = tf.constant([FLAGS.update_batch_size * [np.inf]]) #[0.0,0.0]) #zeros([0,0]) #tf.constant([[]]) #tf.zeros([FLAGS.update_batch_size, FLAGS.update_batch_size])
                        pred_outputc = self.forward(inputa, fast_weights, reuse=reuse)
                        for ki in range(FLAGS.update_batch_size):
                            dists_iter = tf.constant([[]])
                            est_i = tf.gather(pred_outputc, ki)[0]
                            for kj in range(FLAGS.update_batch_size):
                                est_j = tf.gather(pred_outputc, kj)[0]
                                d = tf.cond(avail_tensor[ki], lambda: [tf.math.squared_difference(est_i, est_j)], lambda: tf.constant([np.inf]))
                                dists_iter = tf.concat([dists_iter, d], 1)
                            dists = tf.concat([dists, dists_iter],0)
                        best_idx = tf.math.argmin(tf.reduce_sum(dists[1:,:], 1))
                        best_idx = tf.Print(best_idx, [best_idx], "selected idx:")
                        avail_tensor = tf.logical_and(avail_tensor, tf.gather(masks_tensor, best_idx)[0])
                        next_query = tf.gather(inputa, best_idx)[0]
                        next_label = tf.gather(labela, best_idx)[0]
'''
    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=100)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--iter_qry', type=int, help='learn from examples one-at-a-time', default=1)

    args = argparser.parse_args()

    main(args)
