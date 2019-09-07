#!/bin/bash
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=12 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex12.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=11 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex11.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=10 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex10.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=9 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex9.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=8 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex8.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=7 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex7.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=6 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex6.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=5 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex5.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=4 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex4.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=3 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex3.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=2 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex2.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=1 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex1.txt
python -u train_affordance.py --k_spt=1 --k_qry=1 --update_lr=0.1 --meta_lr=0.001 --epoch=10000 --task_num=10 --exclude=0 | tee /home/tesca/software/v3-maml/maml-pytorch/data/k1_ex0.txt
