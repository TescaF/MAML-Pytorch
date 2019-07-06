#  MAML-Pytorch
PyTorch implementation of the supervised learning experiments from the paper:
[Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

Adapted from [MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch).

# Requirements
- python: 3.x
- Pytorch: 0.4+

# Execution
Batch baseline:
python regression_test.py --func_type="cat" --k_spt=10 --k_qry=10 --k_model=5 --grasps=1 --tuned_layers=6 --update_step_test=30 --task_num=1000 --iter_qry=0 --split=0.7 --split_cat=0

Random iterative baseline:
python regression_test.py --func_type="cat" --k_spt=10 --k_qry=10 --k_model=5 --grasps=1 --tuned_layers=6 --update_step_test=30 --task_num=1000 --al_method="random" --split=0.7 --split_cat=0

Active Learning, input-space metric:
python regression_test.py --func_type="cat" --k_spt=10 --k_qry=10 --k_model=5 --grasps=1 --tuned_layers=6 --update_step_test=30 --task_num=1000 --al_method="input_space" --split=0.7 --split_cat=0

Active Learning, output-space metric:
python regression_test.py --func_type="cat" --k_spt=10 --k_qry=10 --k_model=5 --grasps=1 --tuned_layers=6 --update_step_test=30 --task_num=1000 --al_method="k_centers" --split=0.7 --split_cat=0

Active Learning, output-space batch:
python regression_test.py --func_type="cat" --k_spt=10 --k_qry=10 --k_model=5 --grasps=1 --tuned_layers=6 --update_step_test=30 --task_num=1000 --al_method="k_centers" --split=0.7 --split_cat=0 --batch_sz=3

