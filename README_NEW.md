# ACT: Action Chunking with Transformers

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation
```console
for waypoint:
git clone git@github.com:lucys0/awe.git
cd awe
conda create -n vitawe_venv python=3.9
conda activate vitawe_venv
pip install -e .
# install robomimic
pip install -e robomimic/
# install robosuite
pip install -e robosuite/

pip uninstall cython
pip install cython==0.29.21
pip install free-mujoco-py
还不行就：
pip install modern_robotics
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

命令：
python example/act_waypoint.py --err_threshold=0.01 --save_waypoints 
展示：--plot_3d --end_idx=0




for act:
conda env update -f act/conda_env.yaml
pip install gym
conda install pytorch torchvision -c pytorch

命令：
python act/imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir data/outputs/act_ckpt/norlvitsim_transfer_cube_scripted_waypoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 768 --batch_size 8 --dim_feedforward 3200 --num_epochs 18000 --lr 1e-5 --seed 0 --temporal_agg --use_waypoint
python act/imitate_episodes.py --task_name sim_transfer_cube_human --ckpt_dir data/outputs/act_ckpt/norlvitsim_transfer_cube_human_waypoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 768 --batch_size 8 --dim_feedforward 3200 --num_epochs 18000 --lr 1e-5 --seed 0 --temporal_agg --use_waypoint
python act/imitate_episodes.py --task_name sim_insertion_human --ckpt_dir data/outputs/act_ckpt/norlvitsim_insertion_human_waypoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 768 --batch_size 8 --dim_feedforward 3200 --num_epochs 18000 --lr 1e-5 --seed 0 --temporal_agg --use_waypoint
python act/imitate_episodes.py --task_name sim_insertion_scripted --ckpt_dir data/outputs/act_ckpt/norlvitsim_insertion_scripted_waypoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 768 --batch_size 8 --dim_feedforward 3200 --num_epochs 18000 --lr 1e-5 --seed 0 --temporal_agg --use_waypoint
python act/imitate_episodes.py --task_name doll_basket_session --ckpt_dir data/outputs/act_ckpt/norlvitdoll_basket_session_waypoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 768 --batch_size 8 --dim_feedforward 3200 --num_epochs 18000 --lr 1e-5 --seed 0 --temporal_agg --use_waypoint
```

1. transfer the red cube to the other arm with human-assisted
1. transfer the red cube to the other arm with script-controlled
2. insert the red peg into the blue socket with human-assisted
2. insert the red peg into the blue socket with script-controlled
3. place the yellow doll in the white basket