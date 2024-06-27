import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
# from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from act_utils import load_data  # data functions
from act_utils import sample_box_pose, sample_insertion_pose  # robot functions
from act_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from transformers import RobertaTokenizer
from sim_env import BOX_POSE

import IPython

e = IPython.embed

os.environ['WANDB_API_KEY'] = '806fe2f74471b3d0f6e60edb8846c140ca1e48b9'

def main(args):
    set_seed(1)# 设置随机种子以保证结果可重现
    # 解析命令行参数
    # command line parameters
    is_eval = args['eval']  # 是否为评估模式的布尔标志
    ckpt_dir = args['ckpt_dir']  # 保存/加载checkpoint的目录
    policy_class = args['policy_class']  # 使用的策略类
    onscreen_render = args['onscreen_render']  # 是否进行屏幕渲染的标志
    task_name = args['task_name']  # 任务名称
    batch_size_train = args['batch_size']  # 训练批大小
    batch_size_val = args['batch_size']  # 验证批大小
    num_epochs = args['num_epochs']  # 训练的总周期数

    use_waypoint = args["use_waypoint"]  # 是否使用航点
    constant_waypoint = args["constant_waypoint"]  # 持续航点的设置
    if use_waypoint:
        print("Using waypoint")
    if constant_waypoint is not None:
        print(f"Constant waypoint: {constant_waypoint}")

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    # is_sim = True  # hardcode to True to avoid finding constants from aloha
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    # 从任务配置中获取相关参数
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters 定义模型的架构和超参数，包括学习率、网络结构、层数等
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    lr_obs_backbone = 1e-5  # 主干网络的学习率 1e-6
    obs_encoder = 'obs_encoder'  # 使用的主干网络类型
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            'obs_encoder': obs_encoder,
            'lr_obs_backbone': lr_obs_backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
    }
    '''
    1. transfer the red cube to the other arm with human-assisted
    1. transfer the red cube to the other arm with script-controlled
    2. insert the red peg into the blue socket with human-assisted
    2. insert the red peg into the blue socket with script-controlled
    3. place the yellow doll in the white basket
    '''
    print(f"Please input your hoping task:")
    task_name = input()
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval:
        wandb.init(project="awe", name=expr_name)
        wandb.config.update(config)
    ''' 
       如果`is_eval`为True，那么代码将进入评估模式。在这种模式下，它将加载名为`policy_best.ckpt`的模型检查点，
       并使用`eval_bc`函数对其进行评估。`eval_bc`函数的返回值是成功率和平均回报，这些值将被存储在`results`列表中
       然后，代码将遍历`results`列表，并打印每个检查点的名称、成功率和平均回报
   '''
    print('is_eval', is_eval)
    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, task_name, save_episode=True)
            wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()
    '''
    如果`is_eval`为False，那么代码将进入训练模式。在这种模式下，它将调用`load_data`函数来加载训练和验证数据
    `load_data`函数的返回值是训练数据加载器、验证数据加载器、统计数据和一个布尔值，该布尔值表示是否为模拟任务
    '''
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_waypoint,
        constant_waypoint,
    )

    # save dataset stats
    # 保存数据集统计信息
    '''
    首先，它检查是否存在一个名为`ckpt_dir`的目录，如果不存在，它将创建这个目录。然后，它将数据集的统计信息`stats`保存到一个名为`dataset_stats.pkl`的文件中。这些统计信息可能包括数据集的一些特性，如平均值、标准差等
    '''
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    # 训练并获取最佳检查点信息
    '''
     调用`train_bc`函数来训练模型。`train_bc`函数接受训练数据加载器、验证数据加载器和配置字典作为参数，并返回最佳检查点的信息，包括最佳周期、最小验证损失和最佳状态字典
     train_bc：训练行为克隆BC模型
     train_dataloader：训练数据的数据加载器，用于从训练集中获取批次的数据
     val_dataloader：验证数据的数据加载器，用于从验证集中获取批次的数据
     config：包含训练配置信息的字典
    '''
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, task_name, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")
    wandb.finish()


# 通过ACTPolicy获取policy
'''
make_policy的定义：policy = ACTPolicy(policy_config)
根据指定的policy_class(策略类别，目前支持两种类型："ACT"和"CNNMLP")，和policy_config(策略配置)创建一个策略模型对象
'''


def make_policy(policy_class, policy_config):
    # 可以看到policy调用了act - main / policy.py中定义的ACTPolicy，那ACTPolicy则是基于CVAE实现的
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy



'''
make_optimizer用于创建策略模型的优化器(optimizer)，并返回创建的优化器对象。
优化器的作用是根据策略模型的损失函数来更新模型的参数，以使损失函数尽量减小
'''


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

'''
`get_image`函数的作用是从给定的时间步对象中获取指定摄像头的图像，并将其处理为适合模型输入的格式
get_image的作用是获取一个时间步(ts)的图像数据。函数接受两个参数：ts和camera_names
`ts`是一个时间步对象，它包含了当前时间步的观察结果。`camera_names`是一个列表，包含了需要获取图像的摄像头的名称
函数首先创建一个空列表`curr_images`，用于存储从每个摄像头获取的图像
它遍历`camera_names`列表，对于每个摄像头名称，它从`ts.observation['images']`中获取对应的图像，
并使用`rearrange`函数将图像的维度从'高度 宽度 通道数'重新排列为'通道数 高度 宽度'
然后将重新排列后的图像添加到`curr_images`列表中
'''
def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)  # 将处理后的图像添加到列表中
    # 它使用`np.stack`函数将`curr_images`列表中的所有图像堆叠在一起，形成一个新的numpy数组`curr_image`
    curr_image = np.stack(curr_images, axis=0)  # 将图像列表堆叠成数组
    # 将`curr_image`数组的数据类型转换为torch张量，并将其值归一化到0 - 1之间，然后将其转移到GPU上，并增加一个新的维度
    # 最后，函数返回处理后的图像张量`curr_image`(包含时间步图像数据的PyTorch张量，这个图像数据可以被用于输入到神经网络模型中进行处理)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

'''
# eval_bc：评估一个行为克隆(behavior cloning)模型
# 传参与配置信息,该函数用于评估给定的策略。它接受两个参数：`config`和`ckpt_name`
config`是一个字典，包含了评估过程中需要的各种配置信息，如策略类名称、摄像头名称、任务名称等
`ckpt_name`是一个字符串，表示要加载的策略的检查点文件的名称
函数首先从`config`中提取出各种配置信息，并设置随机种子以确保结果的可复现性
'''


def eval_bc(config, ckpt_name, task_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # 它加载策略的检查点文件，并将策略模型转移到GPU上，并将其设置为评估模式
    # load policy and stats 加载策略和统计信息
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    # 定义预处理和后处理函数
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment函数加载环境
    # 如果`real_robot`为True，那么它将加载真实机器人的环境；否则，它将加载模拟环境
    print('real_robot', real_robot)
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha
        print('make_real_env start')
        env = make_real_env(init_node=True)
        env_max_reward = 0
        print('make_real_env end')
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
    # 设置查询频率和时间聚合参数
    query_frequency = policy_config["num_queries"]
    print('temporal_agg', temporal_agg)
    if temporal_agg:
        query_frequency = 1
        print('query_frequency', query_frequency)
        num_queries = policy_config["num_queries"]
    # if have base
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY
    # 设置最大时间步数
    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    '''
        开始评估BC：两大循环——大循环50回合、每个回合下的小循环跑完时间步长
        先是大循环，对于每个回合，它首先重置环境
        再是小循环，即内层循环，对于每个时间步，它先获取当前的观察结果，然后查询策略以获取动作，
        最后执行动作并获取奖励，最后将奖励添加到`rewards`列表中.
        首先开始大循环，它首先初始化一些变量，比如一个模拟环境的回合数（`num_rollouts`）为50，
        并初始化了两个空列表：`episode_returns`和`highest_rewards`，用于存储每个回合的回报和最高奖励
    '''
    num_rollouts = 50  # 回放循环，学它个50回合
    episode_returns = []  # 初始化结果列表
    highest_rewards = []
    '''
        然后，它使用一个for循环来进行每个回合的模拟，且在每个回合开始时，它会根据任务名称（`task_name`）来设置任务的初始状态
        如果任务名称中包含'sim_transfer_cube'，那么它会调用`sample_box_pose`函数来随机生成一个立方体的位置和姿态，并将其赋值给全局变量`BOX_POSE[0]`，该变量表示盒子的位置或姿态信息
        如果任务名称中包含'sim_insertion'，那么它会调用`sample_insertion_pose`函数来随机生成一个插入任务的初始状态，包括插入物（peg）和插入孔（socket）的位置和姿态，并将其赋值给`BOX_POSE[0]`。
        这些初始状态将在模拟环境重置时被使用.
        最后，它调用`env.reset`函数来重置模拟环境，并将返回的时间步对象赋值给`ts`
    '''
    print('learning start:', num_rollouts)
    for rollout_id in range(num_rollouts):
        # if real_robot:
        # e()
        # pass
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()  # 重置环境
        '''
            代码检查`onscreen_render`是否为True
            如果为True，那么它将创建一个matplotlib的子图，并在子图上显示模拟环境的渲染结果。这里使用了`env._physics.render`方法来获取模拟环境
            的渲染图像，其中`height`和`width`参数指定了渲染图像的大小，`camera_id`参数指定了用于渲染的摄像头
            然后，它调用`plt.ion`方法来开启交互模式，这样就可以在模拟过程中实时更新显示的图像
        '''
        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()
        '''
            检查`temporal_agg`是否为True
            如果为True，那么它将创建一个全零的torch张量`all_time_actions`，用于存储所有时间步的动作
        '''
        ### evaluation loop
        if temporal_agg:
            '''
                这个张量的形状为`[max_timesteps, max_timesteps+num_queries, state_dim]`，其中
                `max_timesteps`是每个回合的最大时间步数
                `num_queries`是查询的数量
                `state_dim`是状态的维度.之后这个张量被转移到了GPU上
            '''
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
        '''
            创建了一个全零的torch张量`qpos_history`，用于存储每个时间步的机器人关节位置（`qpos`），这个张量的形状为`(1, max_timesteps, state_dim)`
        '''
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        # 创建了四个空列表：`image_list`、`qpos_list`、`target_qpos_list`和`rewards`
        image_list = []  # for visualization 用于可视化的图像列表`用于存储每个时间步的图像
        qpos_list = []  # 用于存储每个时间步的机器人关节位置
        target_qpos_list = []  # 用于存储每个时间步的目标机器人关节位置
        rewards = []  # `用于存储每个时间步的奖励
        '''
        小循环：获取每个时间步的观察结果.
        内层循环了：对于每个时间步，它先获取当前的观察结果(相当于获取每个时间步的观察结果，包括图像和机器人的关节位置，并将这些信息存储起来)，
        然后查询策略以获取动作且位于一个`torch.inference_mode()`上下文管理器中，这意味着在这个上下文中的所有PyTorch操作都不会跟踪梯度，
        这对于推理和测试非常有用，因为它可以减少内存使用并提高计算速度
        '''
        # 在不计算梯度的模式下执行
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT 更新屏幕渲染和等待时间
                '''
                    在每个时间步中，如果`onscreen_render`为True，那么它会通过`env._physics.render`方法获取模拟环境的渲染图像(其中`height`和`width`参数指定了渲染图像的大小，`camera_id`参数指定了用于渲染的摄像头)
                    并使用`plt_img.set_data(image)`方法来更新显示的图像(它接受一个图像数组作为参数，并将其设置为图像的新数据)
                    然后，它调用`plt.pause(DT)`方法来暂停一段时间
                '''
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                # 从`ts`（可能是一个时间步对象）中获取观察结果`obs`。然后，它检查`obs`中是否包含`images`键
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])  # 如果包含，那么它将`obs['images']`添加到`image_list`中；
                else:
                    image_list.append({"main": obs["image"]})  # 否则，它将一个包含`obs['image']`的字典添加到`image_list`中
                qpos_numpy = np.array(obs["qpos"])  # 从`obs`中获取机器人的关节位置`qpos`，并将其转换为numpy数组
                qpos = pre_process(qpos_numpy)  # 使用之前定义的`pre_process`函数对`qpos`进行预处理，这个函数会将`qpos`标准化
                # 将标准化后的`qpos`转换为torch张量，并将其转移到GPU上。这个张量的形状被增加了一个新的维度，这是通过`unsqueeze(0)`方法实现的
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # 将处理后的`qpos`存储到`qpos_history`张量的对应位置。这里的`t`是当前的时间步，所以`qpos_history[:, t]`表示的是在第`t`个时间步的`qpos`
                # qpos_history[:, t] = qpos
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, task_name)
                    '''
                        如果`temporal_agg`为True(相当于要做指数加权)，那么它会将所有的动作存储到`all_time_actions`张量的对应位置，
                        并从中获取当前步骤的动作`actions_for_curr_step`。然后，它检查`actions_for_curr_step`中的所有元素是否都不为0，
                        如果是，那么它会保留这些动作
                    '''
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        '''
                        创建一个指数权重`exp_weights`，并将其转换为torch张量。最后，它使用这些权重对动作进行加权平均，得到`raw_action`
                        '''
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        # 如果`temporal_agg`为False(相当于不做指数加权)，那么它会直接从`all_actions`中获取当前步骤的动作`raw_action`
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions  对动作的进一步处理
                raw_action = raw_action.squeeze(0).cpu().numpy()
                print('raw_action', raw_action)
                action = post_process(raw_action)
                target_qpos = action
                print('target_qpos', target_qpos)
                # print('base_action', base_action)
                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        print('real_robot', real_robot)
        if real_robot:
            print('real_robot', real_robot)
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        # 计算当前回合的总奖励，之后保存图像数据,在内层循环结束后，它计算当前回合的总奖励，并将其添加到`episode_returns`列表中
        # 计算回报和奖励
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        # 将每次评估的图像数据保存为视频
        # if save_episode:
        #     save_videos(
        #         image_list,
        #         DT,
        #         video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
        #     )
    # 计算成功率和平均回报
    # 计算成功率，即最高奖励的次数与环境最大奖励相等的比率
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)  # 计算平均回报
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    # 遍历奖励范围，计算每个奖励范围内的成功率
    for r in range(env_max_reward + 1):
        # 统计最高奖励大于等于 r 的次数
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        # 计算成功率
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return
    # `eval_bc`函数的作用是评估给定的策略在指定任务上的性能


# 前向传播以生成模型的输出
'''
data：包含输入数据的元组，其中包括图像数据、关节位置数据、动作数据以及填充标志
policy：行为克隆模型
函数的主要步骤如下：
将输入数据转移到GPU上，以便在GPU上进行计算。
调用行为克隆模型的前向传播方法(policy)，将关节位置数据、图像数据、动作数据和填充标志传递给模型
返回模型的输出，这可能是模型对动作数据的预测结果
'''


def forward_pass(data, task_name, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    sample_texts = [task_name] * 8
    return policy(qpos_data, image_data, sample_texts, action_data, is_pad)  # TODO remove None (8,14)


def train_bc(train_dataloader, val_dataloader, task_name, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 1:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            loading_status = policy.load_state_dict(torch.load(ckpt_path))
            print(loading_status)
        else:
            print("Not loading checkpoint")
            latest_idx = 0
    else:
        latest_idx = 0

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(latest_idx, num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        # 首先进行验证。将模型设置为评估模式，并对验证数据集进行遍历
        # 对于每一批数据，都会进行一次前向传播，并将结果添加到 `epoch_dicts` 列表中
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, task_name, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            # 如果这个轮次的验证损失小于之前的最小验证损失，就更新最小验证损失，并保存当前的模型状态
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        # print(validation_history)
        # for k in list(validation_history[0].keys()):
        #     validation_history[0][f'val_{k}'] = validation_history.pop(k)
        wandb.log({"val_loss": epoch_val_loss}, step=epoch)
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)
        # evaluation
        if (epoch > 0) and (epoch % 600 == 0):
            # first save then eval
            ckpt_name = f'policy_epoch_{epoch-100}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            # torch.save(policy.serialize(), ckpt_path)
            # 注释掉下面两行, 使用实际数据来训练策略 , num_rollouts=10
            success, _ = eval_bc(config, ckpt_name, task_name, save_episode=True)
            wandb.log({'success': success}, step=epoch)
        # training
        # 训练：在训练集上进行模型的训练，计算损失并执行反向传播来更新模型的权重
        # 将模型设置为训练模式，并对训练数据集进行遍历。对于每一批数据，都会进行一次前向传播，然后进行反向传播，并使用优化器更新模型的参数
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, task_name, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        e = epoch - latest_idx
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        wandb.log({'train_loss': epoch_train_loss}, step=epoch)
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)
        # 每隔一定周期，保存当前模型的权重和绘制训练曲线图
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)
    # 保存最佳模型的权重和绘制训练曲线图
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # for waypoints
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument(
        "--constant_waypoint",
        action="store",
        type=int,
        help="constant_waypoint",
        required=False,
    )
    # for vit/causal/cross
    main(vars(parser.parse_args()))
