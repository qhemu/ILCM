import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


def relabel_waypoints(arr, waypoint_indices):
    start_idx = 0
    for key_idx in waypoint_indices:
        # Replace the items between the start index and the key index with the key item
        arr[start_idx:key_idx] = arr[key_idx]
        start_idx = key_idx
    return arr


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=False,
        constant_waypoint=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_waypoint = use_waypoint
        self.constant_waypoint = constant_waypoint
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            # print('is_sim:', is_sim)
            if '/base_action' in root:
                base_action = root['/base_action'][()]
                # base_action = preprocess_base_action(base_action)
                # action = np.concatenate([root['/action'][()], base_action], axis=-1)
            else:
                if "/action/action" in root:
                    action = root["/action/action"][()]
                else:
                    action = root["/action"][()]
                # dummy_base_action = np.zeros([action.shape[0], 2])
                # action = np.concatenate([action, dummy_base_action], axis=-1)
            original_action_shape = action.shape  # (400,16)
            # original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

            if self.use_waypoint and self.constant_waypoint is None:
                waypoints = root["/waypoints"][()]

        if self.use_waypoint:
            # constant waypoints
            if self.constant_waypoint is not None:
                assert self.constant_waypoint > 0
                waypoints = np.arange(1, action_len, self.constant_waypoint)
                if len(waypoints) == 0:
                    waypoints = np.array([action_len - 1])
                elif waypoints[-1] != action_len - 1:
                    waypoints = np.append(waypoints, action_len - 1)
            # auto waypoints
            else:
                waypoints = waypoints - start_ts
                waypoints = waypoints[waypoints >= 0]
                waypoints = waypoints[waypoints < action_len]
                waypoints = np.append(waypoints, action_len - 1)
                waypoints = np.unique(waypoints)
                waypoints = waypoints.astype(np.int32)
            # waypoints替换action
            action = relabel_waypoints(action, waypoints)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float  没有augmentation增强了
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        # with h5py.File(dataset_path, 'r') as hdf:
        #     print(list(hdf.keys()))  # This will print all top-level groups and datasets
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            if "/action/action" in root:
                action = root["/action/action"][()]
            else:
                action = root["/action"][()]
            # print('action',action)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    # all_qpos_data = torch.stack(all_qpos_data)
    # all_action_data = torch.stack(all_action_data)
    # all_action_data = all_action_data
    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # 50个（400，16）
    all_action_data = torch.cat(all_action_data, dim=0)  # (20000,14)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10)  # clipping
    eps = 0.0001
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze()-eps,
        "qpos_std": qpos_std.numpy().squeeze()+eps,
        "example_qpos": qpos,
    }

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    use_waypoint=False,
    constant_waypoint=None,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=use_waypoint,
        constant_waypoint=constant_waypoint,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=use_waypoint,
        constant_waypoint=constant_waypoint,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
