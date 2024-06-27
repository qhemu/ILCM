import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from ..common.module_attr_mixin import ModuleAttrMixin
from ..vision.vit_position_encoding import build_position_encoding
# from vit_position_encoding import build_position_encoding
from ..common.pytorch_util import replace_submodules
from transformers import CLIPTextModel, CLIPTokenizer


logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            camera_names: list,
            args,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
        ):
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )

        if frozen:  # false
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = (480, 640)
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0]['type'] == 'RandomCrop'
            ratio = transforms[0]['ratio']
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=224, antialias=True),
                # add more
                torchvision.transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)
        shape = tuple([3, 224, 224])
        for key in camera_names:
            key_shape_map[key] = shape
            rgb_keys.append(key)  # 'camera0_rgb'

            this_model = model if share_rgb_model else copy.deepcopy(model)
            key_model_map[key] = this_model

            this_transform = transform
            key_transform_map[key] = this_transform

        feature_map_shape = [x // downsample_ratio for x in image_shape]  # 0-7 1-7 [7, 7]
            
        rgb_keys = sorted(rgb_keys)  # 'camera0_rgb'
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)  # 'camera0_rgb'
        print('no low_dim_keys keys:', low_dim_keys)  # ０－'robot0_eef_pos'　１－'robot0_eef_rot_axis_angle' 2-'robot0_eef_rot_axis_angle_wrt_start' 3-'robot0_gripper_width'
        self.camera_names = camera_names
        self.model_name = model_name  # 'vit_base_patch16_clip_224.openai'
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model  # false
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        self.position_embedding = build_position_encoding(args)
        if model_name.startswith('vit'):
            if self.feature_aggregation == 'all_tokens':
                pass
            elif self.feature_aggregation is not None:  # use this
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':  # no use
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):  # use this
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)
        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512
        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
    def forward(self, obs_dict):
        features = list()
        pos = []
        for key in self.camera_names:  # 'camera0_rgb'
            img = obs_dict[key]  # 64.2.3.224.224
            B, T = img.shape[:2]  # 64 2  T 在这个上下文中很可能指的是时间步数（time steps）或者序列长度，这通常用于处理连续的数据（如视频帧、时间序列数据等）。
            assert B == img.shape[0]
            img = img.reshape(B*T, *img.shape[2:])  # 128.3.224.224
            img = self.key_transform_map[key](img)  # 2.3.224.224
            raw_feature = self.key_model_map[key](img)  # 2,197,768  (1,197,768)
            features.append(raw_feature)  # (64,1536)
            pos.append(self.position_embedding(raw_feature).to(raw_feature.dtype))
            # features = self.aggregate_feature(raw_feature)  # (128,768) 暂时不调用这一步，使用所有的token，而不是只有cls的token,否则无法计算位置信息
            # assert len(feature.shape) == 2 and feature.shape[0] == B * T
            # features.append(feature.reshape(B, -1))


        # result = torch.cat(features, dim=-1)

        return features, pos


    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        for key in self.camera_names:
            shape = tuple([3, 480, 640])
            this_obs = torch.zeros(
                (1, 1) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _ = self.forward(example_obs_dict)
        result = torch.cat(example_output, dim=-1)
        return result.shape  # (1,2304)

class CLIPModel(nn.Module):
    def __init__(self, text_encoder):
        super(CLIPModel, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = text_encoder

    def forward(self, text_inputs):
        # 处理文本输入
        text_inputs = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
        text_inputs = text_inputs.to('cuda')
        # print(text_inputs)
        # exit()
        text_features = self.text_encoder(**text_inputs).pooler_output
        return text_features


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
