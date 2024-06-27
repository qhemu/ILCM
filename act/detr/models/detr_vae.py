# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from ..obsmodel.vision.timm_obs_encoder import TimmObsEncoder, CLIPModel
import numpy as np
import torchvision.transforms as transforms
from transformers import CLIPTextModel
import IPython
from transformers import RobertaTokenizer, RobertaModel
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class FeatureMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
        # self.linear2 = nn.Linear(output_dim,output_dim)

    def forward(self, x):
        return self.activation(self.linear(x))
        # return self.linear2(self.activation(self.linear(x)))
        # return self.linear(x)

class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(self, backbones, obs_encoder, text_encoder, transformer,
                 encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.obs_encoder = obs_encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.text_encoder = text_encoder
        self.text_featuremapper = FeatureMapper(768, 768)
        self.visual_featuremapper = FeatureMapper(768, 768)
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        num_channels = 512
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                # backbones[0].num_channels, hidden_dim, kernel_size=1
                num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        self.text_fc = nn.Linear(512, hidden_dim)  # 假设图像和文本的特征都是512维，然后合并为1024维

    # encode
    def forward(self, qpos, image, sample_texts, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token action(8,50,14)
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim) (8,50,512)
            qpos_embed = self.encoder_action_proj(qpos)  # (bs, hidden_dim)  (8,14)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim) (8,1,512)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim) 52,8,512
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim) 8,1,512
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only (8,768)
            latent_info = self.latent_proj(encoder_output)  # (8,64)
            mu = latent_info[:, : self.latent_dim]  # (8,32)
            logvar = latent_info[:, self.latent_dim :]  # (8,32)
            latent_sample = reparametrize(mu, logvar)  # (8,32)
            latent_input = self.latent_out_proj(latent_sample)  # (8, 768)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            # all_cam_features = []
            # all_cam_pos = []
            # image(8,3代表３个摄像头,3,480,640)
            images = dict()
            for cam_id, cam_name in enumerate(self.camera_names):
                images[cam_name] = image[:, cam_id, None, :, :, :]  # image(8,1, 3,480,640)
                # features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                # features = features[0]  # take the last layer feature
                # pos = pos[0]
                # all_cam_features.append(self.input_proj(features))
                # all_cam_pos.append(pos)
            all_vitcam_features, all_vitcam_pos = self.obs_encoder(images)
            # proprioception features 本体感受特征
            proprio_input = self.input_proj_robot_state(qpos)  # (8,768)
            # # print(proprio_input.size())
            with torch.no_grad():  # 禁用梯度计算
                sample_text = self.tokenizer(sample_texts, return_tensors='pt').to('cuda')
                text_inputs = self.text_encoder(**sample_text.to('cuda'))

            text_input = text_inputs.last_hidden_state  # (8,15,768)
            vit_src = torch.cat(all_vitcam_features, axis=2)  # (8,197,768)
            vit_pos = torch.cat(all_vitcam_pos, axis=2)  # (1,197,512)
            batch_size = vit_src.shape[0]
            if text_input.shape[0] != batch_size:
                text_input = text_input[:batch_size]
            text_input = self.text_featuremapper(text_input)  # (8, 14, 768)
            vit_src = self.visual_featuremapper(vit_src)  # (8, 197, 768)
            # mapped_text_features = text_input
            # mapped_visual_features = vit_src  # (8, 197, 768)
            # print('mapped_text_features', mapped_text_features.shape)
            # print('mapped_visual_features', mapped_visual_features.shape)
            # # 扩展文本特征到与视觉特征相同的长度
            # extended_text_features = torch.nn.functional.interpolate(
            #     text_input.permute(0, 2, 1), size=197, mode='linear', align_corners=False
            # ).permute(0, 2, 1)  # (8, 197, 768)
            # print("extended_text_features", extended_text_features.shape)
                        # print('attended_text_features', attended_text_features.shape)
            # # 视觉特征作为查询，文本特征作为键和值
            # print('attended_visual_features', attended_visual_features.shape)
            # fused_features = vit_src
            # features-src+PosEmb-pos+joints-proprio_input+z-latent_input+self.additional_pos_embed.weight?
            hs = self.transformer(vit_src, text_input, None, self.query_embed.weight, vit_pos, latent_input, proprio_input,
                                      self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5),
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(
                input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2
            )
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


# build_encoder函数负责创建编码器。编码器输出一个隐变量，是一个32维的高斯分布，
# 实现代码位于detr/models/detr_vae.py的DETRVAE类的encode函数.
def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)
    # for _ in args.camera_names:
    #     backbone = build_backbone(args)
    #     backbones.append(backbone)
    # ADD VIT
    obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='vit_base_patch16_clip_224.openai',
        pretrained=True,
        frozen=False,
        global_pool='',
        # transforms=[{'type':'RandomCrop','ratio':0.95},{'type':'Resize','ratio':0.95}],
        transforms=[{'type': 'RandomCrop', 'ratio': 0.95},
                    transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.6, 1.4), saturation=(0.5, 1.5),
                                           hue=(-0.08, 0.08))],
        feature_aggregation='attention_pool_2d',
        # replace BatchNorm with GroupNorm
        use_group_norm=True,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm=True,
        position_encording='sinusoidal',
        camera_names=args.camera_names,
        args=args
    )  # vit?
    obs_feature_dim = np.prod(obs_encoder.output_shape())
    print('obs_feature_dim:', obs_feature_dim)
    # text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    # text_encoder = CLIPModel(text_encoder=text_model).to('cuda')
    # 加载RoBERTa模型
    text_encoder = RobertaModel.from_pretrained('roberta-base').to('cuda')

    # 创建vae中的解码器/生成器，输入
    # 至于build_transformer在act - main / detr / models / transformer.py被定义如下
    transformer = build_transformer(args)

    encoder = build_encoder(args)
    model = DETRVAE(
        backbones,
        obs_encoder,
        text_encoder,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # for _ in args.camera_names:
    #     backbone = build_backbone(args)
    #     backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
