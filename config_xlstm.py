# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration flags to run the main script.
"""

import os  # 用于处理文件路径和文件操作
import pickle  # 用于序列化和反序列化对象，可能用于保存和加载模型或数据
import torch

from .blocks.mlstm.block import mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlockConfig
from .xlstm_block_stack import xLSTMBlockStackConfig
from .blocks.mlstm.layer import mLSTMLayerConfig
from .blocks.slstm.layer import sLSTMLayerConfig
from .components.feedforward import FeedForwardConfig
from .blocks.retnet.block import RetNetBlockConfig

class Config():
    retrain = False # 如果为True，表示将重新训练模型
    tb_log = False  # 是否启用 TensorBoard 日志记录如果为 True，训练过程中的日志将被记录，以便后期可视化
    device = torch.device("cuda")
    #     device = torch.device("cpu")

    max_epochs = 10  # 训练的最大周期数
    batch_size = 32  # 每次训练的批次大小
    n_samples = 16  # 样本数量

    init_seqlen = 30  # 输入船舶序列时，保证船舶轨迹的长度在36-120之间，并读取船舶轨迹前18个时间点作为输入
    max_seqlen = 120
    min_seqlen = 36


    dataset_name = "ct_dma"  # 指定使用的数据集名称
    model_type = "xlstm_retnet_kl"

    if dataset_name == "ct_dma":  # ==============================

        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and
        # dlon
        # 归一化并离散化（将这些值映射到格子编号内）
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        # 将上面的到的格子编号映射为向量，这些向量就来自于这下面的嵌入矩阵
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        # 设定纬度和经度的最小值和最大值，用于数据范围的限制
        lat_min = 55.5 
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13
    # 结果就是一个表示“这个时刻船的状态”的高维向量表示，可以送入 Transformer 模型来进行轨迹预测

    # ===========================================================================
    # Model and sampling flags
    mode = "pos"  # 指定模型的运行模式，如 pos 表示位置预测。#"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1",
    # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode = "pos_vicinity"  # 指定采样模式，这里为位置邻域（pos_vicinity）# "pos", "pos_vicinity" or "velo"
    top_k = 10  # 设置模型输出时返回的前 k 个最优预测。# int or None
    r_vicinity = 40  # 邻域半径，可能用于选择位置附近的数据点# int

    # LSTM 相关配置参数
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    lstm_input_size = n_embd 
    lstm_hidden_size = 256
    lstm_num_layers = 4
    lstm_output_size = full_size
    # Blur flags
    # ===================================================
    blur = True  # 是否启用模糊处理。模糊处理可能有助于模型的泛化能力。
    blur_learnable = False  # 是否让模糊处理过程成为可学习的。
    blur_loss_w = 1.0  # 模糊损失的权重
    blur_n = 2  # 模糊处理的次数。
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    # Data flags
    # ===================================================
    ROOT_DIR = os.path.dirname(__file__)
    datadir = os.path.join(ROOT_DIR, "data", dataset_name)
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
 
    # model parameters
    # ===================================================
    n_head = 8  # Transformer 模型中注意力头的数量
    n_layer = 8  # Transformer 模型的层数
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    # optimization parameters
    # ===================================================
    betas = (0.9, 0.95)  # 用于 Adam 优化器的两个动量参数
    grad_norm_clip = 1.0  # 梯度裁剪阈值，用于防止训练过程中的 梯度爆炸 设置最大梯度范数为 1.0，超出的部分将被缩小
    weight_decay = 0.1  # 权重衰减（L2正则化）是一种防止过拟合的方法 only applied on matmul weigh
    final_tokens = 260e9  # 到了训练中后期，使用 Cosine 衰减将学习率逐步降低到初始值的 10% (at what point we reach 10% of original LR)
    num_workers = 0  # 设置 PyTorch 的 DataLoader 读取数据时使用 4 个子线程for DataLoader
    # 单独设置每一层的基础配置
    mlstm_layer_cfg = mLSTMLayerConfig(
        embedding_dim=256,
        dropout=0.2,
        bias=False,
        context_length=10
    )
    slstm_layer_cfg = sLSTMLayerConfig(
        embedding_dim=256,
        dropout=0.2
    )
    feedforward_cfg = FeedForwardConfig(
        embedding_dim=256,
        dropout=0.2
    )

    # 封装成 BlockConfig
    mlstm_block_cfg = mLSTMBlockConfig(mlstm=mlstm_layer_cfg)
    slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=feedforward_cfg)
    # ★ 新增 RetNet 配置（示例）
    retnet_block_cfg = RetNetBlockConfig(
        embedding_dim=256,  # 必须与总模型 n_embd 对齐
        layers=1,  # 一般每个 Block 放 1
        heads=4,  # 需整除 embedding_dim
        ffn_size=1024,
        double_v_dim=False,
    )
    # 构造 xLSTMBlockStack 的配置
    xlstm_stack_cfg = xLSTMBlockStackConfig(
        mlstm_block=mlstm_block_cfg,
        slstm_block=slstm_block_cfg,
        retnet_block=retnet_block_cfg,
        context_length=10,
        num_blocks=5,
        embedding_dim=256,
        add_post_blocks_norm=True,
        bias=False,
        dropout=0.2,
        slstm_at=[1],  # 第 2、3 层用 sLSTM
        retnet_at=[3,4]     # 第 0 层用 RetNet
    # 实际顺序：RetNet(0) -> mLSTM(1) -> sLSTM(2) -> sLSTM(3)
    )
    # ========== 关键参数修改 ==========
    depth = 5
    dropout = 0.2
    xlstm_config = xlstm_stack_cfg

    resid_pdrop = 0.5      # 从 0.5 改为 0.2
    embd_pdrop = 0.3       # 从 0.3 改为 0.1
    attn_pdrop = 0.1
    learning_rate = 1e-4   # 从 1e-4 改为 3e-5
    lr_decay = False
    warmup_tokens = 512 * 20
    # KL 正则化相关超参数
    # ======================
    kl_lambda = 0.75  # KL 散度损失权重
    kl_sigma_lat = 2.0  # 纬度高斯邻域宽度
    kl_sigma_lon = 2.0  # 经度高斯邻域宽度
    kl_sigma_sog = 1.5 # 航速高斯邻域宽度
    kl_sigma_cog = 1.5  # 航向高斯邻域宽度

 #========== 新增：生成架构描述的函数 ==========
    @staticmethod
    def get_architecture_string(slstm_at, retnet_at, num_blocks):
        
        block_types = ['M'] * num_blocks  # 默认全是 mLSTM
        
        # 标记 sLSTM 的位置
        for idx in slstm_at:
            if idx < num_blocks:
                block_types[idx] = 'S'
        
        # 标记 RetNet 的位置
        for idx in retnet_at:
            if idx < num_blocks:
                block_types[idx] = 'R'
        
        # 拼接成字符串
        arch_string = "".join(block_types)
        return arch_string
        # 生成架构描述
    arch_string = get_architecture_string(
        xlstm_stack_cfg.slstm_at,
        xlstm_stack_cfg.retnet_at,
        xlstm_stack_cfg.num_blocks
    )
    
    
    

        # 简洁的文件名格式：时间戳_架构_KL参数
    filename = f"{model_type}_{arch_string}_kl{kl_lambda}"

    # 例如：1101_1425_xlstm_RMSS_kl0.1

    savedir = "./outputs/results/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model_3.pt")
