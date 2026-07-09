#!/usr/bin/env python
# coding: utf-8
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

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb
from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
# 其他需要的结构也可以用 from blocks.mlstm.cell import ... 之类

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from . import models, trainers, datasets, utils

from .config_xlstm import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# 设定随机种子以保证实验可重复
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================创建保存目录 + 日志系统初始化
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05#用于过滤不动的船
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]#从配置文件中取出 train/valid/test 的文件名
    Data, aisdatasets, aisdls = {}, {}, {}
    #加载并处理每一类数据（train、valid、test）
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:#滤除静止/无效数据
            try:#如果某条轨迹一开始是静止的（速度 SOG < 0.05），就从第一个 动起来的时间点 开始截取。
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:#如果始终没动过，就会保留空轨迹，后面会被过滤掉
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")#保留 干净且有足够长度 的轨迹用于训练/验证/测试
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):#根据配置选择使用 AISDataset 还是 AISDataset_grad（后者包含位置梯度信息）
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False#如果当前阶段是 "test"，数据加载时不打乱（shuffle=False），因为测试阶段通常需要保持数据的顺序
        else:
            shuffle = True#其他阶段（如训练和验证）需要打乱数据（shuffle=True），这样可以避免训练时的顺序偏差
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)#创建 PyTorch 的 DataLoader，用于批量训练
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen#设置训练总 token 数量

    ## Model
    # ===============================
    model = models.xLSTMShipTrajectory(cf, partition_model=None)

    ## Trainer实例化：创建训练器
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training启动训练
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model训练过程中验证集损失最小的模型参数文件
    model.load_state_dict(torch.load(cf.ckpt_path,weights_only=True))
   #这两项是为了将 [0, 1) 归一化范围的 AIS 数据转换为 实际地理坐标（纬度、经度）
    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 10#设置采样长度

    model.eval()#模型设置为评估模式
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():#禁用梯度计算
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks#用 Haversine 距离计算输入和预测坐标之间的地理距离
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))#用于存储所有批次的最小误差，便于后续的分析和汇总
            l_mean_errors.append(error_ens.mean(dim=-1))#用于存储每个批次的平均误差，便于后续的分析和汇总
            l_masks.append(masks[:, cf.init_seqlen:])#用于存储所有批次中有效部分的掩码数据

    l_min = [x.values for x in l_min_errors]#得到一个包含所有批次最小误差值的列表
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)#这一步计算的是每个时间步的平均预测误差
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================绘制预测误差随时间变化的曲线图
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6-1
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12-1
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18-1
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 24 - 1  # 24 表示四个个小时
    plt.plot(4, pred_errors[timestep], "o")
    plt.plot([4, 4], [0, pred_errors[timestep]], "r")
    plt.plot([0, 4], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(4.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 30 - 1  # 30 表示五个个小时
    plt.plot(5, pred_errors[timestep], "o")
    plt.plot([5, 5], [0, pred_errors[timestep]], "r")
    plt.plot([0, 5], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(5.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 36 - 1  # 36 表示六个个小时
    plt.plot(6, pred_errors[timestep], "o")
    plt.plot([6, 6], [0, pred_errors[timestep]], "r")
    plt.plot([0, 6], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(6.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 42 - 1  # 42 表示七个个小时
    plt.plot(7, pred_errors[timestep], "o")
    plt.plot([7, 7], [0, pred_errors[timestep]], "r")
    plt.plot([0, 7], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(7.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 48 - 1  # 48 表示八个个小时
    plt.plot(8, pred_errors[timestep], "o")
    plt.plot([8, 8], [0, pred_errors[timestep]], "r")
    plt.plot([0, 8], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(8.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 54 - 1  # 54 表示九个个小时
    plt.plot(9, pred_errors[timestep], "o")
    plt.plot([9, 9], [0, pred_errors[timestep]], "r")
    plt.plot([0, 9], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(9.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 60 - 1  # 60 表示十个个小时
    plt.plot(10, pred_errors[timestep], "o")
    plt.plot([10, 10], [0, pred_errors[timestep]], "r")
    plt.plot([0, 10], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(10.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error_test.png")

