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

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,  # 包含多个字典的列表，每个字典表示一条船舶的轨迹。每条轨迹包含字段 "mmsi"（船舶的 MMSI 唯一标识符）和 "traj"（一个包含位置、速度、航向和时间戳的矩阵
                 max_seqlen=96,  # 最大序列长度，默认值为 96。超过该长度的轨迹将被截断，不足的部分则通过填充（用 0）补全
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)  # 定义数据集的长度，即返回数据集中轨迹样本的个数

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]  # 获取数据集中第 idx 个样本（即某条船舶的轨迹）
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999  # 获取该轨迹的前四列（纬度、经度、速度、航向）。对于数据中的最大值（1.0），将其替换为 0.9999，以防止可能的错误值
        seqlen = min(len(m_v), self.max_seqlen)  # 得到轨迹有效长度（实际轨迹长度与 max_seqlen 之间取最小值，保证在96以内）
        seq = np.zeros((self.max_seqlen, 4))  # 建一个大小为 max_seqlen 的全零矩阵，并将实际轨迹数据填充到该矩阵中。这样确保每个轨迹的长度一致
        seq[:seqlen, :] = m_v[:seqlen,
                          :]  # 拷贝实际轨迹数据到前 seqlen 行（如果轨迹本身长度 < max_seqlen，那只填前 seqlen 行，其它位置仍是 0（补零）如果轨迹太长，只会取前 seqlen = max_seqlen 行（截断））
        seq = torch.tensor(seq, dtype=torch.float32)  # 转成 PyTorch 的 Tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, mask, seqlen, mmsi, time_start


class AISDataset_grad(Dataset):  # 这个类是 AISDataset 的一个改进版本，用于返回轨迹点的 位置 + 位移（梯度）信息
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """

    def __init__(self,
                 l_data,
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """

        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])  # 构建一个形如 [0.04, 0.04] 的 numpy 数组，后面归一化差分用
        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog
        m_v[m_v == 1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4))
        # lat and lon
        seq[:seqlen, :2] = m_v[:seqlen, :2]  # 前两列放的是标准化后的 [lat, lon]
        # dlat and dlon
        dpos = (m_v[1:, :2] - m_v[:-1, :2] + self.dpos_max) / (
                    2 * self.dpos_max)  # dpos = (delta + 最大值) / 2最大值 用来把位移差归一化到 [0,1] 区间
        dpos = np.concatenate((dpos[:1, :], dpos),
                              axis=0)  # 因为 dpos 只有 T-1 行（相邻差值），少了一帧，所以复制第一行补回去(始轨迹有 T 帧数据，dpos 会只有 T-1 行，因为每行代表相邻两帧之间的变化,为了使得每条轨迹的形状一致)
        dpos[dpos >= 1] = 0.9999
        dpos[dpos <= 0] = 0.0  # 规避归一化误差，防止超过 [0,1) 的范围
        seq[:seqlen, 2:] = dpos[:seqlen, :2]  # 轨迹的前两列是 lat/lon，后两列是 dlat/dlon（速度信息的替代）

        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)  # 阻止模型在 self-attention 机制中把注意力分配给“补零的位置”,以为船舶静止不动
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, mask, seqlen, mmsi, time_start  # 更直接体现“运动变化”有利于建模运动趋势，避免角度不连续问题