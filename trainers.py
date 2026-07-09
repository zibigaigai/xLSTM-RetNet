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

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging
from torch.utils.tensorboard import SummaryWriter
#tb = SummaryWriter(log_dir="./tf_logs")# 设置日志目录，TensorBoard 会在此目录下存储日志文件
# 加上这一行，防止后面代码报错
tb = None
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils
import time
from .xlstm_ais import TB_LOG

logger = logging.getLogger(__name__)


@torch.no_grad()#告诉 PyTorch 不要计算梯度，也不记录 autograd 计算图，用于推理加速、节省显存。
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """#从一个已有的历史轨迹 seqs 开始，连续生成 steps 步预测轨迹点，形成“未来轨迹”
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time.
    """
    max_seqlen = model.get_max_seqlen()#模型支持的最大序列长度
    model.eval()#设置为推理模式，关闭 Dropout 和 BatchNorm 的训练行为
    for k in range(steps):#循环 steps 次，每次预测 1 个新轨迹点，接到原序列后继续作为下一步输入
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed
        #只保留最近一段轨迹作为输入上下文，避免输入太长
        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)#输出 logits：维度是 (batch_size, seqlen, full_size)，表示每个时间点对 [lat, lon, sog, cog] 预测的“概率分布（未 softmax）
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        #  只取最新时间点的输出，调节 temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):#限制邻近区域采样
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:#限制 top-k 候选项
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely采样 or 贪婪选最大概率
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # 将新预测点 x_sample 添加到当前轨迹中，继续用它预测下一个点
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


class TrainerConfig:
    # 这是一个用于存储训练超参数的配置类，主要用于简洁地管理模型训练过程中所需的各种参数，如：学习率、批大小、训练轮数等
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
     #初始化训练器
    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):#保存最佳模型
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")#记录当前训练过程中最佳周期以及保存模型的路径
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(raw_model.parameters())#获取模型与优化器
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True#判断是否启用 loss 分离（模糊训练时）
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)

            # 【修改】使用更简洁的进度条配置
            if is_train:
                pbar = tqdm(enumerate(loader),
                            total=len(loader),
                            desc=f"Epoch {epoch + 1}/{config.max_epochs}",  # 【新增】简化描述
                            ncols=100,  # 【新增】固定进度条宽度
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')  # 【新增】自定义格式
            else:
                pbar = tqdm(enumerate(loader),
                            total=len(loader),
                            desc=f"Valid Epoch {epoch + 1}",
                            ncols=100)

            d_loss, d_reg_loss, d_n = 0, 0, 0

            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    loss = loss.mean()
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # 【修改】更新进度条的后缀信息，而不是描述
                    if it % 10 == 0:  # 【新增】每 10 个 iter 更新一次，减少 I/O
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{lr:.2e}',
                            'avg_loss': f'{d_loss / d_n:.4f}'
                        })

                    # 【删除】移除 time.sleep(0.1)，这会严重拖慢训练！

                    # tb logging
                    if TB_LOG:
                        tb.add_scalar("loss", loss.item(), epoch * n_batches + it)
                        tb.add_scalar("lr", lr, epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)
                else:
                    # 【新增】验证时也显示进度信息
                    if it % 10 == 0:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 【修改】epoch 结束后的日志输出
            avg_loss = d_loss / d_n
            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"[{split}] Epoch {epoch + 1}/{config.max_epochs} | "
                        f"Loss: {avg_loss:.5f} | Reg Loss: {d_reg_loss / d_n:.5f} | LR: {lr:.2e}")
                else:
                    logging.info(
                        f"[{split}] Epoch {epoch + 1}/{config.max_epochs} | "
                        f"Loss: {avg_loss:.5f} | LR: {lr:.2e}")
            else:
                logging.info(
                    f"[{split}] Epoch {epoch + 1}/{config.max_epochs} | "
                    f"Loss: {avg_loss:.5f}")

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

        best_loss = float('inf')#初始化 best_loss 为一个非常大的数，表示初始时没有最佳的损失值
        self.tokens = 0  # 这个计数器用于学习率衰减
        best_epoch = 0
        total_time = 0

        for epoch in range(config.max_epochs):
            start_time = time.time()
            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            #涉及到 早停（early stopping） 和 模型保存（checkpointing） supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model#使用训练好的模型进行推理，预测未来的轨迹
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 8
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)#它用这些初始样本调用 sample() 函数，从模型中生成未来的轨迹预测

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)#从测试集选择若干轨迹，绘制它们的输入和模型的预测输出
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)#每个轨迹的输入部分以点划线显示，预测部分以 x 标记显示
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)#将每个训练周期（epoch）的图像保存为 JPG 文件
            plt.close()
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
        average_time = total_time / config.max_epochs
        logging.info(f"average time per epoch: {average_time:.4f}")

        # Final state保存训练模型的最后状态，通常在训练结束时执行。它保存模型的参数（state_dict）到指定的文件路径
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        #logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)
