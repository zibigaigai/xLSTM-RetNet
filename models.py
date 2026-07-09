# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import logging

import torch
from torch import nn

from components.init import small_init_init_

from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from torch.nn import functional as F
logger = logging.getLogger(__name__)

class xLSTMShipTrajectory(nn.Module):
    """
       xLSTM-based model for ship trajectory prediction.
       Takes AIS features as input: lat, lon, sog, cog.
       """
    def __init__(self,config,partition_model = None):
        super().__init__()


        self.lat_size = config.lat_size  
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer(
            "att_sizes",
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes",
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))

        if hasattr(config, "partition_mode"):  
            self.partition_mode = config.partition_mode  
        else:
            self.partition_mode = "uniform"  
        self.partition_model = partition_model
        if hasattr(config, "blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1 / 3)
            else:
                self.blur_module = None  
        else:
            
            self.blur = False
            self.blur_module = None

        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer-->xLSTM
        # 创建配置
        xlstm_config = xLSTMBlockStackConfig(
            context_length=config.max_seqlen,
            num_blocks=config.depth,
            embedding_dim=config.n_embd,
            dropout=config.dropout,
            slstm_at=[1],  
            mlstm_block=config.mlstm_block_cfg,  
            slstm_block=config.slstm_block_cfg,  
        )

 
        self.blocks = xLSTMBlockStack(config=xlstm_config)

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd) 
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)  
        else:
            self.head = nn.Linear(config.n_embd, self.full_size,
                                  bias=False) 

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights) 
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

  

        self.kl_lambda = getattr(config, "kl_lambda", 1.0) 
        self.kl_sigma_lat = getattr(config, "kl_sigma_lat", 2.5)
        self.kl_sigma_lon = getattr(config, "kl_sigma_lon", 2.5)  
        self.kl_sigma_sog = getattr(config, "kl_sigma_sog", 2.0)  
        self.kl_sigma_cog = getattr(config, "kl_sigma_cog", 2.0)  

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.

        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            model: currenly only supports "uniform".

        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x * self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):

            idxs = (x * self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:, :, :2])
            #             pdb.set_trace()
            idxs[:, :, 0] = torch.round(lat_ids.reshape((bs, seqlen))).long()
            idxs[:, :, 1] = torch.round(lon_ids.reshape((bs, seqlen))).long()
            return idxs, idxs_uniform
    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:],
                otherwise inputs = x.
        Returns:
            logits, loss
        """

        if self.mode in ("mlp_pos", "mlp",):
            idxs, idxs_uniform = x, x  # use the real-values of x.
        else:
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)

        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()
            targets_uniform = idxs_uniform[:, 1:, :].contiguous()
            inputs_real = x[:, :-1, :].contiguous()
            targets_real = x[:, 1:, :].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:, :, 0])  # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings), dim=-1)

 
        position_embeddings = self.pos_emb[:, :seqlen, :]
        hidden_states = self.drop(token_embeddings + position_embeddings)

     
        hidden_states = self.blocks(hidden_states)

   
        hidden_states = self.ln_f(hidden_states)


        logits = self.head(hidden_states)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
  
        loss = None
        loss_tuple = None  
        if targets is not None:
            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size),
                                       targets[:, :, 2].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                       targets[:, :, 3].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                       targets[:, :, 0].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                       targets[:, :, 1].view(-1),
                                       reduction="none").view(batchsize, seqlen)

         
            if self.blur and self.blur_module is not None:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1, 1, self.lat_size)).reshape(
                        lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1, 1, self.lon_size)).reshape(
                        lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1, 1, self.sog_size)).reshape(
                        sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1, 1, self.cog_size)).reshape(
                        cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:, :, 0].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:, :, 1].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:, :, 2].view(-1),
                                                  reduction="none").view(batchsize, seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:, :, 3].view(-1),
                                                  reduction="none").view(batchsize, seqlen)

                    lat_loss += self.blur_loss_w * blurred_lat_loss
                    lon_loss += self.blur_loss_w * blurred_lon_loss
                    sog_loss += self.blur_loss_w * blurred_sog_loss
                    cog_loss += self.blur_loss_w * blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs

 
            def gaussian_target(target_idx, size, sigma=2.0):
         
                idxs = torch.arange(size, device=target_idx.device).float().view(1, 1, -1)
                target = target_idx.unsqueeze(-1).float()
                gauss = torch.exp(-0.5 * ((idxs - target) / sigma) ** 2)
                gauss = gauss + 1e-10 
                gauss = gauss / gauss.sum(dim=-1, keepdim=True)
                return gauss

       
            lat_log_probs = F.log_softmax(lat_logits, dim=-1)
            lon_log_probs = F.log_softmax(lon_logits, dim=-1)
            sog_log_probs = F.log_softmax(sog_logits, dim=-1)
            cog_log_probs = F.log_softmax(cog_logits, dim=-1)

 
            lat_gauss = gaussian_target(targets[:, :, 0], self.lat_size, sigma=self.kl_sigma_lat)
            lon_gauss = gaussian_target(targets[:, :, 1], self.lon_size, sigma=self.kl_sigma_lon)
            sog_gauss = gaussian_target(targets[:, :, 2], self.sog_size, sigma=self.kl_sigma_sog)
            cog_gauss = gaussian_target(targets[:, :, 3], self.cog_size, sigma=self.kl_sigma_cog)

       
            kl_lat = F.kl_div(lat_log_probs, lat_gauss, reduction="none").sum(dim=-1)
            kl_lon = F.kl_div(lon_log_probs, lon_gauss, reduction="none").sum(dim=-1)
            kl_sog = F.kl_div(sog_log_probs, sog_gauss, reduction="none").sum(dim=-1)
            kl_cog = F.kl_div(cog_log_probs, cog_gauss, reduction="none").sum(dim=-1)
            kl_loss = (kl_lat + kl_lon + kl_sog + kl_cog) / 4.0  # (B, T)

      
            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            ce_loss = sum(loss_tuple)

            if masks is not None:
                ce_loss = (ce_loss * masks).sum(dim=1) / masks.sum(dim=1)
                kl_loss = (kl_loss * masks).sum(dim=1) / masks.sum(dim=1)

            ce_loss = ce_loss.mean()
            kl_loss = kl_loss.mean()

    
            loss = ce_loss + self.kl_lambda * kl_loss

        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss
