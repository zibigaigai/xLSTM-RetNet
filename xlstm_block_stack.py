# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
from torch import nn

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from components.ln import LayerNorm
from .blocks.retnet.block import RetNetBlock, RetNetBlockConfig

@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None
    slstm_block: Optional[sLSTMBlockConfig] = None
    retnet_block: Optional[RetNetBlockConfig] = None

    context_length: int = 10
    num_blocks: int = 5
    embedding_dim: int = 256
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.2

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: Union[list[int], Literal["all"]] = field(default_factory=lambda:[1] )

    #  RetNet 的位置
    retnet_at: list[int] = field(default_factory=list)
    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    # 2: use the RetNet block
    _block_map: str = "0,1,0,2,2"

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        block_map = [0] * self.num_blocks
        for idx in self.slstm_at:
            assert idx < self.num_blocks
            block_map[idx] = 1
        for idx in self.retnet_at:  # ★ 新增
            assert idx < self.num_blocks
            block_map[idx] = 2  # RetNet
        return ",".join(map(str, block_map))

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            
            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()
            # ========== RetNet 配置同步 ==========
        if self.retnet_block is not None:
            self.retnet_block.embedding_dim = self.embedding_dim
            self.retnet_block._num_blocks = self.num_blocks

            if self.retnet_block is not None:
                self.retnet_block.embedding_dim = self.embedding_dim

                # 强制校验必需参数
                assert hasattr(self.retnet_block, 'heads') and self.retnet_block.heads is not None
                assert hasattr(self.retnet_block, 'layers') and self.retnet_block.layers > 0
                assert self.embedding_dim % self.retnet_block.heads == 0

            # ★ 校验：layers 合法（至少1层）
            if hasattr(self.retnet_block, "layers"):
                assert self.retnet_block.layers > 0, \
                    "RetNetBlockConfig.layers 必须 >= 1"

            # 如果需要，可以在这里继续调用 __post_init__（如果 RetNetBlockConfig 有）
            if hasattr(self.retnet_block, "__post_init__"):
                self.retnet_block.__post_init__()
        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: xLSTMBlockStackConfig):
        #根据数量创建slstm与mlstm
        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config))
            elif block_type_int == 1:
                config = deepcopy(self.config.slstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(sLSTMBlock(config=config))
            elif block_type_int == 2:  # ★ 新增
                cfg = deepcopy(self.config.retnet_block)
                if hasattr(cfg, "_block_idx"):
                    cfg._block_idx = block_idx
                blocks.append(RetNetBlock(config=cfg))
            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:#重置参数
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.post_blocks_norm(x)

        return x

    def step(
        self, x: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        if state is None:
            state = {}

        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(x, **state.get(f"block_{block_idx}", {}))

        x = self.post_blocks_norm(x)

        return x, state
