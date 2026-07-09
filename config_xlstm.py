
import os 
import pickle 
import torch

from .blocks.mlstm.block import mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlockConfig
from .xlstm_block_stack import xLSTMBlockStackConfig
from .blocks.mlstm.layer import mLSTMLayerConfig
from .blocks.slstm.layer import sLSTMLayerConfig
from .components.feedforward import FeedForwardConfig
from .blocks.retnet.block import RetNetBlockConfig

class Config():
    retrain = False 
    tb_log = False  
    device = torch.device("cuda")
    #     device = torch.device("cpu")

    max_epochs = 10  
    batch_size = 32  
    n_samples = 16  

    init_seqlen = 30  
    max_seqlen = 120
    min_seqlen = 36


    dataset_name = "ct_dma"  
    model_type = "xlstm_retnet_kl"

    if dataset_name == "ct_dma":  
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72
        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

     
        lat_min = 55.5 
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13


    # ===========================================================================
    # Model and sampling flags
    mode = "pos" 
    # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode = "pos_vicinity"
    top_k = 10  
    r_vicinity = 40  

  
    blur = True  
    blur_learnable = False  
    blur_loss_w = 1.0  
    blur_n = 2  
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
    n_head = 8  # Transformer
    n_layer = 8  # Transformer 
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    # optimization parameters
    # ===================================================
    betas = (0.9, 0.95)  
    grad_norm_clip = 1.0  
    weight_decay = 0.1 
    final_tokens = 260e9  
    num_workers = 0  
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


    mlstm_block_cfg = mLSTMBlockConfig(mlstm=mlstm_layer_cfg)
    slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=feedforward_cfg)
   
    retnet_block_cfg = RetNetBlockConfig(
        embedding_dim=256,  
        layers=1,
        heads=4, 
        ffn_size=1024,
        double_v_dim=False,
    )
  
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
        slstm_at=[1],  
        retnet_at=[3,4]    
    )
  
    depth = 5
    dropout = 0.2
    xlstm_config = xlstm_stack_cfg

    resid_pdrop = 0.5     
    embd_pdrop = 0.3      
    attn_pdrop = 0.1
    learning_rate = 1e-4  
    lr_decay = False
    warmup_tokens = 512 * 20
   
 
    kl_lambda = 0.75  
    kl_sigma_lat = 2.0  
    kl_sigma_lon = 2.0  
    kl_sigma_sog = 1.5 
    kl_sigma_cog = 1.5 

    @staticmethod
    def get_architecture_string(slstm_at, retnet_at, num_blocks):
        
        block_types = ['M'] * num_blocks  
        
  
        for idx in slstm_at:
            if idx < num_blocks:
                block_types[idx] = 'S'
        

        for idx in retnet_at:
            if idx < num_blocks:
                block_types[idx] = 'R'
        

        arch_string = "".join(block_types)
        return arch_string

    arch_string = get_architecture_string(
        xlstm_stack_cfg.slstm_at,
        xlstm_stack_cfg.retnet_at,
        xlstm_stack_cfg.num_blocks
    )
    

      
    filename = f"{model_type}_{arch_string}_kl{kl_lambda}"

    # 例如：1101_1425_xlstm_RMSS_kl0.1

    savedir = "./outputs/results/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model_3.pt")
