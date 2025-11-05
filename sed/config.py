# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_sed_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES = "datasets/coco/coco_stuff/split/seen_indexes.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES = "datasets/coco/coco_stuff/split/unseen_indexes.json"

    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"

    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"

    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask

    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM = 128

    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [64, 32]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS = [256, 128]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS = [32, 16]

    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = 4
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS = 128
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [6, 6]
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES = 12
    cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE = "linear"
    cfg.MODEL.SEM_SEG_HEAD.CNEXT_TYPE = "V1"
    cfg.MODEL.SEM_SEG_HEAD.KERNEL_SIZE = [7, 7, 7]

    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.SOLVER.CLIP_MULTIPLIER = 0.01

    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.FAST_INFERENCE = False
    cfg.TEST.TOPK = 1

    # convnext backbone
    cfg.MODEL.ENC = CN()
    cfg.MODEL.ENC.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.ENC.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.ENC.EMBED_DIM = 768


    cfg.MODEL.LORA = CN()
    cfg.MODEL.LORA.ENABLED = True
    cfg.MODEL.LORA.DB_PATH = "loradb/"
    cfg.MODEL.LORA.NAME = "default"
    cfg.MODEL.LORA.RANK = 8
    cfg.MODEL.LORA.ALPHA = 16
    cfg.MODEL.LORA.DROPOUT = 0
    cfg.MODEL.LORA.USE_RSLORA = False
    cfg.MODEL.LORA.USE_DORA = False
    cfg.MODEL.LORA.BIAS = "none"



    
def add_lora_config(
    cfg,
):
    # List of the torch.nn.Module in the model that LoRAs should be attached to    
    MODUELS = [
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.0.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.0.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.1.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.1.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.2.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.0.blocks.2.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.0.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.0.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.1.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.1.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.2.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.1.blocks.2.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.0.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.0.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.1.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.1.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.2.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.2.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.2.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.3.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.3.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.4.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.4.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.5.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.5.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.6.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.6.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.7.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.7.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.8.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.8.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.9.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.9.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.10.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.10.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.11.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.11.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.12.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.12.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.13.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.13.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.14.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.14.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.15.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.15.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.16.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.16.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.17.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.17.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.18.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.18.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.19.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.19.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.20.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.20.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.21.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.21.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.22.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.22.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.23.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.23.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.24.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.24.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.25.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.25.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.26.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.2.blocks.26.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.0.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.0.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.1.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.1.mlp.fc2",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.2.mlp.fc1",
        "sem_seg_head.predictor.clip_model.visual.trunk.stages.3.blocks.2.mlp.fc2",
        # "sem_seg_head.predictor.clip_model.visual.head.proj.weight",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.0.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.0.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.0.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.1.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.1.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.1.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.2.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.2.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.2.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.3.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.3.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.3.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.4.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.4.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.4.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.5.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.5.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.5.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.6.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.6.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.6.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.7.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.7.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.7.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.8.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.8.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.8.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.9.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.9.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.9.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.10.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.10.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.10.attn.v_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.11.attn.q_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.11.attn.k_proj",
        "sem_seg_head.predictor.clip_model.transformer.resblocks.11.attn.v_proj",
        
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.0.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.0.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.1.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.1.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.2.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.2.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.3.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.3.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.4.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.4.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.5.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.5.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.6.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.6.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.7.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.7.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.8.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.8.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.9.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.9.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.10.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.10.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.11.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.11.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.12.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.12.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.13.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.13.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.14.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.14.mlp.c_proj",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.15.mlp.c_fc",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.15.mlp.c_proj",


        # "sem_seg_head.predictor.clip_model.transformer.resblocks.0.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.0.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.1.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.1.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.2.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.2.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.3.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.3.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.4.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.4.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.5.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.5.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.6.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.6.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.7.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.7.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.8.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.8.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.9.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.9.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.10.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.10.ln_2",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.11.ln_1",
        # "sem_seg_head.predictor.clip_model.transformer.resblocks.11.ln_2",
        # "sem_seg_head.predictor.clip_model.token_embedding",
        # "sem_seg_head.predictor.clip_model.ln_final",

        # "sem_seg_head.predictor.transformer.decoder_guidance_projection.0.0",
        # "sem_seg_head.predictor.transformer.decoder_guidance_projection.1.0",
        # "sem_seg_head.predictor.transformer.decoder_guidance_projection.2.0",
        # "sem_seg_head.predictor.transformer.decoder_corr_guidance_projection.0.0",
        # "sem_seg_head.predictor.transformer.decoder_corr_guidance_projection.1.0",
        # "sem_seg_head.predictor.transformer.decoder_corr_guidance_projection.2.0",
        # "sem_seg_head.predictor.transformer.decoder1.up",

        # "sem_seg_head.predictor.transformer.decoder1.attention.attention.q",
        # "sem_seg_head.predictor.transformer.decoder1.attention.attention.k",
        # "sem_seg_head.predictor.transformer.decoder1.attention.attention.v",
        # "sem_seg_head.predictor.transformer.decoder1.attention.MLP.0",
        # "sem_seg_head.predictor.transformer.decoder1.attention.MLP.2",
        # "sem_seg_head.predictor.transformer.decoder1.attention.norm1",
        # "sem_seg_head.predictor.transformer.decoder1.attention.norm2",
        # "sem_seg_head.predictor.transformer.decoder2.up",

        # "sem_seg_head.predictor.transformer.decoder2.attention.attention.q",
        # "sem_seg_head.predictor.transformer.decoder2.attention.attention.k",
        # "sem_seg_head.predictor.transformer.decoder2.attention.attention.v",
        # "sem_seg_head.predictor.transformer.decoder2.attention.MLP.0",
        # "sem_seg_head.predictor.transformer.decoder2.attention.MLP.2",
        # "sem_seg_head.predictor.transformer.decoder2.attention.norm1",
        # "sem_seg_head.predictor.transformer.decoder2.attention.norm2",
        # "sem_seg_head.predictor.transformer.decoder3.up",

        # "sem_seg_head.predictor.transformer.decoder3.attention.attention.q",
        # "sem_seg_head.predictor.transformer.decoder3.attention.attention.k",
        # "sem_seg_head.predictor.transformer.decoder3.attention.attention.v",
        # "sem_seg_head.predictor.transformer.decoder3.attention.MLP.0",
        # "sem_seg_head.predictor.transformer.decoder3.attention.MLP.2",
        # "sem_seg_head.predictor.transformer.decoder3.attention.norm1",
        # "sem_seg_head.predictor.transformer.decoder3.attention.norm2",
        # "sem_seg_head.predictor.transformer.head0",
        # "sem_seg_head.predictor.transformer.head1",
        # "sem_seg_head.predictor.transformer.head2",
        # "sem_seg_head.predictor.transformer.head",
    ]


    cfg.MODEL.LORA = CN()
    cfg.MODEL.LORA.ENABLED = True
    cfg.MODEL.LORA.DB_PATH = "loradb/"
    cfg.MODEL.LORA.NAME = "default"
    cfg.MODEL.LORA.MODUELS = MODUELS
    cfg.MODEL.LORA.RANK = 8
    cfg.MODEL.LORA.ALPHA = 16
    cfg.MODEL.LORA.DROPOUT = 0
    cfg.MODEL.LORA.USE_RSLORA = False
    cfg.MODEL.LORA.USE_DORA = False
    cfg.MODEL.LORA.BIAS = "none"
