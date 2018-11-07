# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            '''e.g.
            gbackbone.fpn.fpn_inner1.bias
            gbackbone.fpn.fpn_layer1.bias
            gbackbone.fpn.fpn_inner2.bias
            gbackbone.fpn.fpn_layer2.bias
            gbackbone.fpn.fpn_inner3.bias
            gbackbone.fpn.fpn_layer3.bias
            gbackbone.fpn.fpn_inner4.bias
            gbackbone.fpn.fpn_layer4.bias
            grpn.head.conv.bias
            grpn.head.cls_logits.bias
            grpn.head.bbox_pred.bias
            groi_heads.box.feature_extractor.fc6.bias
            groi_heads.box.feature_extractor.fc7.bias
            groi_heads.box.predictor.cls_score.bias
            groi_heads.box.predictor.bbox_pred.bias
            '''
            # e.g. 0.01 * 2
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS, # e.g. (60000, 80000)
        cfg.SOLVER.GAMMA, # e.g. 0.1
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
