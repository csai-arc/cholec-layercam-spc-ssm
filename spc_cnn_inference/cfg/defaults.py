# -*- coding: utf-8 -*-
"""
/*********************************************************************
*
*  Copyright (c) 2022, Toshiba Software India Pvt Limited
*
*  This file is part of Wafer Defect Classification Inference Application Package
*  File Description : defaults.py
*  This file contains default values for configuration settings
*
*  All rights reserved.
*
*********************************************************************/
"""

from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

#Model path
_C.TRAIN.MODEL_PATH_MODE1 = "./models/model_best_mode1.pth.tar"
_C.TRAIN.MODEL_PATH_MODE2 = "./models/model_best_mode2.pth.tar"
_C.TRAIN.MODEL_NAME = "effnetv2_s"
#-----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 9

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.5, 0.5, 0.5]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.5, 0.5, 0.5]

# The spatial crop size for training.
_C.DATA.RESIZE_SIZE = 480

_C.DATA.IMG_HEIGHT = 480

_C.DATA.IMG_WIDTH = 480

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

#output result path
_C.OUT_MODE1_RES_PTH = "./output/wdc_out_mode_1.xlsx"
_C.OUT_MODE2_RES_PTH = "./output/wdc_out_mode_2.xlsx"
_C.OUT_EVAL_MODE1_RES_PTH = "./output_eval/wdc_out_mode_1.xlsx"
_C.OUT_EVAL_MODE2_RES_PTH = "./output_eval/wdc_out_mode_2.xlsx"
_C.OUT_EVAL_MODE1_PTH = "./output_eval/"

_C.INPUT_PATH = "./input"
_C.INPUT_EVAL_PATH = "./input_eval"
_C.GRAD_CAM_FLAG = 1

#class names
_C.CLASS_NAMES = [
    "10", "11", "152", "159", "66", "7",
    "8", "80", "99"
]


def _assert_and_infer_cfg(cfg):
    '''assertion of configuration file'''
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
