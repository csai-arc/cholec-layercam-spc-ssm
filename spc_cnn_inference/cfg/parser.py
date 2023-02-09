# -*- coding: utf-8 -*-
"""
/*********************************************************************
*
*  Copyright (c) 2022, Toshiba Software India Pvt Limited
*
*  This file is part of Wafer Defect Classification Inference Application Package
*  File Description : parser.py
*  This file contains parser functions for WDC application
*
*  All rights reserved.
*
*********************************************************************/
"""

import argparse
import sys

from cfg.defaults import get_cfg

def parse_args():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(
        description="WDC inference package"
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="cfg/wdc_config.yaml",
        type=str,
    )

    parser.add_argument(
        "--mode",
        dest="mode",
        help="Execution mode. --mode 1: Inference with single greyscale image. --mode 2: Inference with two greyscale images.",
        default=2,
        type=int,
    )
    
    parser.add_argument(
        "--eval_mode",
        dest="eval_mode",
        help="Eval Execution mode. --eval_mode 0: Inference with no evaluation metric. --eval_mode 1: Inference with evaluation metric.",
        default=0,
        type=int,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """Load command line arguments"""
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Create the checkpoint dir.
    return cfg
