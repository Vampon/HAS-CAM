from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager
from fastreid.predictor import FeatureExtractionDemo
from scipy.spatial.distance import cosine,euclidean, cityblock, minkowski, correlation, jaccard
from PIL import Image
import pickle
import sys
import numpy as np
import tqdm
import argparse
import glob
import os
from torch.backends import cudnn

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


class fakeCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# 模拟命令行参数
fake_args_dict = {
    "config_file": "fastreid/configs/VeRi/VeRi_sbs_R50-ibn.yml",
    "parallel": True,
    "input": [],
    "output": "reid_output_folder",
    "opts": []
}
# 创建Namespace对象来模拟命令行参数
args = fakeCfg(**fake_args_dict)

cfg = setup_cfg(args)
feature_extractor = FeatureExtractionDemo(cfg, parallel=args.parallel)