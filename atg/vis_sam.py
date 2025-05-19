import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision.io import write_video
from advtg.dmodel import AdvKinematicDiffusionModel
from advtg.gen import *
from advtg.ma import *
from advtg.deb import *
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--sample_path', type=str)
arg.add_argument('--batch_path', type=str, default=None)
arg.add_argument('--video_out', type=str)
arg.add_argument('--st_frm_idx', type=int, default=0)
arg.add_argument('--ed_frm_idx', type=int, default=-1)

if __name__ == '__main__':
    arg = arg.parse_args()
    sam = pickle.load(open(arg.sample_path, 'rb'))[0]
    if arg.batch_path is not None:
        batch = pickle.load(open(arg.batch_path, 'rb'))
        sample_to_mp4(sam, arg.video_out, with_bev=True, batch=batch, \
                      obj_box=True, st_frm_idx=arg.st_frm_idx, ed_frm_idx=arg.ed_frm_idx,
                      up_scale=True)
    else:
        sample_to_mp4(sam, arg.video_out, with_bev=False)