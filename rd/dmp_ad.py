import pickle
import torch
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from PIL import Image
import os
import uuid
import json
import tqdm
from advg.nusc_dump import *

ds_name = 'advnusc'

create_new_ds(f'adv_ns/nuscs/{ds_name}', 'data/nuscenes')

with open(f'txts/{ds_name}.txt', 'r') as f:
    pths = f.readlines()

num_scenes = len(pths)
for i in tqdm.tqdm(range(num_scenes)):
    sam_pth, batch_pth, suf = pths[i].split()
    samples = pickle.load(open(sam_pth, 'rb')).to(torch.float32)
    batch = pickle.load(open(batch_pth, 'rb'))
    dump_a_scene(batch, f'adv_ns/nuscs/{ds_name}', 'data/nuscenes', f'-{suf}', samples, True if i == num_scenes - 1 or i % 10 == 0 else False)