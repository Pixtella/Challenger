import pickle
import torch
from advtg.deb import *
from advtg.gen import *
from tqdm import tqdm
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--batch_in', type=str)
arg.add_argument('--batch_out', type=str)
arg.add_argument('--ckpt', type=str)
arg.add_argument('--obj', type=int)
arg.add_argument('--objs', type=str, default=None)
arg.add_argument('--temperature', type=float)
arg.add_argument('--cem_iters', type=int)
arg.add_argument('--seed', type=int, default=0)



if __name__ == '__main__':
    arg = arg.parse_args()

    batch = pickle.load(open(arg.batch_in, 'rb'))
    akd = AdvKinematicDiffusionModel(ckpt_path=arg.ckpt)
    akd.cuda()
    if arg.objs is not None:
        arg.objs = [int(i) for i in arg.objs.split(',')]
        for obj in arg.objs:
            batch = run_es_on_mdd_scene(batch, akd, obj, arg.seed, cem_iters=arg.cem_iters, temperature=arg.temperature)
            if batch is None:
                exit(0)
            if obj != arg.objs[-1]:
                batch = prepare_batch_for_more_adv(batch)
    else:
        batch = run_es_on_mdd_scene(batch, akd, arg.obj, arg.seed, cem_iters=arg.cem_iters, temperature=arg.temperature, disable_ego_dis=False)
    if batch is not None:
        pickle.dump(batch, open(arg.batch_out, 'wb'))