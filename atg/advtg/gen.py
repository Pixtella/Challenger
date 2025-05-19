from advtg.dmodel import AdvKinematicDiffusionModel
from advtg.ma import *
import torch
import numpy as np
import pickle
import math
from typing import Optional
from magicdrivedit.mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from einops import rearrange
import torch.nn.functional as F
import shapely.geometry as sg
from shapely import vectorized
import tqdm
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
import time


EDITABLE_CLASSES = [0]

xbound = [-50.0, 50.0, 0.5]
ybound = [-50.0, 50.0, 0.5]

patch_h = ybound[1] - ybound[0]
patch_w = xbound[1] - xbound[0]

canvas_h = int(patch_h / ybound[2])
canvas_w = int(patch_w / xbound[2])

lidar2canvas = np.array([
    [canvas_h / patch_h, 0, canvas_h / 2],
    [0, canvas_w / patch_w, canvas_w / 2],
    [0, 0, 1]
])


def get_editable_objects(batch):
    DIST_THRESH = 20.0
    bboxes: torch.Tensor = batch['bboxes_3d_data'][0].data['bboxes'] # (NUM_FRAMES, 1, NUM_OBJS, 8, 3)
    masks: torch.Tensor = batch['bboxes_3d_data'][0].data['masks']   # (NUM_FRAMES, NCAM, NUM_OBJS)
    classes: torch.Tensor = batch['bboxes_3d_data'][0].data['classes'] # (NUM_FRAMES, 1, NUM_OBJS)
    bev: torch.Tensor = batch['bev_map_with_aux'] # (1, NUM_FRAMES, C, H, W)
    NUM_FRAMES, _, NUM_OBJS = masks.shape
    obj_classes = classes[:, 0, :].max(dim=0)[0] # (NUM_OBJS,)
    assert obj_classes.min().item() != -1, 'Check object classes!'
    editable_objs = []
    for i in range(NUM_OBJS):
        if obj_classes[i].item() not in EDITABLE_CLASSES:
            continue
        valid_ts_mask = classes[:, 0, i] != -1 # (NUM_FRAMES, )
        
        center_traj = bboxes[:, 0, i, [0, 3, 4, 7], :].mean(dim=1) # (NUM_FRAMES, 3)
        center_traj[:, 2] = 1.0
        center_traj_canvas = center_traj @ lidar2canvas.T # (NUM_FRAMES, 3)
        center_traj_canvas = center_traj_canvas.round().long() # (NUM_FRAMES, 3)
        center_dist = center_traj[:, :2].norm(dim=1) # (NUM_FRAMES, )
        
        in_canvas_mask = (center_traj_canvas[:, 0] >= 0) & (center_traj_canvas[:, 0] < canvas_w) & \
                         (center_traj_canvas[:, 1] >= 0) & (center_traj_canvas[:, 1] < canvas_h) & \
                          valid_ts_mask # (NUM_FRAMES, )
        in_canvas_ts = in_canvas_mask.nonzero().view(-1) # (NUM_TS, )
        in_road_percent = bev[0, in_canvas_ts, 0, center_traj_canvas[in_canvas_ts, 0], center_traj_canvas[in_canvas_ts, 1]].sum() / len(in_canvas_ts)
        if in_road_percent < 0.7:
            continue
        if in_canvas_ts.shape[0] == 0:
            continue
        assert center_dist[in_canvas_ts].min() > 1e-3, 'Check center_dist!'
        if center_dist[in_canvas_ts].min() > DIST_THRESH:
            continue
        editable_objs.append(i)
    return editable_objs

def rebuild_mask(batch, obj_idx: int):
    T = batch['bboxes_3d_data'][0].data['bboxes'].shape[0]
    for t in range(T):
        pts = batch['bboxes_3d_data'][0].data['bboxes'][t, 0, obj_idx, :, :] # (8, 3)
        pts = torch.cat([pts, torch.ones(8, 1)], dim=1)
        prj = batch['meta_data']['lidar2image'][t][0].data # (Ncam, 4, 4)
        pts_img = torch.matmul(prj, pts.T).permute(0, 2, 1) # (Ncam, 8, 4)
        mask = pts_img[:, :, 2] > 0.01
        pts_img = pts_img[:, :, :2] / pts_img[:, :, 2:3]
        mask_ = \
        torch.logical_and(torch.logical_and(pts_img[:, :, 0] > 0, pts_img[:, :, 0] < 1600), \
                          torch.logical_and(pts_img[:, :, 1] > 0, pts_img[:, :, 1] < 900))
        mask = torch.logical_and(mask, mask_)
        mask = torch.any(mask, dim=1)
        batch['bboxes_3d_data'][0].data['masks'][t, :, obj_idx] = mask.float()


def build_obj_history(instance_box: torch.Tensor, ts: int):
    obj_history = torch.zeros((1, 5, 7), device=instance_box.device)
    t = torch.tensor([1, 1, -1], device=instance_box.device, dtype=instance_box.dtype)
    b = torch.tensor([0, 0, -math.pi/2], device=instance_box.device, dtype=instance_box.dtype)
    obj_history[0, :, :3] = global_state_se2_tensor_to_local(instance_box[ts-16:ts+1:4, [0, 1, 6]] * t + b, instance_box[ts, [0, 1, 6]] * t + b)
    return obj_history

def convert_to_global_convert_heading(ref, points):
    if isinstance(ref, EgoState):
        ref2 = np.array([ref.rear_axle.x, ref.rear_axle.y, ref.rear_axle.heading])
    else:
        ref2 = ref.copy()
        ref2[2] = -ref2[2] - math.pi / 2
    return convert_to_global(ref2, points)

def convert_to_local_convert_heading(ref, points):
    if isinstance(ref, EgoState):
        ref2 = np.array([ref.rear_axle.x, ref.rear_axle.y, ref.rear_axle.heading])
    else:
        ref2 = ref.copy()
        ref2[2] = -ref2[2] - math.pi / 2
    return convert_to_local(ref2, points)

def interp_instance_box(instance_box: torch.Tensor):
    # instance_box: (NUM_FRAMES, 7)
    NUM_FRAMES = instance_box.shape[0]
    KEYFRAME_STEP = 6
    t = 0
    while t + KEYFRAME_STEP < NUM_FRAMES:
        for i in range(1, KEYFRAME_STEP):
            instance_box[t + i, [0, 1, 6]] = instance_box[t, [0, 1, 6]] + \
                (instance_box[t + KEYFRAME_STEP, [0, 1, 6]] - instance_box[t, [0, 1, 6]]) * i / KEYFRAME_STEP
        t += KEYFRAME_STEP
    return instance_box

def corners_to_7d(corners: torch.Tensor):
    # corners: (B, 8, 3/4)
    dim4_flag = False
    if len(corners.shape) == 4: # (NUM_FRAMES, NUM_OBJS, 8, 3/4)
        d1, d2, d3, d4 = corners.shape
        corners = corners.view(d1 * d2, d3, d4)
        dim4_flag = True

    B = corners.shape[0]
    ret = torch.zeros((B, 7), device=corners.device)
    ret[:, :3] = corners[:, [0, 3, 4, 7], :3].mean(dim=1) # (x, y, z)
    ret[:, 3] = (corners[:, 0] - corners[:, 4]).norm(dim=1) # (x_size)
    ret[:, 4] = (corners[:, 0] - corners[:, 3]).norm(dim=1) # (y_size)
    ret[:, 5] = (corners[:, 0] - corners[:, 1]).norm(dim=1) # (z_size)
    piv_pts = (corners[:, 4, :3] + corners[:, 7, :3]) / 2 - ret[:, :3] # (B, 3)
    ret[:, 6] = -torch.atan2(piv_pts[:, 1], piv_pts[:, 0]) # (B, )
    if dim4_flag:
        ret = ret.view(d1, d2, 7)
    return ret

def remove_obj(batch, obj_idx: int):
    batch['bboxes_3d_data'][0].data['bboxes'][:, 0, obj_idx, :, :] = 0.0
    batch['bboxes_3d_data'][0].data['masks'][:, :, obj_idx] = False
    batch['bboxes_3d_data'][0].data['classes'][:, 0, obj_idx] = -1
    return batch

def fix_global_traj_heading(traj: torch.Tensor):
    # traj: (B, H, 3)
    HOR = traj.shape[1]
    traj[:, 1:HOR-1, 2] = torch.atan2(traj[:, 2:HOR, 1] - traj[:, 0:HOR-2, 1], traj[:, 2:HOR, 0] - traj[:, 0:HOR-2, 0])
    traj[:, 0, 2] = traj[:, 1, 2]
    traj[:, HOR-1, 2] = traj[:, HOR-2, 2]
    return traj

def fix_heading(traj: np.ndarray):
    # traj: (B, H, 3)
    HOR = traj.shape[1]
    traj[:, 1:HOR-1, 2] = np.arctan2(traj[:, 2:HOR, 1] - traj[:, 0:HOR-2, 1], traj[:, 2:HOR, 0] - traj[:, 0:HOR-2, 0])
    traj[:, 0, 2] = traj[:, 1, 2]
    traj[:, HOR-1, 2] = traj[:, HOR-2, 2]
    return traj

def run_es_on_mdd_scene(batch, model: AdvKinematicDiffusionModel, obj_idx: int, seed: int = 0, cem_iters: int = 20, temperature: float = 0.1, disable_ego_dis: bool = False):
    '''
    :param batch: batch of data that is used as input of MagicDriveDiT
    :param model: model that is used to run the simulation
    :return: edited batch
    '''
    # constants
    LAUNCH_THRESHOLD = 25.0
    ES_POP_SIZE = 256
    H = 16
    RE_PLAN_INTERVAL = 12
    device = 'cuda'
    
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    bboxes: torch.Tensor = batch['bboxes_3d_data'][0].data['bboxes'] # (NUM_FRAMES, 1, NUM_OBJS, 8, 3)
    classes: torch.Tensor = batch['bboxes_3d_data'][0].data['classes'] # (NUM_FRAMES, 1, NUM_OBJS)
    bev_map: torch.Tensor = batch['bev_map_with_aux'] # (1, NUM_FRAMES, 8, CANVAS_H, CANVAS_W)
    odevice = bboxes.device
    bboxes = bboxes.to(device)
    classes = classes.to(device)
    bev_map = bev_map.to(device)
    NUM_FRAMES = bboxes.shape[0]
    NUM_OBJS = bboxes.shape[2]
    center_traj_lidar = bboxes[:, 0, :, [0, 3, 4, 7], :].mean(dim=2) # (NUM_FRAMES, NUM_OBJS, 3)
    corners_lidar = bboxes[:, 0, :, :, :] # (NUM_FRAMES, NUM_OBJS, 8, 3)
    corners_lidar = torch.cat([corners_lidar, torch.ones(NUM_FRAMES, NUM_OBJS, 8, 1, device=device)], dim=3) # (NUM_FRAMES, NUM_OBJS, 8, 4)
    lidar2ego = torch.zeros((NUM_FRAMES, 4, 4))
    ego2global = torch.zeros((NUM_FRAMES, 4, 4))
    for t in range(NUM_FRAMES):
        lidar2ego[t][:3, :3] = qua_list_to_mat(batch['meta_data']['metas'][t][0].data['full_info']['lidar2ego_rotation'])
        lidar2ego[t][:3, 3] = torch.Tensor(batch['meta_data']['metas'][t][0].data['full_info']['lidar2ego_translation'])
        lidar2ego[t][3, 3] = 1.0
        ego2global[t][:3, :3] = qua_list_to_mat(batch['meta_data']['metas'][t][0].data['full_info']['ego2global_rotation'])
        ego2global[t][:3, 3] = torch.Tensor(batch['meta_data']['metas'][t][0].data['full_info']['ego2global_translation'])
        ego2global[t][3, 3] = 1.0
    lidar2ego = lidar2ego.to(center_traj_lidar.device)
    ego2global = ego2global.to(center_traj_lidar.device)
    corners_global = corners_lidar @ (lidar2ego.permute(0, 2, 1)[:, None, :, :]) \
                                   @ (ego2global.permute(0, 2, 1)[:, None, :, :]) # (NUM_FRAMES, NUM_OBJS, 8, 4)
    ego_corners_lidar = LiDARInstance3DBoxes(torch.Tensor([[0, 0, 0.9, 3.5, 7, 1.8, -math.pi]])).corners[0] # (8, 3)
    ego_corners_lidar = torch.cat([ego_corners_lidar, torch.ones(8, 1)], dim=1).to(lidar2ego.device) # (8, 4)
    ego_corners_global = ego_corners_lidar[None, ...] @ lidar2ego.permute(0, 2, 1) @ ego2global.permute(0, 2, 1) # (NUM_FRAMES, 8, 4)
    instance_boxes_global = corners_to_7d(torch.cat([corners_global, ego_corners_global[:, None, :]], dim=1)) # (NUM_FRAMES, NUM_OBJS+1, 7)
    
    ego_traj_global = torch.cat([torch.zeros(1,3), torch.ones(1,1)], dim=-1)\
                        .to(center_traj_lidar.device).repeat(NUM_FRAMES, 1).view(NUM_FRAMES, 1, 4)\
                        @ ego2global.permute(0, 2, 1) # (NUM_FRAMES, 1, 4)
    if 'des_info' not in batch:
        batch['des_info'] = {}
    batch['des_info']['pop_traj_lidar'] = torch.zeros((NUM_FRAMES, ES_POP_SIZE, H, 2), device=device)
    batch['des_info']['pop_traj_lidar_rds'] = torch.zeros((5, NUM_FRAMES, ES_POP_SIZE, H, 2), device=device)
    batch['des_info']['pop_traj_score'] = torch.zeros((NUM_FRAMES, ES_POP_SIZE), device=device)
    batch['des_info']['pop_traj_score_rds'] = torch.zeros((5, NUM_FRAMES, ES_POP_SIZE), device=device)
    batch['des_info']['lidar2ego'] = lidar2ego.to('cpu')
    batch['des_info']['ego2global'] = ego2global.to('cpu')
    batch['des_info']['best_score'] = 0.0
    batch['des_info']['edited_obj'] = obj_idx
    batch['des_info']['rel_pos'] = torch.zeros((NUM_FRAMES, 2), device=device)
    batch['des_info']['rel_spd'] = torch.zeros((NUM_FRAMES, 2), device=device)
    batch['des_info']['bad_gen'] = False
    batch['des_info']['cem_iters'] = cem_iters
    batch['des_info']['temperature'] = temperature
    batch['des_info']['seed'] = seed

    center_traj_lidar[:, :, 2] = 1.0
    center_traj_canvas = center_traj_lidar[:, :, :3] @ (torch.Tensor(lidar2canvas.T).to(device))[None, :, :] # (NUM_FRAMES, NUM_OBJS, 3)
    center_traj_canvas = center_traj_canvas.round().long() # (NUM_FRAMES, NUM_OBJS, 3)
    center_dist = center_traj_lidar[:, :, :2].norm(dim=2) # (NUM_FRAMES, NUM_OBJS)
    obj_instance_box_global = instance_boxes_global[:, obj_idx, :] # (NUM_FRAMES, 7)

    st_ts = -1
    for t in range(4, NUM_FRAMES): # AdvKDM needs 5 frames of history trajectory
        if classes[t, 0, obj_idx] == -1:
            continue
        if center_dist[t, obj_idx] < LAUNCH_THRESHOLD:
            if classes[t-4:t, 0, obj_idx].min() != -1:
                st_ts = t
                break
    if st_ts == -1 or st_ts > 140:
        return None
    st_ts = max(st_ts, 20)
    batch['des_info']['st_ts'] = st_ts
    batch['des_info']['avg_ego_dist'] = 0
    
    # start simulation
    cur_ts = st_ts
    lst_rep_ts = None
    best_trajectory_global = None
    instance_boxes_global[st_ts+1:, obj_idx, 3:6] = instance_boxes_global[st_ts, obj_idx, 3:6]
    rel_spd_ma = torch.zeros(2, device=device)
    rel_dist_ma = 0
    batch['des_info']['rel_pos'][st_ts - 1] = obj_instance_box_global[st_ts - 1, :2] - ego_traj_global[st_ts - 1, 0, :2]
    for cur_ts in tqdm.tqdm(range(st_ts, NUM_FRAMES - 1)):
        batch['des_info']['rel_pos'][cur_ts] = obj_instance_box_global[cur_ts, :2] - ego_traj_global[cur_ts, 0, :2]
        batch['des_info']['rel_spd'][cur_ts - 1] = \
            (batch['des_info']['rel_pos'][cur_ts] - batch['des_info']['rel_pos'][cur_ts - 1]) / 0.1
        rel_spd_ma = rel_spd_ma * 0.7 + batch['des_info']['rel_spd'][cur_ts - 1] * 0.3
        rel_dist_ma = rel_dist_ma * 0.7 + batch['des_info']['rel_pos'][cur_ts - 1].norm().cpu().item() * 0.3
        batch['des_info']['avg_ego_dist'] += batch['des_info']['rel_pos'][cur_ts].norm().cpu().item()
        ego_dis_ena = 0 if cur_ts > st_ts + 36 and rel_dist_ma < 5 and np.random.rand() < 0.5 else 1
        if (cur_ts - st_ts) % RE_PLAN_INTERVAL == 0:
            obj_history = build_obj_history(obj_instance_box_global, cur_ts) # ego_agent_features in original code
            batch_size = ES_POP_SIZE
            
            state_features = model.encode_scene_features(obj_history) # ((B, 5, D), (B, 5, D))
            state_features = (
                state_features[0].repeat_interleave(batch_size, dim=0),
                state_features[1].repeat_interleave(batch_size, dim=0)
            )
            obj_history = obj_history.repeat_interleave(batch_size, dim=0)

            trunc_step_schedule = np.linspace(5,1,cem_iters).astype(int)
            noise_scale = 1.0

            trajectory_shape = (batch_size, H * 3)
            
            # set up simulator config
            obj_x = obj_instance_box_global[cur_ts, 0].cpu().item()
            obj_y = obj_instance_box_global[cur_ts, 1].cpu().item()
            obj_heading = -obj_instance_box_global[cur_ts, 6].cpu().item() - math.pi / 2
            obj_x_size = obj_instance_box_global[cur_ts, 3].cpu().item()
            obj_y_size = obj_instance_box_global[cur_ts, 4].cpu().item()
            obj_z_size = obj_instance_box_global[cur_ts, 5].cpu().item()
            obj_param = VehicleParameters(obj_x_size, obj_y_size * 0.8,
                                          obj_y_size * 0.2, obj_y_size * 0.3,
                                          obj_y_size * 0.6, 'obj', 'obj', obj_z_size)
            obj_se2 = StateSE2(obj_x, obj_y, obj_heading)
            obj_carfp = CarFootprint(obj_se2, obj_param)
            obj_v2d = StateVector2D(((obj_instance_box_global[cur_ts, 0] - obj_instance_box_global[cur_ts - 2, 0]) / 2).cpu().item(),
                                    ((obj_instance_box_global[cur_ts, 1] - obj_instance_box_global[cur_ts - 2, 1]) / 2).cpu().item())
            obj_v2d.x = (obj_v2d.x ** 2 + obj_v2d.y ** 2) ** 0.5
            obj_v2d.y = 0.0
            obj_a2d = StateVector2D(0.0, 0.0)
            obj_dyn = DynamicCarState(obj_y_size * 0.3, obj_v2d, obj_a2d)
            obj_tp = TimePoint(cur_ts * 100000)
            obj_state = EgoState(obj_carfp, obj_dyn, 0, True, obj_tp)
            obj_simulator = PDMSimulator(TrajectorySampling(80, 80, 1.0))
            
            
            def score_fn(candi_traj: torch.Tensor):
                final_scores = torch.zeros_like(candi_traj[:, 0]) # (batch_size, ), lower is better
                final_scores += 0.02 + torch.rand_like(final_scores) * 1e-3
                final_info = {}

                candi_traj_int = torch.cat([torch.zeros_like(candi_traj.view(-1, H, 3)[:, :1, :]), candi_traj.view(-1, H, 3)], axis=1)
                candi_traj_int = interpolate_trajectory(candi_traj_int.cpu().numpy(), 81)
                
                stp_idx = np.random.randint(ES_POP_SIZE)
                
                candi_traj_int = fix_heading(candi_traj_int)
                candi_traj_global = convert_to_global_convert_heading(obj_state, candi_traj_int) # (batch_size, H, 3)

                candi_traj_global_padded = np.concatenate([candi_traj_global,
                                            np.zeros((candi_traj_global.shape[0], candi_traj_global.shape[1], 8))], axis=2)
                candi_traj_global_sim = obj_simulator.simulate_proposals(candi_traj_global_padded, obj_state)[..., :3]


                cpfra = obj_state._car_footprint._vehicle_parameters.cog_position_from_rear_axle
                candi_traj_global_sim[:, :, 0] += cpfra * np.cos(candi_traj_global_sim[:, :, 2])
                candi_traj_global_sim[:, :, 1] += cpfra * np.sin(candi_traj_global_sim[:, :, 2])
                final_info['candi_traj_global_sim'] = candi_traj_global_sim
                candi_traj_global = torch.tensor(candi_traj_global_sim) \
                                    .to(obj_instance_box_global.device, 
                                        dtype=obj_instance_box_global.dtype)
                corners_offset = torch.tensor([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])\
                                .to(candi_traj_global.device).to(candi_traj_global.dtype) # (4, 3)
                corners_offset[:, 0] *= obj_x_size / 2
                corners_offset[:, 1] *= obj_y_size / 2
                candi_traj_corners_global = candi_traj_global.clone() # (batch_size, 81, 3)
                
                candi_traj_corners_global = candi_traj_corners_global[:, :, None, :].repeat(1, 1, 4, 1) # (batch_size, 81, 4, 3)
                candi_traj_corners_global[..., 0] += corners_offset[None, None, :, 0] * \
                                                    torch.cos(candi_traj_global[:, :, None, 2])
                candi_traj_corners_global[..., 1] += corners_offset[None, None, :, 1] * \
                                                    torch.sin(candi_traj_global[:, :, None, 2])
                
                SCR_H = min(81, NUM_FRAMES - cur_ts)
                candi_traj_global_homo = torch.concat([candi_traj_global,
                                                       torch.ones_like(candi_traj_global[:, :, :1])], dim=2) 
                                                    # (batch_size, 81, 4)
                candi_traj_global_homo[:, :, 2] = 0.0

                candi_traj_lidar = candi_traj_global_homo @ \
                                   torch.inverse(ego2global[cur_ts].permute(1, 0)) @ \
                                   torch.inverse(lidar2ego[cur_ts].permute(1, 0)) # (batch_size, 81, 4)
                candi_traj_lidar[:, :, 2] = 1.0
                candi_traj_canvas = candi_traj_lidar[:, :, :3] @ \
                                    (torch.Tensor(lidar2canvas.T).to(candi_traj_lidar.device)[None, ...])
                                    # (batch_size, 81, 3)
                candi_traj_canvas = candi_traj_canvas.round().long() # (batch_size, 81, 3)
                candi_traj_canvas = torch.clamp(candi_traj_canvas, 0, canvas_h - 1)

                candi_traj_corners_global_homo = torch.concat(
                    [candi_traj_corners_global,
                     torch.ones_like(candi_traj_corners_global[..., :1])],
                    dim=-1
                )
                candi_traj_corners_global_homo[..., 2] = 0.0
                candi_traj_corners_lidar = candi_traj_corners_global_homo @ \
                                           torch.inverse(ego2global[cur_ts].permute(1, 0)) @ \
                                           torch.inverse(lidar2ego[cur_ts].permute(1, 0))
                candi_traj_corners_lidar[..., 2] = 1.0
                candi_traj_corners_canvas = candi_traj_corners_lidar[..., :3] @ \
                                            (torch.Tensor(lidar2canvas.T).to(candi_traj_lidar.device)[None, None, ...])
                candi_traj_corners_canvas = candi_traj_corners_canvas.round().long() # (batch, 81, 4, 3)
                candi_traj_corners_canvas = torch.clamp(candi_traj_corners_canvas, 0, canvas_h - 1)
                
                out_of_road = (1 - bev_map[0, cur_ts, 0, 
                                           candi_traj_canvas[:, :, 0], candi_traj_canvas[:, :, 1]].to(device))
                                # (batch_size, 81)
                num_corners_to_penalty = torch.tensor([0, 1, 1.1, 1.2, 1.3]).to(device)
                corners_out_of_road = (1 - bev_map[0, cur_ts, 0,
                                                   candi_traj_corners_canvas[..., 0], candi_traj_corners_canvas[..., 1]]
                                       .to(device)).sum(dim=-1).round().long() # (batch_size, 81)
                fst_out_of_road = out_of_road.argmax(dim=1)
                ever_out_of_road = out_of_road.any(dim=1)
                
                oor_step_weight = torch.zeros((81, ), device=device)
                oor_step_weight[:SCR_H] = torch.linspace(1.0, 0.5, SCR_H, device=device) ** 2 # (81, )
                oor_step_weight[RE_PLAN_INTERVAL + 2:] *= 0.03
                out_of_road_score = (out_of_road * oor_step_weight).sum(dim=1) * ever_out_of_road
                out_of_road_score += (num_corners_to_penalty[corners_out_of_road] * oor_step_weight * 0.2).sum(dim=1)
                final_scores += 5 * out_of_road_score


                col_dists = (candi_traj_global.permute(1, 0, 2))[:SCR_H, :, None, :2] - \
                            instance_boxes_global[cur_ts:cur_ts+SCR_H, None, :, :2]
                col_dists = col_dists.norm(dim=-1) # (SCR_H, batch_size, NUM_OBJS+1)
                must_col = (2 * col_dists < \
                            instance_boxes_global[cur_ts:cur_ts+SCR_H, None, :, 3:5].min(dim=-1).values + \
                            instance_boxes_global[cur_ts:cur_ts+SCR_H, None, obj_idx:obj_idx+1, 3:5].min(dim=-1).values)\
                            # (SCR_H, batch_size, NUM_OBJS+1)
                must_col[:, :, obj_idx] = False
                must_col = must_col.any(dim=-1) # (SCR_H, batch_size)
                
                impossible_col = (2 * col_dists > \
                                  instance_boxes_global[cur_ts:cur_ts+SCR_H, None, :, 3:5].max(dim=-1).values + \
                                  instance_boxes_global[cur_ts:cur_ts+SCR_H, None, obj_idx:obj_idx+1, 3:5].max(dim=-1).values)
                impossible_col[:, :, obj_idx] = True
                to_check = torch.logical_not(impossible_col) # (SCR_H, batch_size, NUM_OBJS+1)
                to_check = to_check & (torch.logical_not(must_col)[:, :, None])
                sample_mask = torch.zeros((SCR_H, 1, 1), device=device, dtype=torch.bool)
                sample_mask[::4, 0, 0] = True
                sample_mask[:RE_PLAN_INTERVAL + 2, 0, 0] = True
                if SCR_H > 4:
                    sample_mask[[1, 3], 0, 0] = True
                to_check = to_check & sample_mask
                to_check[RE_PLAN_INTERVAL + 13:, :, :] = False
                to_check = to_check.nonzero() # (num_check, 3)
                col = must_col.clone()
                ego_col = torch.zeros((batch_size, ), device=device, dtype=torch.bool)
                for i in range(to_check.shape[0]):
                    ts, traj_idx, col_obj = to_check[i]
                    if col[ts, traj_idx]:
                        continue
                    col_instance_box = LiDARInstance3DBoxes(instance_boxes_global[None, cur_ts + ts, col_obj, :])
                    col_instance_poly = sg.Polygon(col_instance_box.corners[0, [0, 3, 7, 4], :2].cpu().numpy())
                    candi_instance_box = LiDARInstance3DBoxes(torch.cat([candi_traj_global[traj_idx, ts, :2],
                                                                         torch.zeros_like(candi_traj_global[traj_idx, ts, 2:3]),
                                                                         obj_instance_box_global[cur_ts + ts, 3:6] * 1.2,
                                                                         -candi_traj_global[traj_idx, ts, 2:3] - math.pi / 2],
                                                                         dim=0)[None, ...])
                    candi_instance_poly = sg.Polygon(candi_instance_box.corners[0, [0, 3, 7, 4], :2].cpu().numpy())
                    if candi_instance_poly.intersects(col_instance_poly):
                        col[ts, traj_idx] = True
                        if col_obj == NUM_OBJS and ts <= RE_PLAN_INTERVAL + 1:
                            ego_col[traj_idx] = True
                col_ts = col.long().argmax(dim=0)
                ever_col = col.any(dim=0)
                col_penalty = torch.linspace(1.0, 0.5, 81, device=device) ** 2
                col_penalty = col_penalty[:SCR_H]
                col_penalty[RE_PLAN_INTERVAL + 1:] *= 0.03
                col_score = (col * col_penalty[:, None]).sum(dim=0) * ever_col
                final_scores += 10 * col_score

                
                # reward trajs that challenges ego
                ego_dist_weight = torch.zeros((81, ), device=device)
                ego_dist_weight[:RE_PLAN_INTERVAL + 1] = torch.linspace(0.7, 1.0, RE_PLAN_INTERVAL + 1, device=device)
                ego_dist_weight[RE_PLAN_INTERVAL + 1:] = torch.linspace(0.2, 0.05, 81 - RE_PLAN_INTERVAL - 1, device=device) ** 2
                dist_to_ego = (ego_traj_global[None, cur_ts:cur_ts+SCR_H, 0, :2] - candi_traj_global[:, :SCR_H, :2]).norm(dim=-1)
                dist_to_ego_score = ((dist_to_ego * ego_dist_weight[:SCR_H]) ** 2).sum(dim=-1)
                min_sc = dist_to_ego_score.min()
                max_sc = dist_to_ego_score.max()
                dist_to_ego_score[ego_col] = max_sc
                dist_to_ego_score = (dist_to_ego_score - min_sc) / (max_sc - min_sc + 1e-7)
                if not disable_ego_dis:
                    final_scores += dist_to_ego_score * ego_dis_ena
                    
                return final_scores, final_info
 

            noise = torch.randn(trajectory_shape, device=device)
            population_trajectories, population_scores, population_info = model.rollout(
                state_features=state_features,
                ego_trajectory=noise,
                ego_agent_features=obj_history,
                scorer_fn=score_fn,
                initial_rollout=True,
                deterministic=False,
            )

            for i in range(cem_iters):
                population_trajectories = model.standardizer.transform_features(obj_history, population_trajectories)
                n_trunc_steps = trunc_step_schedule[i]

                # Compute reward-probabilities
                reward_probs = torch.exp(temperature * -population_scores)
                reward_probs = reward_probs / reward_probs.sum()
                probs = reward_probs

                # Resample and mutate (renoise-denoise)
                indices = torch.multinomial(probs, batch_size, replacement=True) # torch.multinomial(probs, 1).squeeze(1)
                population_trajectories = population_trajectories[indices]
                population_trajectories = model.renoise(population_trajectories, n_trunc_steps)

                # Denoise
                # rta_bf_diff = time.time()
                population_trajectories, population_scores, population_info = model.rollout(
                    state_features=state_features,
                    ego_trajectory=population_trajectories,
                    ego_agent_features=obj_history,
                    # scorer_fn=fn,
                    scorer_fn=score_fn,
                    initial_rollout=False,
                    deterministic=False,
                    n_trunc_steps=n_trunc_steps,
                    noise_scale=noise_scale,
                )

                pop_traj_global = (population_info['candi_traj_global_sim'][:, ::5, :3])[:, 1:, :]
                pop_traj_global = torch.tensor(pop_traj_global).to(obj_instance_box_global.device, dtype=obj_instance_box_global.dtype).view(1, -1, 3)
                pop_traj_global = torch.concat([pop_traj_global, torch.ones_like(pop_traj_global[:, :, :1])], dim=2) # (1, batch_size * H, 4)
                pop_traj_global[:, :, 2] = 0.0
                pop_traj_global[:, :, 3] = 1.0
                cur_plan_hor = min(RE_PLAN_INTERVAL, NUM_FRAMES - cur_ts - 1)
                batch['des_info']['pop_traj_lidar_rds'][i, cur_ts + 1: cur_ts + cur_plan_hor + 1] = \
                    ((pop_traj_global @ torch.inverse(ego2global[cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)].permute(0, 2, 1)))
                    @ torch.inverse(lidar2ego[cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)].permute(0, 2, 1))).view(-1, batch_size, H, 4)[:, :, :, :2]
                # (NUM_FRAMES, ES_POP_SIZE, H, 2)
                batch['des_info']['pop_traj_score_rds'][i, cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)] = \
                    population_scores.view(-1, batch_size)

            
            pop_traj_global = (population_info['candi_traj_global_sim'][:, ::5, :3])[:, 1:, :]
            pop_traj_global = torch.tensor(pop_traj_global).to(obj_instance_box_global.device, dtype=obj_instance_box_global.dtype).view(1, -1, 3)
            pop_traj_global = torch.concat([pop_traj_global, torch.ones_like(pop_traj_global[:, :, :1])], dim=2) # (1, batch_size * H, 4)
            pop_traj_global[:, :, 2] = 0.0
            pop_traj_global[:, :, 3] = 1.0
            cur_plan_hor = min(RE_PLAN_INTERVAL, NUM_FRAMES - cur_ts - 1)
            batch['des_info']['pop_traj_lidar'][cur_ts + 1: cur_ts + cur_plan_hor + 1] = \
                ((pop_traj_global @ torch.inverse(ego2global[cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)].permute(0, 2, 1)))
                @ torch.inverse(lidar2ego[cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)].permute(0, 2, 1))).view(-1, batch_size, H, 4)[:, :, :, :2]
            # (NUM_FRAMES, ES_POP_SIZE, H, 2)
            batch['des_info']['pop_traj_score'][cur_ts + 1: min(cur_ts + RE_PLAN_INTERVAL + 1, NUM_FRAMES)] = \
                population_scores.view(-1, batch_size)
            best_trajectory_global = (population_info['candi_traj_global_sim'][population_scores.argmin()])[None, ...]
            best_trajectory_global = torch.tensor(best_trajectory_global).to(obj_instance_box_global.device, dtype=obj_instance_box_global.dtype)
            lst_rep_ts = cur_ts
            batch['des_info']['best_score'] += population_scores.min().cpu().item()
            if population_scores.min().cpu().item() > 7:
                batch['des_info']['bad_gen'] = True

        # update instance_box_global
        obj_instance_box_global[cur_ts + 1, :] = obj_instance_box_global[cur_ts, :]
        obj_instance_box_global[cur_ts + 1, [0, 1]] = best_trajectory_global[0, (cur_ts - lst_rep_ts) + 1, :2]
        obj_instance_box_global[cur_ts + 1, 6] = -best_trajectory_global[0, (cur_ts - lst_rep_ts) + 1, 2] - math.pi / 2
    
    new_obj_bbox = LiDARInstance3DBoxes(obj_instance_box_global)
    new_obj_corners_global = new_obj_bbox.corners # (NUM_FRAMES, 8, 3)
    new_obj_corners_global = torch.cat([new_obj_corners_global, torch.ones(NUM_FRAMES, 8, 1, device=device)], dim=2) # (NUM_FRAMES, 8, 4)
    new_obj_corners_lidar = new_obj_corners_global @ torch.inverse(ego2global.permute(0, 2, 1)) @ torch.inverse(lidar2ego.permute(0, 2, 1)) # (NUM_FRAMES, 8, 3)
    bboxes[:, 0, obj_idx, :, :] = new_obj_corners_lidar[:, :, :3]
    batch['bboxes_3d_data'][0].data['bboxes'] = bboxes.to(odevice)
    assert classes[st_ts, 0, obj_idx] != -1, "check classes!"
    batch['bboxes_3d_data'][0].data['classes'][st_ts:, 0, obj_idx] = classes[st_ts, 0, obj_idx]
    rebuild_mask(batch, obj_idx)
    to_ego_dists = batch['des_info']['rel_pos'].norm(dim=-1).sort()[0]
    if to_ego_dists[0] > 15 or batch['des_info']['rel_pos'].abs().max() > 50:
        batch['des_info']['bad_gen'] = True
    batch['des_info']['avg_ego_dist'] /= (NUM_FRAMES - st_ts)
    if batch['des_info']['bad_gen']:
        return None
    return batch

def prepare_batch_for_more_adv(batch):
    if 'pop_traj_lidar_pres' not in batch['des_info']:
        batch['des_info']['pop_traj_lidar_pres'] = []
        batch['des_info']['pop_traj_score_pres'] = []
        batch['des_info']['edited_obj_pres'] = []
    batch['des_info']['pop_traj_lidar_pres'].append(batch['des_info']['pop_traj_lidar'].clone().detach())
    batch['des_info']['pop_traj_score_pres'].append(batch['des_info']['pop_traj_score'].clone().detach())
    batch['des_info']['edited_obj_pres'].append(batch['des_info']['edited_obj'])
    return batch