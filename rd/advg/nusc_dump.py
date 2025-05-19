import os
import torch
import numpy as np
import pickle
import uuid
import shutil
import json
from pyquaternion import Quaternion
import math
from torchvision.transforms import Resize, ToPILImage
from torchvision.utils import save_image
import signal

CHANNELS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
DEFAULT_CATEGORY_TOKEN = \
{
    0: 'fd69059b62a3469fbaef25340c0eab7f',
    1: '6021b5187b924d64be64a702e5570edf',
    2: '5b3cd6f2bca64b83aa3d0008df87d0e4',
    3: 'fedb11688db84088883945752e480c2c',
    4: '90d0f6f8e7c749149b1b6c3a029841a8',
    5: '653f7efbb9514ce7b81d44070d6208c1',
    6: 'dfd26f200ade4d24b540184e16050022',
    7: 'fc95c87b806f48f8a1faea2dcc2222a4',
    8: '1fa93b757fc74fb197cdd60001ad8abf',
    9: '85abebdccd4d46c7be428af5a6173947',
}
DEFAULT_ATTRIBUTE_TOKENS = \
{
    0: ['cb5118da1ab342aa947717dc53544259'],
    1: ['cb5118da1ab342aa947717dc53544259'],
    2: ['cb5118da1ab342aa947717dc53544259'],
    3: ['cb5118da1ab342aa947717dc53544259'],
    4: ['cb5118da1ab342aa947717dc53544259'],
    5: [],
    6: ['a14936d865eb4216b396adae8cb3939c'],
    7: ['a14936d865eb4216b396adae8cb3939c'],
    8: ['ab83627ff28b465b85c427162dec722f'],
    9: [],
}

CAN_BUS_LIST = ['_meta', '_ms_imu', '_pose', '_route', '_steeranglefeedback', '_vehicle_monitor', '_zoe_veh_info', '_zoesensors']

ts_to_ego_pose = None
ori_scenes = None
ori_logs = None
channel_to_token = {}
token_to_channel = {}
calibrated_sensors = None
samples = None
instances = None
sample_annotations = None
sample_datas = None
ego_poses = None
scenes = None


def new_token():
    return uuid.uuid4().hex


def dump_a_scene(batch, dump_ds_root, ori_ds_root, scene_suffix, samples_tensor, write_jsons=True):
    global ts_to_ego_pose, ori_scenes, ori_logs, channel_to_token, token_to_channel
    global calibrated_sensors, samples, instances, sample_annotations, sample_datas, ego_poses, scenes
    
    dmp_pth = os.path.join(dump_ds_root, 'v1.0-trainval')
    
    if ts_to_ego_pose is None:
        ori_scenes = json.load(open(os.path.join(ori_ds_root, 'v1.0-trainval', 'scene.json')))
        ori_logs = json.load(open(os.path.join(ori_ds_root, 'v1.0-trainval', 'log.json')))
        ts_to_ego_pose = {}
        ori_ego_poses = json.load(open(os.path.join(ori_ds_root, 'v1.0-trainval', 'ego_pose.json')))
        for ep in ori_ego_poses:
            ts_to_ego_pose[ep['timestamp']] = ep
        sensors = json.load(open(os.path.join(dmp_pth, 'sensor.json')))
        for sensor in sensors:
            token = sensor['token']
            channel = sensor['channel']
            channel_to_token[channel] = token
            token_to_channel[token] = channel

    fst_smp_token = batch['meta_data']['metas'][0][0].data['token']
    for scene in ori_scenes:
        if scene['first_sample_token'] == fst_smp_token:
            ori_scene = scene
            break
    for log in ori_logs:
        if log['token'] == ori_scene['log_token']:
            ori_log = log
            break
    
    # calibrated_sensor.json
    if calibrated_sensors is None:
        if os.path.exists(os.path.join(dmp_pth, 'calibrated_sensor.json')):
            calibrated_sensors = json.load(open(os.path.join(dmp_pth, 'calibrated_sensor.json')))
        else:
            calibrated_sensors = []
        
        
    CHANNEL_TO_CAL_TOKEN = {}
    item = {}
    item['token'] = new_token()
    item['sensor_token'] = channel_to_token['LIDAR_TOP']
    item['translation'] = batch['meta_data']['metas'][0][0].data['full_info']['lidar2ego_translation']
    item['rotation'] = batch['meta_data']['metas'][0][0].data['full_info']['lidar2ego_rotation']
    item['camera_intrinsic'] = []
    CHANNEL_TO_CAL_TOKEN['LIDAR_TOP'] = item['token']
    calibrated_sensors.append(item)
    
    for i, channel in enumerate(CHANNELS):
        item = {}
        item['token'] = new_token()
        item['sensor_token'] = channel_to_token[channel]
        item['translation'] = batch['meta_data']['metas'][0][0].data['full_info']['cams'][channel]['sensor2ego_translation']
        item['rotation'] = batch['meta_data']['metas'][0][0].data['full_info']['cams'][channel]['sensor2ego_rotation']
        item['camera_intrinsic'] = batch['meta_data']['metas'][0][0].data['full_info']['cams']['CAM_FRONT_LEFT']['camera_intrinsics'].tolist()
        calibrated_sensors.append(item)
        CHANNEL_TO_CAL_TOKEN[channel] = item['token']
        
        
    # sample.json
    if samples is None:
        if os.path.exists(os.path.join(dmp_pth, 'sample.json')):
            samples = json.load(open(os.path.join(dmp_pth, 'sample.json')))
        else:
            samples = []
    NUM_FRAMES = len(batch['meta_data']['metas'])
    NUM_SAMPLES = (NUM_FRAMES + 5) // 6
    sample_tokens = [new_token() for _ in range(NUM_SAMPLES)]
    new_scene_token = new_token()
    for i in range(NUM_SAMPLES):
        item = {}
        item['token'] = sample_tokens[i]
        item['timestamp'] = batch['meta_data']['metas'][i * 6][0].data['full_info']['timestamp']
        item['prev'] = sample_tokens[i - 1] if i > 0 else ''
        item['next'] = sample_tokens[i + 1] if i < NUM_SAMPLES - 1 else ''
        item['scene_token'] = new_scene_token
        samples.append(item)
    
    # scene.json
    if scenes is None:
        if os.path.exists(os.path.join(dmp_pth, 'scene.json')):
            scenes = json.load(open(os.path.join(dmp_pth, 'scene.json')))
        else:
            scenes = []
    item = {}
    item['token'] = new_scene_token
    item['log_token'] = ori_scene['log_token']
    item['nbr_samples'] = NUM_SAMPLES
    item['first_sample_token'] = sample_tokens[0]
    item['last_sample_token'] = sample_tokens[-1]
    item['name'] = ori_scene['name'] + scene_suffix
    item['description'] = ori_scene['description']
    scenes.append(item)
    
    # instance.json & sample_annotation
    if instances is None:
        if os.path.exists(os.path.join(dmp_pth, 'instance.json')):
            instances = json.load(open(os.path.join(dmp_pth, 'instance.json')))
        else:
            instances = []
    if sample_annotations is None:
        if os.path.exists(os.path.join(dmp_pth, 'sample_annotation.json')):
            sample_annotations = json.load(open(os.path.join(dmp_pth, 'sample_annotation.json')))
        else:
            sample_annotations = []
    NUM_OBJS = batch['bboxes_3d_data'][0].data['bboxes'].shape[2]
    inst_tokens = [new_token() for _ in range(NUM_OBJS)]
    for i in range(NUM_OBJS):
        valid_ts = (batch['bboxes_3d_data'][0].data['classes'][:, 0, i] != -1).nonzero().view(-1)
        if valid_ts.shape[0] == 0:
            continue
        fst_smp = (valid_ts[0].cpu().item() + 5) // 6
        lst_smp = valid_ts[-1].cpu().item() // 6
        if len(valid_ts) != valid_ts[-1] - valid_ts[0] + 1:
            continue
        # assert len(valid_ts) == valid_ts[-1] - valid_ts[0] + 1, "check ts!"
        assert fst_smp <= lst_smp, "check ts!"
        
        obj_cls = batch['bboxes_3d_data'][0].data['classes'][valid_ts[0], 0, i].item()
        anno_tokens = [new_token() for _ in range(lst_smp - fst_smp + 1)]
        
        inst_item = {}
        inst_item['token'] = inst_tokens[i]
        inst_item['category_token'] = DEFAULT_CATEGORY_TOKEN[obj_cls]
        inst_item['nbr_annotations'] = lst_smp - fst_smp + 1
        inst_item['first_annotation_token'] = anno_tokens[0]
        inst_item['last_annotation_token'] = anno_tokens[-1]
        instances.append(inst_item)
        
        for j in range(fst_smp, lst_smp + 1):
            smp_anno_item = {}
            smp_anno_item['token'] = anno_tokens[j - fst_smp]
            smp_anno_item['sample_token'] = sample_tokens[j]
            smp_anno_item['instance_token'] = inst_tokens[i]
            smp_anno_item['visibility_token'] = '3'
            smp_anno_item['attribute_tokens'] = DEFAULT_ATTRIBUTE_TOKENS[obj_cls]
            
            cor_pts_lidar = batch['bboxes_3d_data'][0].data['bboxes'][j * 6, 0, i] # (8, 3)
            cor_pts_lidar = torch.cat([cor_pts_lidar, torch.ones((8, 1), device=cor_pts_lidar.device)], dim=1) # (8, 4)
            lidar2ego = torch.eye(4, device=cor_pts_lidar.device)
            lidar2ego_q = Quaternion(batch['meta_data']['metas'][j * 6][0].data['full_info']['lidar2ego_rotation'])
            lidar2ego[:3, :3] = torch.Tensor(lidar2ego_q.rotation_matrix)
            lidar2ego[:3, 3] = torch.Tensor(batch['meta_data']['metas'][j * 6][0].data['full_info']['lidar2ego_translation'])
            ego2global = torch.eye(4, device=cor_pts_lidar.device)
            ego2global_q = Quaternion(batch['meta_data']['metas'][j * 6][0].data['full_info']['ego2global_rotation'])
            ego2global[:3, :3] = torch.Tensor(ego2global_q.rotation_matrix)
            ego2global[:3, 3] = torch.Tensor(batch['meta_data']['metas'][j * 6][0].data['full_info']['ego2global_translation'])
            lidar2global = torch.matmul(ego2global, lidar2ego)
            cor_pts_global = torch.matmul(lidar2global, cor_pts_lidar.T).T[:, :3] # (8, 3)
            center = cor_pts_global[[0, 3, 4, 7], :].mean(dim=0) # (x, y, z)
            smp_anno_item['translation'] = center.tolist()
            smp_anno_item['size'] = [(cor_pts_global[0] - cor_pts_global[4]).norm().item(), 
                                     (cor_pts_global[0] - cor_pts_global[3]).norm().item(), 
                                     (cor_pts_global[0] - cor_pts_global[1]).norm().item()]
            piv_pts = (cor_pts_global[4] + cor_pts_global[7]) / 2 - center # (3, )
            heading = -torch.atan2(piv_pts[1], piv_pts[0]).item() # (1, )
            heading = -heading - math.pi / 2
            if heading < 0:
                heading += math.pi * 2 * (int((-heading) / (math.pi * 2) + 1))
            else:
                heading -= math.pi * 2 * int(heading / (math.pi * 2))
            
            heading_q = Quaternion(axis=[0, 0, 1], angle=heading)
            smp_anno_item['rotation'] = [heading_q.w, heading_q.x, heading_q.y, heading_q.z]
            
            smp_anno_item['prev'] = anno_tokens[j - fst_smp - 1] if j > fst_smp else ''
            smp_anno_item['next'] = anno_tokens[j - fst_smp + 1] if j < lst_smp else ''
            smp_anno_item['num_lidar_pts'] = 1
            smp_anno_item['num_radar_pts'] = 1
            
            sample_annotations.append(smp_anno_item)
    
    # sample_data.json & ego_pose.json
    if sample_datas is None:
        if os.path.exists(os.path.join(dmp_pth, 'sample_data.json')):
            sample_datas = json.load(open(os.path.join(dmp_pth, 'sample_data.json')))
        else:
            sample_datas = []
    if ego_poses is None:
        if os.path.exists(os.path.join(dmp_pth, 'ego_pose.json')):
            ego_poses = json.load(open(os.path.join(dmp_pth, 'ego_pose.json')))
        else:
            ego_poses = []
    
    sd_items = []
    
    for frm_idx in range(NUM_FRAMES):
        lidar_ts = os.path.basename(batch['meta_data']['metas'][frm_idx][0].data['full_info']['lidar_path'])
        lidar_ts = lidar_ts.split('.')[-3]
        lidar_ts = lidar_ts.split('_')[-1]
        lidar_ts = int(lidar_ts)
        
        item = {}
        item['token'] = new_token()
        item['sample_token'] = sample_tokens[frm_idx // 6]
        item['ego_pose_token'] = item['token']
        ep_item = {}
        ep_item['token'] = item['ego_pose_token']
        ep_item['timestamp'] = lidar_ts
        ep_item['translation'] = ts_to_ego_pose[lidar_ts]['translation']
        ep_item['rotation'] = ts_to_ego_pose[lidar_ts]['rotation']
        ego_poses.append(ep_item)
        item['calibrated_sensor_token'] = CHANNEL_TO_CAL_TOKEN['LIDAR_TOP']
        item['timestamp'] = lidar_ts
        item['fileformat'] = 'pcd'
        item['is_key_frame'] = batch['meta_data']['metas'][frm_idx][0].data['full_info']['is_key_frame']
        item['height'] = 0
        item['width'] = 0
        old_filename = os.path.sep.join(batch['meta_data']['metas'][frm_idx][0].data['full_info']['lidar_path'].split(os.path.sep)[3:])
        item['filename'] = old_filename.replace('.pcd.bin', scene_suffix + '.pcd.bin')
        # shutil.copy(os.path.join(ori_ds_root, old_filename),
        #             os.path.join(dump_ds_root, item['filename']))
        with open(os.path.join(dump_ds_root, item['filename']), 'wb') as f:
            pass
        sd_items.append(item)
    
    for i in range(NUM_FRAMES):
        sd_items[i]['prev'] = sd_items[i - 1]['token'] if i > 0 else ''
        sd_items[i]['next'] = sd_items[i + 1]['token'] if i < NUM_FRAMES - 1 else ''
    
    sample_datas.extend(sd_items)

    resize = Resize((900, 1600))
    to_pil = ToPILImage()
    for cam_idx, channel in enumerate(CHANNELS):
        sd_items = []
        for frm_idx in range(NUM_FRAMES):
            cam_ts = os.path.basename(batch['meta_data']['metas'][frm_idx][0].data['full_info']['cams'][channel]['data_path'])
            cam_ts = cam_ts.split('.')[-2]
            cam_ts = cam_ts.split('_')[-1]
            cam_ts = int(cam_ts)
            
            item = {}
            item['token'] = new_token()
            item['sample_token'] = sample_tokens[frm_idx // 6]
            item['ego_pose_token'] = item['token']
            ep_item = {}
            ep_item['token'] = item['ego_pose_token']
            ep_item['timestamp'] = cam_ts
            ep_item['translation'] = ts_to_ego_pose[cam_ts]['translation']
            ep_item['rotation'] = ts_to_ego_pose[cam_ts]['rotation']
            ego_poses.append(ep_item)
            item['calibrated_sensor_token'] = CHANNEL_TO_CAL_TOKEN[channel]
            item['timestamp'] = cam_ts
            item['fileformat'] = 'jpg'
            item['is_key_frame'] = batch['meta_data']['metas'][frm_idx][0].data['full_info']['is_key_frame']
            item['height'] = 900
            item['width'] = 1600
            old_filename = os.path.sep.join(batch['meta_data']['metas'][frm_idx][0].data['full_info']['cams'][channel]['data_path'].split(os.path.sep)[3:])
            item['filename'] = old_filename.replace('.jpg', scene_suffix + '.jpg')

            img = samples_tensor[0, cam_idx, :, frm_idx] # (C, H, W)
            img = ((img + 1) * 128)
            img = torch.clamp(img, min=0, max=255).to(torch.uint8)
            img = to_pil(img)
            img = resize(img)
            img.save(os.path.join(dump_ds_root, item['filename']))
            
            sd_items.append(item)
            
        for i in range(NUM_FRAMES):
            sd_items[i]['prev'] = sd_items[i - 1]['token'] if i > 0 else ''
            sd_items[i]['next'] = sd_items[i + 1]['token'] if i < NUM_FRAMES - 1 else ''
        sample_datas.extend(sd_items)    
    
    for cb_ele in CAN_BUS_LIST:
        frm_pth = os.path.join(ori_ds_root, 'can_bus', ori_scene['name'] + cb_ele + '.json')
        to_pth = os.path.join(dump_ds_root, 'can_bus', ori_scene['name'] + scene_suffix + cb_ele + '.json')
        shutil.copy(frm_pth, to_pth)
    
    if not write_jsons:
        return
    
    def ignore_signal(signum, frame):
        pass

    print(f"Transaction begins")
    original_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, ignore_signal)
        json.dump(calibrated_sensors,
                open(os.path.join(dmp_pth, 'calibrated_sensor.json'), 'w'),
                indent=0)
        json.dump(samples,
                open(os.path.join(dmp_pth, 'sample.json'), 'w'),
                indent=0)
        json.dump(scenes,
                open(os.path.join(dmp_pth, 'scene.json'), 'w'),
                indent=0)
        json.dump(instances,
                open(os.path.join(dmp_pth, 'instance.json'), 'w'),
                indent=0)
        json.dump(sample_annotations,
                open(os.path.join(dmp_pth, 'sample_annotation.json'), 'w'),
                indent=0)
        json.dump(sample_datas,
                open(os.path.join(dmp_pth, 'sample_data.json'), 'w'),
                indent=0)
        json.dump(ego_poses,
                open(os.path.join(dmp_pth, 'ego_pose.json'), 'w'),
                indent=0)
    finally:
        signal.signal(signal.SIGINT, original_handler)
    print(f"Transaction ends")

def create_new_ds(new_ds_root, ori_ds_root):
    os.makedirs(new_ds_root, exist_ok=True)
    os.makedirs(os.path.join(new_ds_root, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(new_ds_root, 'sweeps'), exist_ok=True)
    os.makedirs(os.path.join(new_ds_root, 'v1.0-trainval'), exist_ok=True)
    os.makedirs(os.path.join(new_ds_root, 'can_bus'), exist_ok=True)
    shutil.copytree(os.path.join(ori_ds_root, 'maps'),
                     os.path.join(new_ds_root, 'maps'),
                     dirs_exist_ok=True)
    ori_pth = os.path.join(ori_ds_root, 'v1.0-trainval')
    new_pth = os.path.join(new_ds_root, 'v1.0-trainval')
    shutil.copy(os.path.join(ori_pth, 'attribute.json'),
                os.path.join(new_pth, 'attribute.json'))
    shutil.copy(os.path.join(ori_pth, 'category.json'),
                os.path.join(new_pth, 'category.json'))
    # shutil.copy(os.path.join(ori_pth, 'ego_pose.json'),
    #             os.path.join(new_pth, 'ego_pose.json'))
    shutil.copy(os.path.join(ori_pth, 'log.json'),
                os.path.join(new_pth, 'log.json'))
    shutil.copy(os.path.join(ori_pth, 'map.json'),
                os.path.join(new_pth, 'map.json'))
    shutil.copy(os.path.join(ori_pth, 'visibility.json'),
                os.path.join(new_pth, 'visibility.json'))
    ori_sensors = json.load(open(os.path.join(ori_pth, 'sensor.json')))
    new_sensors = []
    for sensor in ori_sensors:
        if 'RADAR' in sensor['channel']:
            continue
        new_sensors.append(sensor)
        os.makedirs(os.path.join(new_ds_root, 'samples', sensor['channel']), exist_ok=True)
        os.makedirs(os.path.join(new_ds_root, 'sweeps', sensor['channel']), exist_ok=True)
    json.dump(new_sensors,
              open(os.path.join(new_pth, 'sensor.json'), 'w'),
              indent=0)

    
    