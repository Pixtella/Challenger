import numpy as np
import torch
import os
import pickle

channel_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

# frm_idx, ann_obj_id, points, labels
# points = np.array([[1300, 500]], dtype=np.float32)
# labels = np.array([1], np.int32)
# {'CAM_FRONT': [{}, {}]} ...
# class_to_seg = [0, 1, 2, 3, 4, 6, 7, 8]
class_to_seg = [0]
def dump_anchor_points(batch, dump_path):
    ret = {cn: [] for cn in channel_names}
    num_obj = batch['bboxes_3d_data'][0].data['classes'].shape[2]
    for obj_idx in range(num_obj):
        obj_cls = batch['bboxes_3d_data'][0].data['classes'][:, 0, obj_idx].max().item()
        assert obj_cls != -1, "Invalid object!"
        if obj_cls not in class_to_seg:
            continue
        rebuild_mask(batch, obj_idx)
        for cam_idx, cn in enumerate(channel_names):
            obj_mask = batch['bboxes_3d_data'][0].data['masks'][:, cam_idx, obj_idx] # (T,)
            obj_mask_shift_bk = torch.cat([torch.tensor([0], device=obj_mask.device), obj_mask[:-1]], dim=0)
            obj_mask_shift_fw = torch.cat([obj_mask[1:], torch.tensor([0], device=obj_mask.device)], dim=0)
            start_idx = ((obj_mask - obj_mask_shift_bk) == 1).nonzero()[:, 0]
            end_idx = ((obj_mask - obj_mask_shift_fw) == 1).nonzero()[:, 0]
            assert start_idx.shape == end_idx.shape, "Invalid mask!"
            for i in range(start_idx.shape[0]):
                start_idx_ = start_idx[i].item()
                end_idx_ = end_idx[i].item()
                pts = get_anchor_point(batch, start_idx_, obj_idx, cam_idx)
                ret[cn].append({'ann_frame_idx': start_idx_, 'ann_obj_id': obj_idx, 'points': pts, 'labels': [1], 'end_frame_idx': end_idx_})
    with open(dump_path, 'wb') as f:
        pickle.dump(ret, f)

def get_anchor_point(batch, frm_idx, obj_idx, cam_idx):
    pts = batch['bboxes_3d_data'][0].data['bboxes'][frm_idx, 0, obj_idx, :, :] # (8, 3)
    pts = torch.cat([pts, torch.ones(8, 1)], dim=1)
    prj = batch['meta_data']['lidar2image'][frm_idx][0].data[cam_idx] # (4, 4)
    pts_img = torch.matmul(prj, pts.T).T # (8, 4)
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    x = pts_img[:, 0].cpu().numpy().mean()
    y = pts_img[:, 1].cpu().numpy().mean()
    x = np.clip(x, 0, 1600)
    y = np.clip(y, 0, 900)
    return np.array([[x, y]], dtype=np.float32)
    

def rebuild_mask(batch, idx):
    T = batch['bboxes_3d_data'][0].data['bboxes'].shape[0]
    for t in range(T):
        pts = batch['bboxes_3d_data'][0].data['bboxes'][t, 0, idx, :, :] # (8, 3)
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
        batch['bboxes_3d_data'][0].data['masks'][t, :, idx] = mask.float()

def dump_perspective_center(batch, dump_path):

    bboxes: torch.Tensor = batch['bboxes_3d_data'][0].data['bboxes'] # (NUM_FRAMES, 1, NUM_OBJS, 8, 3)
    classes: torch.Tensor = batch['bboxes_3d_data'][0].data['classes'] # (NUM_FRAMES, 1, NUM_OBJS)
    masks: torch.Tensor = batch['bboxes_3d_data'][0].data['masks'] # (NUM_FRAMES, N_CAM, NUM_OBJS)

    NUM_FRAMES = bboxes.shape[0]
    NUM_OBJS = bboxes.shape[2]
    N_CAM = masks.shape[1]

    ret = torch.zeros((NUM_FRAMES, N_CAM, NUM_OBJS, 2))
    for obj_idx in range(NUM_OBJS):
        obj_cls = batch['bboxes_3d_data'][0].data['classes'][:, 0, obj_idx].max().item()
        assert obj_cls != -1, "Invalid object!"
        if obj_cls not in class_to_seg:
            continue
        for t in range(NUM_FRAMES):
            pts = bboxes[t, 0, obj_idx, [0, 3, 7, 4], :].mean(dim=0) # (3, )
            pts = torch.cat([pts, torch.ones(1)], dim=0) # (4, )
            prj = batch['meta_data']['lidar2image'][t][0].data # (Ncam, 4, 4)
            pts_img = torch.matmul(prj, pts[:, None]).permute(0, 2, 1) # (Ncam, 1, 4)
            mask = pts_img[:, :, 2] > 0.01
            pts_img = pts_img[:, :, :2] / pts_img[:, :, 2:3]
            mask_ = \
            torch.logical_and(torch.logical_and(pts_img[:, :, 0] > 0, pts_img[:, :, 0] < 1600), \
                            torch.logical_and(pts_img[:, :, 1] > 0, pts_img[:, :, 1] < 900))
            mask = torch.logical_and(mask, mask_) # (Ncam, 1)
            ret[t, :, obj_idx, :] = pts_img[:, 0, :2] * mask.float()
    with open(dump_path, 'wb') as f:
        pickle.dump(ret, f)