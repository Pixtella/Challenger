import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon, Patch
import torch
import io
from PIL import Image
from torchvision.io import write_video
import tqdm
import os
from joblib import Parallel, delayed
from einops import rearrange
from torchvision.transforms import Resize
from torchvision.utils import draw_keypoints
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm


def cat_6_views(imgs):
    imgs_up = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
    imgs_down = rearrange(imgs[3:], "NC C T H W -> C T H (NC W)")
    imgs = torch.cat([imgs_up, imgs_down], dim=2)
    return imgs

EMOJI_FONT_PATH = os.path.join(os.path.dirname(__file__), 'seguiemj.ttf')
emoji_font = fm.FontProperties(fname=EMOJI_FONT_PATH, size=14)


def frame_vis(batch, frm_idx, ego_mode=False, render_bev=False, ret_img=False, neat_img=False, save_pth=None, rich_bev=False, vis_rd=-1):
    assert not rich_bev or render_bev, 'rich_bev is only supported in render_bev mode'
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    _, _, NUM_OBJS = batch['bboxes_3d_data'][0].data['masks'].shape
    t = frm_idx
    for idx in range(NUM_OBJS + 1):
        obj_cls = batch['bboxes_3d_data'][0].data['classes'][t, 0, idx] if idx != NUM_OBJS else 0
        if idx != NUM_OBJS and obj_cls == -1:
            continue
        if idx != NUM_OBJS:
            points = batch['bboxes_3d_data'][0].data['bboxes'][t, 0, idx, :, :2] # (8, 2)
        else:
            points = torch.tensor([[-1, -2], [-1, 2], [1, 2], [1, -2], [-1, -2], [-1, 2], [1, 2], [1, -2]], dtype=torch.float32).to(batch['bboxes_3d_data'][0].data['bboxes'].device)
        if not ego_mode:
            points = torch.concatenate([points, torch.zeros(8, 2)], dim=1) # (8, 4)
            points[:, 3] = 1
            points = points @ batch['des_info']['lidar2ego'][t].T @ batch['des_info']['ego2global'][t].T
            points = points[:, :2]
        points = points.cpu().numpy()
        centroid = np.mean(points, axis=0)
        if centroid[0] < -50 or centroid[0] > 50 or centroid[1] < -50 or centroid[1] > 50:
            continue
        ordered_points = sorted(points, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))

        # Convert back to numpy array for plotting
        ordered_points = np.array(ordered_points)

        # Add the first point at the end to close the rectangle
        ordered_points = np.vstack([ordered_points, ordered_points[0]])

        if idx != NUM_OBJS and ('des_info' not in batch or idx != batch['des_info']['edited_obj']) \
            and ('des_info' not in batch or 'edited_obj_pres' not in batch['des_info'] or idx not in batch['des_info']['edited_obj_pres']):
            polygon = Polygon(ordered_points, closed=True, edgecolor='blue' if obj_cls <= 8 else '#e31a1c', facecolor='none', linewidth=1.5)
            ax.add_patch(polygon)
            ax.text(centroid[0], centroid[1], f'{idx}', fontsize=12, color='black')
            ax.scatter(centroid[0], centroid[1], color='white', s=5)
        elif 'des_info' in batch and idx == batch['des_info']['edited_obj'] or ('des_info' in batch and 'edited_obj_pres' in batch['des_info'] and idx in batch['des_info']['edited_obj_pres']):
            polygon = Polygon(ordered_points, closed=True, edgecolor='red', facecolor='none', linewidth=1.5)
            ax.add_patch(polygon)
            # ax.text(centroid[0], centroid[1], f'{idx}', fontsize=12, color='black')
            ax.scatter(centroid[0], centroid[1], color='white', s=5)
        else:
            polygon = Polygon(ordered_points, closed=True, edgecolor='green', facecolor='none', linewidth=1.5)
            ax.add_patch(polygon)
            ax.text(centroid[0], centroid[1], f'ego', fontsize=12, color='black')
            
    bv_legend = Patch(color='blue', label='🚗 Background Vehicles', fill=None)
    adv_legend = Patch(color='red', label='👿 Adversarial Vehicle(s)', fill=None)
    ego_legend = Patch(color='green', label='🚘 Ego Vehicle', fill=None)
    ax.legend(handles=[bv_legend, adv_legend, ego_legend], loc='lower right', prop=emoji_font)


    if 'des_info' in batch and 'pop_traj_lidar' in batch['des_info']:
        BATCH_SIZE = batch['des_info']['pop_traj_score'][t].shape[0]
        if not ego_mode:
            pop_traj = batch['des_info']['pop_traj_lidar'][t].view(-1, 2).cpu()
            pop_traj = torch.cat([pop_traj, torch.zeros(pop_traj.shape[0], 2)], dim=1)
            pop_traj[:, 3] = 1
            pop_traj = pop_traj @ batch['des_info']['lidar2ego'][t].T @ batch['des_info']['ego2global'][t].T
            pop_traj = pop_traj[:, :2].view(BATCH_SIZE, 16, 2).numpy()
        else:
            if vis_rd == -1:
                pop_traj = batch['des_info']['pop_traj_lidar'][t].view(BATCH_SIZE, 16, 2).cpu().numpy()
            else:
                pop_traj = batch['des_info']['pop_traj_lidar_rds'][vis_rd, t].view(BATCH_SIZE, 16, 2).cpu().numpy()
        if vis_rd == -1:
            pop_traj_score = batch['des_info']['pop_traj_score'][t].view(BATCH_SIZE).cpu().numpy()
        else:
            pop_traj_score = batch['des_info']['pop_traj_score_rds'][vis_rd, t].view(BATCH_SIZE).cpu().numpy()
        cmp = cm.get_cmap('ocean')
        sc_max = pop_traj_score.max()
        sc_min = pop_traj_score.min()
        norm = plt.Normalize(vmin=sc_min, vmax=sc_max)
        num_traj = pop_traj_score.shape[0]
        bst_idx = np.argmin(pop_traj_score)
        
        for tr in range(num_traj):
            if tr != bst_idx:
                ax.plot(pop_traj[tr, :, 0], pop_traj[tr, :, 1], color=cmp(norm(pop_traj_score[tr])), alpha=0.15, linewidth=1)
        if vis_rd == -1 or vis_rd == 4 or 'nd' in save_pth:
            ax.plot(pop_traj[bst_idx, :, 0], pop_traj[bst_idx, :, 1], color='orange', linewidth=2)
        sm = cm.ScalarMappable(cmap=cmp, norm=norm)
        sm.set_array([])
        if not neat_img:
            fig.colorbar(sm, ax=ax, orientation='vertical', label='Trajectory Score')
        
        if 'pop_traj_lidar_pres' in batch['des_info']:
            num_pre = len(batch['des_info']['pop_traj_lidar_pres'])
            for pre_a in range(num_pre):
                pop_traj = batch['des_info']['pop_traj_lidar_pres'][pre_a][t].view(BATCH_SIZE, 16, 2).cpu().numpy()
                pop_traj_score = batch['des_info']['pop_traj_score_pres'][pre_a][t].view(BATCH_SIZE).cpu().numpy()
                bst_idx = np.argmin(pop_traj_score)
                for tr in range(num_traj):
                    if tr != bst_idx:
                        ax.plot(pop_traj[tr, :, 0], pop_traj[tr, :, 1], color=cmp(norm(pop_traj_score[tr])), alpha=0.15, linewidth=1)
                ax.plot(pop_traj[bst_idx, :, 0], pop_traj[bst_idx, :, 1], color='orange', linewidth=2)
    
        
    if ego_mode:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

    if render_bev:
        bev_extent = [-50, 50, 50, -50]
        da_cmap = mcolors.LinearSegmentedColormap.from_list("da_cmap", ['#F0F0F0FF', '#A6CEE3FF'])
        ax.imshow(batch['bev_map_with_aux'][0, t, 0].T, extent=bev_extent, cmap=da_cmap)
        if rich_bev:
            div_cmap = mcolors.LinearSegmentedColormap.from_list("div_cmap", ['#00000000', '#7794A3FF'])
            ax.imshow(batch['bev_map_with_aux'][0, t, 5].T, extent=bev_extent, cmap=div_cmap)
        
    if neat_img:
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not ret_img:
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        if save_pth is not None:
            if vis_rd != -1 and batch['des_info']['st_ts'] > frm_idx:
                plt.close()
                return
            fig.savefig(save_pth, transparent=True)
        else:
            plt.show()
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = np.array(img)
        plt.close()
        return img
    plt.close()

def convert_ckpt(ckpt_in_path, ckpt_out_path):
    ckpt = torch.load(ckpt_in_path)
    ckpt = ckpt['state_dict']
    ks = list(ckpt.keys())
    for i in range(len(ks)):
        if ks[i].startswith('model.'):
            ckpt[ks[i][6:]] = ckpt.pop(ks[i])
    torch.save(ckpt, ckpt_out_path)
    

CON_LIST = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]
def sample_to_mp4(samples, output_path, with_bev=False, batch=None, obj_box=False, st_frm_idx=0, ed_frm_idx=-1, up_scale=False):
    if len(samples.shape) == 6:
        samples = samples[0]
    samples = samples.cpu()
    assert not with_bev or batch is not None, 'batch must be provided if with_bev is True'
    assert not obj_box or batch is not None, 'batch must be provided if obj_box is True'
    if up_scale:
        rz = Resize((samples.shape[3] * 2, samples.shape[4] * 2))
        new_sam = torch.zeros(samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3] * 2, samples.shape[4] * 2).to(samples.device)
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                new_sam[i, j] = rz(samples[i, j])
        samples = new_sam
    vid_height = samples.shape[3] * 2
    vid_width = samples.shape[4] * 3
    samples = ((samples + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8).cpu()
    NUM_FRAMES = samples.shape[2]
    if ed_frm_idx == -1:
        ed_frm_idx = NUM_FRAMES
    if obj_box:
        to_add_box_objs = []
        to_add_box_objs.append(batch['des_info']['edited_obj'])
        if 'edited_obj_pres' in batch['des_info']:
            to_add_box_objs.extend(batch['des_info']['edited_obj_pres'])
        for obj_idx in to_add_box_objs:
            for t in range(NUM_FRAMES):
                pts = batch['bboxes_3d_data'][0].data['bboxes'][t, 0, obj_idx, :, :] # (8, 3)
                z_size = pts[:, 2].max() - pts[:, 2].min()
                pts[:, 2] -= z_size / 2
                pts = torch.cat([pts, torch.ones(8, 1)], dim=1)
                prj = batch['meta_data']['lidar2image'][t][0].data # (Ncam, 4, 4)
                pts_img = torch.matmul(prj, pts.T).permute(0, 2, 1) # (Ncam, 8, 4)
                mask = pts_img[:, :, 2] > 0.01
                pts_img = pts_img[:, :, :2] / pts_img[:, :, 2:3]
                if up_scale:
                    pts_img = pts_img * 2
                for j in range(6):
                    if mask[j].sum() < 8:
                        continue
                    samples[j, :, t] = draw_keypoints(samples[j, :, t], pts_img[j:j+1] * 0.25, connectivity=CON_LIST, colors=(0, 255, 0), radius=1, width=1) 

    cat_samples = cat_6_views(samples).permute(1, 2, 3, 0) # (T, H, W, C)
    if with_bev:
        resize = Resize((vid_height, vid_height))
        bevs = []
        for i in tqdm.tqdm(range(NUM_FRAMES)):
            vis_img = frame_vis(batch, i, ego_mode=True, render_bev=True, ret_img=True, neat_img=True, rich_bev=True)
            vis_img = torch.tensor(vis_img).permute(2, 0, 1).unsqueeze(0)
            vis_img = resize(vis_img).to(samples.dtype)
            bevs.append(vis_img)
        bevs = torch.cat(bevs, dim=0) # (NUM_FRAMES, 3, vid_height, vid_height)
        bevs = bevs.permute(0, 2, 3, 1) # (NUM_FRAMES, vid_height, vid_height, 3)
        cat_samples = torch.cat([cat_samples, bevs], dim=2) 
    cat_samples = cat_samples[st_frm_idx:ed_frm_idx, ...]
    if ed_frm_idx != st_frm_idx + 1:
        if up_scale:
            write_video(output_path, cat_samples, fps=12, \
                        options={"crf": "33", "video_codec": "libx265", "preset": "veryslow"})
        else:
            write_video(output_path, cat_samples, fps=12)
    else:
        img = Image.fromarray(cat_samples[0].numpy())
        img.save(output_path)