import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from diffusers import DDIMScheduler
import math
from typing import cast
from advtg.nnp import *
import datetime
import pickle


def fix_frictions_and_headings(traj: torch.Tensor):
    # traj: (batch_size, H * 3) in local frame
    # steps at the beginning of each trajectory are usually irregular, interpolate them
    batch_size = traj.shape[0]
    traj = traj.reshape(batch_size, -1, 3) # (batch_size, H, 3)
    # traj[:, 0, :] = 0.0
    valid_mask = traj[:, :, 0].abs() > 0.5
    valid_idxs = valid_mask.long().argmax(dim=1) # (batch_size,)
    for i in range(batch_size):
        # int_input = traj[i:i+1, [0, valid_idxs[i]], :].permute(0, 2, 1) # (1, 3, 2)
        # int_output = F.interpolate(int_input, size=valid_idxs[i] + 1, mode='linear') # (1, 3, vi)
        # traj[i:i+1, :valid_idxs[i] + 1, :] = int_output.permute(0, 2, 1)
        wei = torch.arange(1, valid_idxs[i] + 1, device=traj.device).float() / (valid_idxs[i] + 1) # (vi,)
        traj[i, :valid_idxs[i], :] = traj[i, valid_idxs[i], :] * wei[:, None] 
    
    # traj: (B, H, 3)
    HOR = traj.shape[1]
    traj[:, 1:HOR-1, 2] = torch.atan2(traj[:, 2:HOR, 1] - traj[:, 0:HOR-2, 1], traj[:, 2:HOR, 0] - traj[:, 0:HOR-2, 0])
    traj[:, 0, 2] = traj[:, 1, 2]
    traj[:, HOR-1, 2] = traj[:, HOR-2, 2]

    traj = traj.reshape(batch_size, -1)
    return traj

def aug_traj(traj: torch.Tensor):
    traj = traj.reshape(traj.shape[0], -1, 3)
    # traj: (B, H, 3)
    BATCH_SIZE = traj.shape[0]
    SLICE_SIZE = BATCH_SIZE // 4
    scale = torch.rand((SLICE_SIZE, 1), device=traj.device) * 0.6 + 0.2 # [0.2, 0.8]
    traj[1*SLICE_SIZE:2*SLICE_SIZE, :, 0] *= scale
    scale = torch.rand((SLICE_SIZE, 1), device=traj.device) * 0.6 + 1.2 # [1.2, 1.8]
    traj[2*SLICE_SIZE:3*SLICE_SIZE, :, 0] *= scale
    scale = torch.rand((SLICE_SIZE, 1), device=traj.device) + 0.5 # [0.5, 1.5]
    traj[3*SLICE_SIZE:4*SLICE_SIZE, :, 0:2] *= scale[..., None]
    # idx = np.random.randint(0, BATCH_SIZE)
    # traj[idx] *= 0.01 # add a stop option
    return traj.reshape(traj.shape[0], -1)


class AdvKinematicDiffusionModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        T: int = 100,
        predictions_per_sample: int = 16,
        max_dist: float = 200,           # used to normalize ALL tokens (incl. trajectory)
        easy_validation: bool = False,  # instead of starting from pure noise, start with not that much noise at inference,
        use_verlet: bool = True,
        ignore_history: bool = True,
        ckpt_path: str = None
    ):
        super(AdvKinematicDiffusionModel, self).__init__()
    
        self.feature_dim = feature_dim
        self.T = T
        self.H = 16
        self.output_dim = self.H * 3
        self.predictions_per_sample = predictions_per_sample
        self.max_dist = max_dist
        self.easy_validation = easy_validation
        self.use_verlet = use_verlet
        self.ignore_history = ignore_history

        self.standardizer = VerletStandardizer()

        # DIFFUSER
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.T,
            beta_schedule='scaled_linear',
            prediction_type='epsilon',
        )
        self.scheduler.set_timesteps(self.T)

        self.history_encoder = nn.Sequential(
            nn.Linear(7, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.sigma_proj_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

        self.trajectory_encoder = nn.Linear(3, self.feature_dim)
        self.trajectory_time_embeddings = RotaryPositionEncoding(self.feature_dim)
        self.type_embedding = nn.Embedding(3, self.feature_dim) # trajectory, noise token

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True
            )
        for _ in range(8)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 3)
        )

        self.all_trajs = []

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path))
        else:
            self.apply(self._init_weights)

        self.precompute_variances()

    def precompute_variances(self):
        """
        Precompute variances from alphas
        """
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[self.scheduler.timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[self.scheduler.timesteps]) ** 0.5
        self._variances = sqrt_one_minus_alpha_prod / sqrt_alpha_prod

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode_scene_features(self, his_ego_trajectory):
        if self.ignore_history:
            his_ego_trajectory = torch.zeros_like(his_ego_trajectory)
        ego_features = self.history_encoder(his_ego_trajectory) # Bx5x7 -> Bx5xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],5,1)

        return ego_features, ego_type_embedding

    def denoise(self, ego_trajectory, sigma, state_features):
        batch_size = ego_trajectory.shape[0]

        state_features, state_type_embedding = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        trajectory_type_embedding = self.type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,self.H,1)

        # Concatenate all features
        all_features = torch.cat([state_features, trajectory_features], dim=1)
        all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)

        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self.T
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        # Generate attention mask
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        dists = (indices[None] - indices[:,None]).abs()
        attn_mask = dists > 1       # TODO: magic number

        # Generate relative temporal embeddings
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size,1))

        # Global self-attentions
        for layer in self.global_attention_layers:            
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=all_type_embedding,
                attn_mask_11=attn_mask
            )

        trajectory_features = all_features[:,-self.H:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out # , all_weights
    def rollout(
        self,
        state_features,
        ego_trajectory,
        ego_agent_features,
        scorer_fn,
        initial_rollout=True, 
        deterministic=True, 
        n_trunc_steps=5, 
        noise_scale=1.0, 
        ablate_diffusion=False
    ):
        if initial_rollout:
            timesteps = self.scheduler.timesteps
        else:
            timesteps = self.scheduler.timesteps[-n_trunc_steps:]

        if ablate_diffusion and not initial_rollout:
            timesteps = []

        for t in timesteps:
            residual = torch.zeros_like(ego_trajectory)

            with torch.no_grad():
                residual += self.denoise(ego_trajectory, t.to(ego_trajectory.device), state_features)

            if deterministic:
                eta = 0.0
            else:
                prev_alpha = self.scheduler.alphas[t-1]
                alpha = self.scheduler.alphas[t]
                eta = noise_scale * torch.sqrt((1 - prev_alpha) / (1 - alpha)) * \
                        torch.sqrt((1 - alpha) / prev_alpha)

            out = self.scheduler.step(residual, t, ego_trajectory, eta=eta)
            ego_trajectory = out.prev_sample

        ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)
        if initial_rollout:
            ego_trajectory = aug_traj(ego_trajectory)
        scores, info = scorer_fn(ego_trajectory)

        return ego_trajectory, scores, info

    def renoise(self, ego_trajectory, t):
        noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
        ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, self.scheduler.timesteps[-t])
        return ego_trajectory