import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
import sys 
import os
sys.path.insert(0, os.path.join("..", "..", "exeu"))
from .flow_cnf import get_point_cnf
from .flow_cnf import get_latent_cnf
from utils.CNFutils import truncated_normal, reduce_tensor, standard_normal_logprob


# Model
class PointFlow(nn.Module):
    def __init__(self, args, input_dim, hidden_dims, context_dim, num_blocks, conditional):
        super(PointFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        self.num_blocks = num_blocks
        self.conditional = conditional

        self.point_cnf = get_point_cnf(args, input_dim, hidden_dims, context_dim, num_blocks, True)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_cnf = f(self.point_cnf)
        self.latent_cnf = f(self.latent_cnf)

    def forward(self, x, context):
        # opt.zero_grad()
        batch_size = x.size(0)
        num_points = 1
        # z_mu, z_sigma = self.encoder(x)
        # if self.use_deterministic_encoder:
        #     z = z_mu + 0 * z_sigma
        # else:
        #     z = self.reparameterize_gaussian(z_mu, z_sigma)

        # # Compute H[Q(z|X)]
        # if self.use_deterministic_encoder:
        #     entropy = torch.zeros(batch_size).to(z)
        # else:
        #     entropy = self.gaussian_entropy(z_sigma)

        # # Compute the prior probability P(z)
        # if self.use_latent_flow:
        #     w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
        #     log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
        #     delta_log_pw = delta_log_pw.view(batch_size, 1)
        #     log_pz = log_pw - delta_log_pw
        # else:
        #     log_pz = torch.zeros(batch_size, 1).to(z)

        # # Compute the reconstruction likelihood P(X|z)
        # z_new = z.view(*z.size())
        # z_new = z_new + (log_pz * 0.).mean()
        # print("x", x.size(), "context", context.size())
        x_new = x.view(batch_size, 1, self.input_dim)
        context_new = context.view(batch_size, 1, self.context_dim)
        y, delta_log_py = self.point_cnf(x_new, context_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        recon_loss = -log_px.mean()
        return recon_loss

    def log_prob(self, x, context):
        loss = self.forward(x, context)
        return loss

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, num_points, context, truncate_std=None, truncate_std_latent=None, gpu=0):
        batch_size = context.size(0)
        # Generate the shape code from the prior
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, context, reverse=True).view(*y.size())
        return x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std)
        return x