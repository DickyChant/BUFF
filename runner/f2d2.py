"""Shortcut-F2D2 model utilities for BUFF experiments.

This module lifts the reusable parts of the notebook prototype into a
script-friendly implementation:
  - shortcut network with velocity and divergence heads
  - Algorithm 1 style training losses
  - fast approximate log-probability evaluation
  - shortcut ODE sampling, with optional self-guidance
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange


def exact_div_stopped(net_u, x, t):
    """Compute div_x(u_theta^-(x, t, t)) with exact autodiff.

    The returned divergence is detached, matching the stop-gradient target
    used in the notebook implementation.
    """
    batch_size, dim = x.shape
    x_leaf = x.detach().requires_grad_(True)

    v = net_u(x_leaf, t, t)
    div = torch.zeros(batch_size, device=x.device)

    for i in range(dim):
        selector = torch.zeros_like(v)
        selector[:, i] = 1.0
        grad = torch.autograd.grad(
            v,
            x_leaf,
            grad_outputs=selector,
            create_graph=False,
            retain_graph=(i < dim - 1),
        )[0]
        div = div + grad[:, i]

    return div.detach()


class ShortcutF2D2Net(nn.Module):
    """Shared backbone with velocity and accumulated-divergence heads."""

    def __init__(self, dim=2, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(dim + 2, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.backbone = nn.Sequential(*layers)
        self.u_head = nn.Linear(hidden, dim)
        self.d_head = nn.Linear(hidden, 1)

    def _embed(self, x, t, s):
        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.expand(batch_size)
        if s.dim() == 0:
            s = s.expand(batch_size)
        inputs = torch.cat([x, t.unsqueeze(-1), s.unsqueeze(-1)], dim=-1)
        return self.backbone(inputs)

    def u(self, x, t, s):
        return self.u_head(self._embed(x, t, s))

    def D(self, x, t, s):
        return self.d_head(self._embed(x, t, s)).squeeze(-1)


class ShortcutF2D2Model:
    """Convenience wrapper around the Shortcut-F2D2 network."""

    def __init__(self, dim=2, hidden=128, depth=4, lr=3e-4, device="cpu"):
        self.dim = dim
        self.device = torch.device(device)
        self.net = ShortcutF2D2Net(dim=dim, hidden=hidden, depth=depth).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=lr)

    def fit(self, X, epochs=600, batch_size=256, grad_clip=5.0, log_every=100):
        """Train on unconditional data using the notebook's shortcut losses."""
        if torch.is_tensor(X):
            data = X.detach().to(self.device, dtype=torch.float32)
        else:
            data = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.net.train()
        history = []

        for ep in trange(epochs, desc="training"):
            bs = min(batch_size, len(data))
            idx = torch.randperm(len(data), device=self.device)[:bs]
            x1 = data[idx]
            x0 = torch.randn_like(x1)

            ts = torch.rand(bs, 2, device=self.device)
            t = ts.min(dim=1).values
            s = ts.max(dim=1).values
            r = (t + s) / 2.0

            t_expand = t.unsqueeze(-1)
            x_t = (1 - t_expand) * x0 + t_expand * x1

            with torch.no_grad():
                x_r = x_t + (r - t).unsqueeze(-1) * self.net.u(x_t, t, r)

            u_tt = self.net.u(x_t, t, t)
            loss_vm = ((u_tt - (x1 - x0)) ** 2).mean()

            u_ts = self.net.u(x_t, t, s)
            with torch.no_grad():
                u_target = 0.5 * (
                    self.net.u(x_t, t, r) +
                    self.net.u(x_r, r, s)
                )
            loss_u = ((u_ts - u_target) ** 2).mean()

            div_u = exact_div_stopped(self.net.u, x_t, t)
            d_tt = self.net.D(x_t, t, t)
            loss_div = ((d_tt + div_u) ** 2).mean()

            d_ts = self.net.D(x_t, t, s)
            with torch.no_grad():
                d_target = 0.5 * (
                    self.net.D(x_t, t, r) +
                    self.net.D(x_r, r, s)
                )
            loss_d = ((d_ts - d_target) ** 2).mean()

            loss = loss_vm + loss_u + loss_div + loss_d

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=grad_clip)
            self.opt.step()

            record = {
                "loss": float(loss.item()),
                "loss_vm": float(loss_vm.item()),
                "loss_u": float(loss_u.item()),
                "loss_div": float(loss_div.item()),
                "loss_d": float(loss_d.item()),
            }
            history.append(record)

            if (ep + 1) % log_every == 0:
                print(
                    f"  epoch {ep + 1:4d} | loss={record['loss']:.5f} "
                    f"vm={record['loss_vm']:.4f} u={record['loss_u']:.4f} "
                    f"div={record['loss_div']:.4f} D={record['loss_d']:.4f}"
                )

        return history

    train = fit

    @torch.no_grad()
    def sample(self, n_samples, n_steps=4):
        """Sample using the forward shortcut ODE without guidance."""
        self.net.eval()
        x = torch.randn(n_samples, self.dim, device=self.device)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            batch_size = x.shape[0]
            t_i = ts[i].expand(batch_size)
            t_ip1 = ts[i + 1].expand(batch_size)
            x = x + (ts[i + 1] - ts[i]) * self.net.u(x, t_i, t_ip1)
        return x.cpu().numpy()

    def sample_with_guidance(self, n_samples=300, n_steps=4, lr_guidance=0.01, guidance_steps=1):
        """Sample with the notebook's self-guidance initialization."""
        self.net.eval()

        x0 = nn.Parameter(torch.randn(n_samples, self.dim, device=self.device))
        adam_x0 = Adam([x0], lr=lr_guidance)

        for _ in range(guidance_steps):
            d_01 = self.net.D(
                x0,
                torch.zeros(n_samples, device=self.device),
                torch.ones(n_samples, device=self.device),
            )
            log_p0 = -0.5 * (x0 ** 2 + math.log(2 * math.pi)).sum(dim=-1)
            loss_nll = (-log_p0 - d_01).mean()
            adam_x0.zero_grad()
            loss_nll.backward()
            adam_x0.step()

        x = x0.detach()
        with torch.no_grad():
            ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
            for i in trange(n_steps, desc="forward ODE", leave=False):
                batch_size = x.shape[0]
                t_i = ts[i].expand(batch_size)
                t_ip1 = ts[i + 1].expand(batch_size)
                x = x + (ts[i + 1] - ts[i]) * self.net.u(x, t_i, t_ip1)

        return x.cpu().numpy()

    def log_prob(self, X, n_steps=4):
        """Approximate log p(x) with the shortcut divergence head."""
        self.net.eval()
        if torch.is_tensor(X):
            x_orig = X.detach().to(self.device, dtype=torch.float32)
        else:
            x_orig = torch.tensor(X, dtype=torch.float32, device=self.device)
        batch_size = x_orig.shape[0]

        with torch.no_grad():
            d_01 = self.net.D(
                x_orig,
                torch.zeros(batch_size, device=self.device),
                torch.ones(batch_size, device=self.device),
            )

            x = x_orig.clone()
            ts = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
            for i in range(n_steps):
                t_i = ts[i].expand(batch_size)
                t_ip1 = ts[i + 1].expand(batch_size)
                u = self.net.u(x, t_i, t_ip1)
                x = x + (ts[i + 1] - ts[i]) * u

            log_pz = -0.5 * (x ** 2 + math.log(2 * math.pi)).sum(dim=-1)

        return (log_pz - d_01).cpu().numpy()

    def state_dict(self):
        return {
            "model_state": self.net.state_dict(),
            "dim": self.dim,
        }

    def load_state_dict(self, state):
        self.net.load_state_dict(state["model_state"])
        return self
