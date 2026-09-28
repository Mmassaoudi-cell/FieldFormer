"""A compact selective state-space sequence layer (simplified Mamba/S4D-style
diagonal recurrence), used as Candidate A's temporal backbone in place of the
source method's Stacked LSTM. `mamba_ssm` is not installed in this
environment (its CUDA kernels are non-trivial to build here), so this is a
minimal, pure-PyTorch diagonal selective-SSM: O(N) in sequence length, no
attention matrix, with input-dependent (selective) discretization step,
following the core recurrence idea of Gu & Dao 2023 (Mamba) at a much
smaller, from-scratch-trainable scale appropriate for 32-packet windows.
"""
import torch
import torch.nn as nn


class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # per-channel diagonal state matrix (learned, negative for stability)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.9 + 0.1))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.ones(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, N, d_model)
        B, N, D = x.shape
        u = self.act(self.in_proj(x))
        dt = torch.nn.functional.softplus(self.dt_proj(u))  # (B,N,D) input-dependent step size (selective)
        A = -torch.exp(self.A_log)  # (D, d_state), stable (negative real)
        Bc = self.B_proj(u)  # (B,N,d_state)
        Cc = self.C_proj(u)  # (B,N,d_state)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,N,D,d_state)
        dB = dt.unsqueeze(-1) * Bc.unsqueeze(2)  # (B,N,D,d_state)

        state = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(N):
            state = dA[:, t] * state + dB[:, t] * u[:, t].unsqueeze(-1)
            y_t = (state * Cc[:, t].unsqueeze(1)).sum(-1)  # (B,D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B,N,D)
        y = y + u * self.D
        return self.out_proj(y)


class SSMBackbone(nn.Module):
    """Stack of SelectiveSSMLayer blocks with residual + LayerNorm, playing
    the same role as the Stacked LSTM in the source paper's Fig. 3/4, but
    O(N) and attention-free."""
    def __init__(self, d_in, d_model=128, n_layers=2, d_state=16):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.layers = nn.ModuleList([SelectiveSSMLayer(d_model, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x):
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        return h  # (B, N, d_model) -- full sequence output, matching LSTM's per-step output
