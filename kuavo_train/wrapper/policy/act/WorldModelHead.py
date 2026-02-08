"""
World Model components for ACT: Dynamics Head + Counterfactual Consistency.

Dynamics head predicts next proprioceptive state from the ACT encoder
representation and current action. Counterfactual consistency ensures
sub-block decomposition of z is causally meaningful.

Design:
  - z_t = encoder_out[0].detach()  →  (B, dim_model=512)
  - Split z_t into n_z_blocks sub-blocks (default 4 × 128)
  - Dynamics: ŝ_{t+1} = f(z_t, a_t)  →  (B, state_dim=16)
  - CF: swap one sub-block between batch samples, enforce invariance on output
"""

import torch
import torch.nn as nn


class DynamicsHead(nn.Module):
    """Predicts next proprioceptive state from encoder representation and action.

    ŝ_{t+1} = f(z_t, a_t)

    Architecture: 3-layer MLP with LayerNorm + ReLU.
    Input:  concat(z_t, a_t) ∈ R^{z_dim + action_dim}
    Output: ŝ_{t+1} ∈ R^{state_dim}

    Args:
        z_dim:       Dimension of the encoder representation z_t (default 512)
        action_dim:  Dimension of the action (first step of chunk, default 16)
        state_dim:   Dimension of the proprioceptive state to predict (default 16)
        hidden_dim:  Hidden dimension of the MLP (default 256)
    """

    def __init__(
        self,
        z_dim: int = 512,
        action_dim: int = 16,
        state_dim: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Xavier init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, z_dim)      encoder representation (should be detached from encoder)
            a_t: (B, action_dim)  action at current timestep
        Returns:
            (B, state_dim) predicted next proprioceptive state
        """
        return self.net(torch.cat([z_t, a_t], dim=-1))


def compute_counterfactual_loss(
    dynamics_head: DynamicsHead,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    n_z_blocks: int = 4,
    swap_block_idx: int = 3,
    cf_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute counterfactual consistency loss.

    Protocol
    --------
    1. Split z_t into *n_z_blocks* equal sub-blocks along dim=-1.
    2. For the *swap_block_idx*-th block, swap between sample i and
       sample (i+1) % B  (cyclic shift within batch).
    3. Run dynamics_head on the modified z̃_t  →  ŝ̃_{t+1}.
    4. Compare with the original ŝ_{t+1} on dimensions selected by *cf_mask*.

    Invariance claim (default setup):
      Swapping z_scene (block 3) should NOT change the predicted robot
      joint state because the background has no immediate causal effect on
      proprioceptive dynamics.

    Args:
        dynamics_head:  The dynamics prediction MLP.
        z_t:            (B, D) encoder representation.  **Must** be detached.
        a_t:            (B, action_dim) current action (normalised).
        n_z_blocks:     Number of sub-blocks
                        (default 4: left_arm / right_arm / object / scene).
        swap_block_idx: Which block to swap (3 = scene).
        cf_mask:        (state_dim,) binary tensor.
                        1 → this output dim should remain unchanged after swap.
                        None → all dims are masked (ones).
    Returns:
        Scalar MSE-based consistency loss.
    """
    B, D = z_t.shape
    if B < 2:
        return torch.tensor(0.0, device=z_t.device, requires_grad=True)

    block_size = D // n_z_blocks
    assert D % n_z_blocks == 0, (
        f"z_dim ({D}) must be divisible by n_z_blocks ({n_z_blocks})"
    )

    # ---- original prediction (reference, no grad through this path) ----
    with torch.no_grad():
        s_hat_orig = dynamics_head(z_t, a_t)  # (B, state_dim)

    # ---- construct counterfactual z̃ by cyclic-shifting the target block ----
    perm = torch.roll(torch.arange(B, device=z_t.device), shifts=1)

    z_cf = z_t.clone()
    blk_start = swap_block_idx * block_size
    blk_end = blk_start + block_size
    z_cf[:, blk_start:blk_end] = z_t[perm, blk_start:blk_end]

    # ---- predict with counterfactual z ----
    s_hat_cf = dynamics_head(z_cf, a_t)  # (B, state_dim)

    # ---- consistency loss on masked dimensions ----
    diff = s_hat_cf - s_hat_orig  # (B, state_dim)
    if cf_mask is not None:
        diff = diff * cf_mask.unsqueeze(0)

    return (diff ** 2).mean()
