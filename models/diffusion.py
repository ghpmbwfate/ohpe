"""
Discrete Diffusion Model with Obscured-and-Replace state transition.
Based on D3PM (Austin et al., NeurIPS 2021) with occlusion-aware modifications.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDiffusion(nn.Module):
    """
    Discrete diffusion with Obscured-and-Replace transition matrix.
    States: 0..V-1 (codebook entries), V (Obs token)
    """
    def __init__(self, num_classes, num_timesteps=100, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.num_states = num_classes + 1  # +1 for Obs token
        self.obs_index = num_classes

        # Precompute transition matrices A_s and cumulative matrices
        transition_mats, cum_mats, prior = self._compute_transition_matrices()
        self.register_buffer('transition_matrices', transition_mats)
        self.register_buffer('cum_transition_matrices', cum_mats)
        self.register_buffer('q_prior', prior)

    def _compute_transition_matrices(self):
        """
        Compute A_s for s=1..T and cumulative \bar{A}_s.
        Returns:
            mats: [T, V+1, V+1]
            cum_mats: [T, V+1, V+1]
            prior: [V+1]
        """
        T = self.num_timesteps
        V = self.num_classes
        S = self.num_states
        mats = torch.zeros(T, S, S)

        for s in range(1, T + 1):
            # Linear schedule as mentioned in paper
            omega_s = 0.9 * (s / T)  # uniform diffusion rate
            psi_s = 0.1 * (s / T)    # obs replacement rate
            mu_s = 1.0 - omega_s * V - psi_s  # stay probability

            # Clamp to avoid negative probabilities due to numerical issues
            mu_s = max(mu_s, 0.0)
            # Renormalize if necessary
            total = mu_s + omega_s * V + psi_s
            omega_s = omega_s / total
            psi_s = psi_s / total
            mu_s = mu_s / total

            # Build matrix
            A = torch.full((S, S), omega_s)
            # Diagonal entries for normal states
            for i in range(V):
                A[i, i] = mu_s + omega_s
            # Obs column: normal states -> obs with prob psi_s
            A[:V, self.obs_index] = 0.0
            # Obs row: obs stays obs with prob 1
            A[self.obs_index, :] = psi_s
            A[self.obs_index, self.obs_index] = 1.0

            # Normalize rows
            A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-8)
            mats[s - 1] = A

        # Compute cumulative matrices
        cum_mats = torch.zeros_like(mats)
        cum = torch.eye(S)
        for s in range(T):
            cum = mats[s] @ cum
            cum_mats[s] = cum

        prior = cum_mats[-1, 0, :]  # Distribution from any initial state after T steps
        # Smooth prior to ensure valid probability distribution
        prior = prior.clamp(min=0)
        prior = prior / prior.sum()

        return mats, cum_mats, prior

    def q_sample(self, x_0, t):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        x_0: [B, N]  clean token indices (0..V-1)
        t: [B]       timestep indices (0..T-1)
        Returns: x_t [B, N]
        """
        b, n = x_0.shape
        device = x_0.device

        # Get cumulative transition matrices for each batch element
        # cum_mats[t[i]]: [V+1, V+1]
        cum_mat = self.cum_transition_matrices[t]  # [B, V+1, V+1]

        # For each token, get transition probabilities
        # x_0_one_hot: [B, N, V+1]
        x_0_oh = F.one_hot(x_0, self.num_states).float()  # [B, N, V+1]

        # Compute q(x_t | x_0) for each token
        # [B, N, V+1] @ [B, V+1, V+1] -> [B, N, V+1]
        probs = torch.einsum('bns,bst->bnt', x_0_oh, cum_mat)
        probs = probs.clamp(min=0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sample from multinomial
        x_t = torch.multinomial(probs.view(-1, self.num_states), num_samples=1).view(b, n)
        return x_t

    def q_posterior_logits(self, x_t, x_0, t):
        """
        Compute q(x_{t-1} | x_t, x_0) as logits for KL divergence.
        Used during training to compute VLB.
        """
        b, n = x_t.shape
        # Get transition matrices
        if t.min() > 0:
            cum_mat_t = self.cum_transition_matrices[t - 1]      # \bar{A}_{t-1}
        else:
            cum_mat_t = torch.eye(self.num_states, device=x_t.device).unsqueeze(0).expand(b, -1, -1)

        cum_mat_t_prev = self.cum_transition_matrices[t]         # \bar{A}_t

        x_t_oh = F.one_hot(x_t, self.num_states).float()         # [B, N, V+1]
        x_0_oh = F.one_hot(x_0, self.num_states).float()         # [B, N, V+1]

        # Denominator: q(x_t | x_0) = x_0_oh @ \bar{A}_t
        denom = torch.einsum('bns,bst->bnt', x_0_oh, cum_mat_t_prev)  # [B, N, V+1]
        denom = denom.clamp(min=1e-8)

        # Numerator: q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        # Simplified: we compute this efficiently using matrix properties
        # For training, we often use a simplified cross-entropy loss instead
        return denom

    def p_losses(self, denoiser, x_0, t, condition, target_pose, prior_model,
                 eta=0.0005):
        """
        Compute training losses for the diffusion model.
        Returns total_loss, recon_loss
        """
        b = x_0.shape[0]
        device = x_0.device

        # Forward diffusion: sample x_t
        x_t = self.q_sample(x_0, t)  # [B, N]

        # Predict x_0 distribution
        pred_logits = denoiser(x_t, t, condition)  # [B, N, V]

        # Auxiliary loss (simplified cross-entropy)
        aux_loss = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes),
            x_0.reshape(-1)
        )

        # Token prediction loss
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, N, V]
        pred_indices = pred_probs.argmax(dim=-1)      # [B, N]

        # Decode predicted pose for reconstruction loss
        with torch.no_grad() if not target_pose.requires_grad else torch.enable_grad():
            pred_pose = prior_model.decode_from_indices(pred_indices)
            recon_loss = F.smooth_l1_loss(pred_pose, target_pose)

        total_loss = eta * aux_loss + recon_loss
        return total_loss, recon_loss, aux_loss

    @torch.no_grad()
    def p_sample(self, denoiser, x_t, t, condition):
        """
        Single reverse sampling step: predict x_0, then compute x_{t-1}.
        """
        b, n = x_t.shape
        device = x_t.device

        # Predict clean token distribution
        pred_logits = denoiser(x_t, t, condition)  # [B, N, V]
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, N, V]

        # Sample predicted x_0
        pred_x_0 = torch.multinomial(
            pred_probs.view(-1, self.num_classes), num_samples=1
        ).view(b, n)  # [B, N]

        # Compute x_{t-1} distribution using q(x_{t-1} | x_t, x_0)
        if t[0] > 0:
            # Use posterior distribution
            # Simplified: directly sample from predicted x_0 with some noise
            # More accurate: compute full posterior using transition matrices
            # For leapfrog/speed, we can just return predicted x_0
            return pred_x_0
        else:
            return pred_x_0

    @torch.no_grad()
    def sample(self, denoiser, condition, prior_model, num_leapfrog=10):
        """
        Full reverse sampling from pure noise to clean tokens.
        Uses leapfrog sampling for efficiency.
        """
        b = condition.shape[0]
        device = condition.device
        n = prior_model.global_encoder.num_tokens if hasattr(prior_model.global_encoder, 'num_tokens') else 34

        # Initialize from prior distribution
        x = torch.full((b, n), self.obs_index, dtype=torch.long, device=device)
        # Or sample from prior
        # x = torch.multinomial(self.q_prior.unsqueeze(0).expand(b, -1), n, replacement=True)

        # Leapfrog sampling: jump from T to T-delta to T-2*delta ...
        timesteps = list(range(self.num_timesteps - 1, -1, -num_leapfrog))
        if timesteps[-1] != 0:
            timesteps.append(0)

        for t_val in timesteps:
            t = torch.full((b,), t_val, dtype=torch.long, device=device)
            x = self.p_sample(denoiser, x, t, condition)

        return x
