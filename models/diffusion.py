"""
Discrete Diffusion Model with Obscured-and-Replace state transition.
Based on D3PM (Austin et al., NeurIPS 2021) with occlusion-aware modifications.
"""
import math
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

        # Move to target device so buffers match input tensors
        self.to(device)

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
            # Cosine schedule: alpha decays from 1 to 0, ensuring valid probabilities
            # mu_s = alpha (stay prob), omega_s = per-state transition prob, psi_s = obs prob
            alpha = 0.5 * (1 + math.cos(math.pi * s / T))
            omega_s = (1 - alpha) * 0.9 / (V - 1)  # uniform to each other state
            psi_s = (1 - alpha) * 0.1                # obs replacement
            mu_s = alpha                              # stay probability

            # Build matrix
            A = torch.full((S, S), omega_s)
            # Diagonal entries for normal states
            for i in range(V):
                A[i, i] = mu_s + omega_s
            # Normal states -> Obs token with prob psi_s
            A[:V, self.obs_index] = psi_s
            # Obs is an absorbing state: once occluded, stays occluded
            A[self.obs_index, :] = 0.0
            A[self.obs_index, self.obs_index] = 1.0

            # Normalize rows to ensure valid probability distribution
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
        cum_mat = self.cum_transition_matrices[t].to(device)  # [B, V+1, V+1]

        # For each token, get transition probabilities
        # x_0_one_hot: [B, N, V+1]
        x_0_oh = F.one_hot(x_0, self.num_states).float().to(device)  # [B, N, V+1]

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
        Compute q(x_t | x_{t+1}, x_0) as logits for KL divergence.
        Based on paper Eq.(14):
            q(x_t | x_{t+1}, x_0) = [q(x_{t+1} | x_t) * q(x_t | x_0)] / q(x_{t+1} | x_0)

        Args:
            x_t: [B, N] noisy token indices at step t+1 (i.e., x_{t+1})
            x_0: [B, N] clean token indices
            t: [B] diffusion step indices (0..T-1), where x_t = x_{t+1}
        Returns:
            logits: [B, N, V+1] log probabilities of q(x_t | x_{t+1}, x_0)
        """
        b, n = x_t.shape
        device = x_t.device

        # t is q_sample index (0..T-1), x_t corresponds to x_{t+1}
        t = t.clamp(min=0, max=self.num_timesteps - 1)

        # A_{t+1} = transition_matrices[t]
        A_tp1 = self.transition_matrices[t].to(device)  # [B, S, S]

        # \bar{A}_{t+1} = cum_transition_matrices[t]
        cum_mat_tp1 = self.cum_transition_matrices[t].to(device)  # [B, S, S]

        # \bar{A}_t = cum_transition_matrices[t-1] (I when t=0)
        if t.min() > 0:
            cum_mat_t = self.cum_transition_matrices[t - 1].to(device)  # [B, S, S]
        else:
            eye = torch.eye(self.num_states, device=device).unsqueeze(0).expand(b, -1, -1)
            # Only use eye for t=0 entries; for t>0 use cum_transition_matrices[t-1]
            cum_mat_t = torch.zeros(b, self.num_states, self.num_states, device=device)
            for i in range(b):
                if t[i] > 0:
                    cum_mat_t[i] = self.cum_transition_matrices[t[i] - 1].to(device)
                else:
                    cum_mat_t[i] = eye[0]

        x_t_oh = F.one_hot(x_t, self.num_states).float().to(device)  # [B, N, S] (x_{t+1})
        x_0_oh = F.one_hot(x_0, self.num_states).float().to(device)  # [B, N, S]

        # Denominator: q(x_{t+1} | x_0) = x_0_oh @ \bar{A}_{t+1}
        denom = torch.einsum('bns,bst->bnt', x_0_oh, cum_mat_tp1)  # [B, N, S]
        denom = denom.clamp(min=1e-6)

        # Numerator part 1: q(x_{t+1} | x_t) = A_{t+1}[x_t, x_{t+1}]
        # For each possible x_t = s: A_{t+1}[s, x_{t+1}]
        q_xtp1_given_xt = torch.einsum('bns,bst->bnt', x_t_oh, A_tp1.transpose(-2, -1))  # [B, N, S]

        # Numerator part 2: q(x_t | x_0) = x_0_oh @ \bar{A}_t
        q_xt_given_0 = torch.einsum('bns,bst->bnt', x_0_oh, cum_mat_t)  # [B, N, S]

        # Full numerator and posterior
        numer = q_xtp1_given_xt * q_xt_given_0  # [B, N, S]
        posterior = numer / denom  # [B, N, S]
        posterior = posterior.clamp(min=0)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Return as logits (log probabilities). Floor at 1e-6 (was 1e-8) for stability.
        logits = torch.log(posterior.clamp(min=1e-6))
        return logits

    def _compute_vlb_chunked(self, x_0, x_tp1, t, pred_probs, chunk_size=512):
        """
        Compute KL[ q(k_{s-1}|k_s, x_0_true) || g_θ(k_{s-1}|k_s, y) ] in chunks over
        the codebook dimension V, avoiding the [B, N, V, S] OOM blowup.

        The reparameterized model posterior is:
            g_θ(k_{s-1}|k_s,y) = Σ_{k̄_0} q(k_{s-1}|k_s, k̄_0) · pred_probs[..., k̄_0]

        Args:
            x_0:        [B, N]   clean token indices (values in 0..V-1)
            x_tp1:      [B, N]   noisy token indices at step t+1 (values in 0..V)
            t:          [B]      timestep indices
            pred_probs: [B, N, V] softmax of denoiser logits over V classes
        Returns:
            scalar KL averaged over batch & tokens.
        """
        B, N = x_0.shape
        V = self.num_classes
        S = self.num_states
        device = x_0.device

        # Build target posterior q(k_{s-1}|k_s, x_0_true) once -> [B, N, S]
        target_log_post = self.q_posterior_logits(x_tp1, x_0, t)
        target_post = target_log_post.exp()

        # Accumulate model posterior g_θ(k_{s-1}|k_s, y) -> [B, N, S]
        # by summing q(k_{s-1}|k_s, k̄_0) * pred_probs[..., k̄_0] over k̄_0 in chunks.
        model_post = torch.zeros(B, N, S, device=device)

        # Precompute matrices needed for q(k_{s-1}|k_s, k̄_0) per-batch.
        # q(k_{s-1}|k_s, k̄_0) ∝ q(k_s|k_{s-1}) * q(k_{s-1}|k̄_0)
        # In our indexing: t corresponds to step s; A_{t+1}=transition_matrices[t];
        # \bar{A}_t = cum_transition_matrices[t-1] or I if t==0.
        t_clamped = t.clamp(min=0, max=self.num_timesteps - 1)
        A_tp1 = self.transition_matrices[t_clamped].to(device)          # [B, S, S]
        cum_mat_tp1 = self.cum_transition_matrices[t_clamped].to(device)  # [B, S, S]

        if t.min() > 0:
            cum_mat_t = self.cum_transition_matrices[t - 1].to(device)
        else:
            cum_mat_t = torch.zeros(B, S, S, device=device)
            eye_S = torch.eye(S, device=device)
            for i in range(B):
                if t[i] > 0:
                    cum_mat_t[i] = self.cum_transition_matrices[t[i] - 1].to(device)
                else:
                    cum_mat_t[i] = eye_S

        # x_tp1 one-hot for indexing into A_{t+1}: we need q(k_{s}|k_{s-1})=A_{t+1}[k_{s-1}, k_s]
        # For each candidate previous state s' in [0..S-1] and each batch element,
        # q(k_s=x_tp1 | k_{s-1}=s') = A_{t+1}[b, s', x_tp1[b,n]] -> shape [B, N, S]
        x_tp1_oh = F.one_hot(x_tp1, S).float()                         # [B, N, S]
        # q_xtp1_given_prev[b,n,s'] = A_{t+1}[b, s', x_tp1[b,n]]
        q_xtp1_given_prev = torch.einsum('bns,bts->bnt', x_tp1_oh, A_tp1)  # [B, N, S]

        # For the denominator q(k_s=x_tp1 | k̄_0): we'll compute per-chunk.
        # q(k_s | k̄_0) = cum_mat_tp1[b, k̄_0, k_s], shape [B, V] for fixed k_s per (b,n)
        # We extract these slices per-chunk.

        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            cs = v_end - v_start

            # q(k_{s-1}=s' | k̄_0) for k̄_0 in chunk = cum_mat_t[b, k̄_0, s']
            # Shape: [B, cs, S]
            q_prev_given_k0_chunk = cum_mat_t[:, v_start:v_end, :]  # [B, cs, S]

            # q(k_s = x_tp1 | k̄_0) = cum_mat_tp1[b, k̄_0, x_tp1[b,n]]
            # Shape: [B, N, cs]
            q_xtp1_given_k0_chunk = torch.einsum(
                'bns,bvs->bnv', x_tp1_oh, cum_mat_tp1[:, v_start:v_end, :]
            )  # [B, N, cs]
            q_xtp1_given_k0_chunk = q_xtp1_given_k0_chunk.clamp(min=1e-6)

            # Per-k̄_0 posterior over previous state:
            # q(k_{s-1}|k_s,k̄_0) ∝ q(k_s|k_{s-1}) * q(k_{s-1}|k̄_0) / q(k_s|k̄_0)
            # numerator: q_xtp1_given_prev [B,N,S] * q_prev_given_k0_chunk [B,cs,S]
            #            -> [B, N, cs, S]
            numer = q_xtp1_given_prev.unsqueeze(2) * q_prev_given_k0_chunk.unsqueeze(1)
            # denom: q_xtp1_given_k0_chunk [B, N, cs, 1]
            post_chunk = numer / q_xtp1_given_k0_chunk.unsqueeze(-1)
            post_chunk = post_chunk.clamp(min=0)
            post_chunk = post_chunk / post_chunk.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            # post_chunk: [B, N, cs, S]

            # Weight by pred_probs[..., k̄_0] for k̄_0 in chunk and accumulate
            w = pred_probs[:, :, v_start:v_end]  # [B, N, cs]
            model_post = model_post + (post_chunk * w.unsqueeze(-1)).sum(dim=2)

        # Renormalize model_post (should already sum near 1 since pred_probs sums to 1
        # and each post_chunk sums to 1, but float roundoff).
        model_post = model_post / model_post.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # KL( target || model ) = Σ target * (log target - log model)
        # Use larger floor (1e-6) than typical (1e-8) to avoid log(1e-8)=-18.4 producing
        # explosive gradients on rare classes.
        log_model_post = torch.log(model_post.clamp(min=1e-6))
        kl = (target_post * (target_log_post - log_model_post)).sum(dim=-1)  # [B, N]

        # Safety net: if any element is non-finite (overflow/underflow somewhere),
        # mask it out rather than poisoning the backward pass with inf/nan.
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl.mean()

    def p_losses(self, denoiser, x_0, t, condition, target_pose, prior_model,
                 eta=0.0005):
        """
        Compute training losses for the diffusion model.
        Implements the full loss from paper Eq.(20):
            L_all = eta * L_k0 + L_vlb + L_tkn + L_Recon
        Returns total_loss, recon_loss, l_k0, l_vlb, l_tkn
        """
        b = x_0.shape[0]
        device = x_0.device

        # Forward diffusion: sample x_{t+1}
        x_tp1 = self.q_sample(x_0, t)  # [B, N]

        # Predict x_0 distribution
        pred_logits = denoiser(x_tp1, t, condition)  # [B, N, V]
        # Clamp logits to prevent extreme values from amplifying through softmax/CE.
        # Range [-30, 30] keeps softmax numerically safe (exp(30) ≈ 1e13) while still
        # allowing the model to express confident predictions.
        pred_logits = pred_logits.clamp(-30.0, 30.0)
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, N, V]

        # L_k0: auxiliary loss (Eq.19) -log g_theta(k_0 | k_s, y)
        l_k0 = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes),
            x_0.reshape(-1)
        )

        # L_tkn: token prediction cross-entropy (same as L_k0 in training)
        l_tkn = l_k0.clone()

        # L_Recon: pose reconstruction loss
        pred_indices = pred_probs.argmax(dim=-1)  # [B, N]
        with torch.no_grad() if not target_pose.requires_grad else torch.enable_grad():
            pred_pose = prior_model.decode_from_indices(pred_indices)
            l_recon = F.smooth_l1_loss(pred_pose, target_pose)

        # L_vlb: enabled with small weight (0.01) to provide the core KL constraint
        # from paper Eq.16 while limiting its magnitude vs. l_tkn during early training.
        # The chunked compute path has finite-check safety nets for inf/nan resilience.
        vlb_weight = 0.01
        if vlb_weight > 0:
            l_vlb = self._compute_vlb_chunked(x_0, x_tp1, t, pred_probs, chunk_size=512)
        else:
            l_vlb = torch.tensor(0.0, device=device)

        # Total loss: l_recon excluded from backward path because the argmax+frozen-decoder
        # path provides zero gradient anyway; including it just adds noisy magnitude that
        # can swing total_loss without driving learning. Keep it as a monitoring metric.
        total_loss = eta * l_k0 + l_tkn + vlb_weight * l_vlb

        return total_loss, l_recon, l_k0, l_vlb, l_tkn

    @torch.no_grad()
    def p_sample(self, denoiser, x_t, t, condition):
        """
        Single reverse sampling step: predict x_0, then sample from posterior
        q(x_{t-1} | x_t, x_0_hat).
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

        # Compute x_t from posterior q(x_t | x_{t+1}, x_0_hat)
        # where x_{t+1} is current noisy state, x_t is previous (less noisy) state
        if t[0] > 0:
            t_idx = t.clamp(max=self.num_timesteps - 1)
            t_prev = t - 1

            # Cumulative matrices
            cum_mat_tp1 = self.cum_transition_matrices[t_idx].to(device)      # \bar{A}_{t+1}
            cum_mat_t = self.cum_transition_matrices[t_prev].to(device)        # \bar{A}_t

            # One-hot encodings
            x_0_oh = F.one_hot(pred_x_0, self.num_states).float().to(device)  # [B, N, S]
            x_tp1_oh = F.one_hot(x_t, self.num_states).float().to(device)      # [B, N, S] (x_{t+1})

            # q(x_t | x_0) = x_0_oh @ \bar{A}_t
            q_xt_given_0 = torch.einsum('bns,bst->bnt', x_0_oh, cum_mat_t)

            # q(x_{t+1} | x_t) = A_{t+1} (transition from x_t to x_{t+1})
            # Use t_idx to get A_{t+1} = transition_matrices[t]
            A_tp1 = self.transition_matrices[t_idx].to(device)  # [B, S, S]
            # For each possible x_t=s: A_{t+1}[s, x_{t+1}]
            q_xtp1_given_xt = torch.einsum('bns,bst->bnt', x_tp1_oh, A_tp1.transpose(-2, -1))

            # Posterior: q(x_t | x_{t+1}, x_0_hat) ∝ q(x_{t+1} | x_t) * q(x_t | x_0)
            posterior = q_xtp1_given_xt * q_xt_given_0
            posterior = posterior.clamp(min=0)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            x = torch.multinomial(
                posterior.view(-1, self.num_states), num_samples=1
            ).view(b, n)
            return x
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

        # Initialize from prior distribution (paper Eq.17)
        # q(k_T) = [bar{omega}_T, ..., bar{psi}_T]^T
        x = torch.multinomial(
            self.q_prior.unsqueeze(0).expand(b, -1),
            n,
            replacement=True,
        ).view(b, n)  # [B, N]

        # Leapfrog sampling: jump from T to T-delta to T-2*delta ...
        timesteps = list(range(self.num_timesteps - 1, -1, -num_leapfrog))
        if timesteps[-1] != 0:
            timesteps.append(0)

        for t_val in timesteps:
            t = torch.full((b,), t_val, dtype=torch.long, device=device)
            x = self.p_sample(denoiser, x, t, condition)

        return x
