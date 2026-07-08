"""
saffron.sae
===========
Matryoshka / (Batch)TopK sparse autoencoder for decomposing dense embeddings (or
gene expression data directly) into sparse, human-interpretable features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SPARSITY_MODES = ("topk", "batchtopk", "kl", "l1")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_sparsify(h: torch.Tensor, k: int):
    """Keep only the top-k values per row (per sample); zero the rest."""
    vals, idx = torch.topk(h, k, dim=1)
    z = torch.zeros_like(h)
    z.scatter_(1, idx, vals)
    return z, idx, vals


def batchtopk_sparsify(h: torch.Tensor, k: int):
    """Keep only the top k*batch_size values globally across the batch; zero the rest.

    Unlike per-sample TopK, the budget k*B is shared across all samples so some
    spots may fire more or fewer than k neurons depending on activation magnitude.
    """
    B, M = h.shape
    total_k = min(k * B, B * M)
    flat = h.reshape(-1)
    _, global_idx = torch.topk(flat, total_k)
    mask = torch.zeros_like(flat)
    mask.scatter_(0, global_idx, 1.0)
    z = h * mask.reshape(B, M)
    vals, idx = torch.topk(z, k, dim=1)
    return z, idx, vals


def usage_stats_from_z(z: torch.Tensor) -> torch.Tensor:
    """Fraction of the batch where each neuron fires."""
    return (z > 0).float().mean(dim=0)


def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float64)
    if np.all(x <= 0):
        return 0.0
    x = np.clip(x, 0, None)
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    denom = cumx[-1] + eps
    return float(1.0 - 2.0 * np.sum(cumx / denom) / n + 1.0 / n)


def effective_num_neurons(usage: np.ndarray, eps: float = 1e-12) -> float:
    u = np.clip(usage.astype(np.float64), 0, None)
    s = u.sum()
    if s <= eps:
        return 0.0
    p = u / s
    H = -np.sum(p * np.log(p + eps))
    return float(np.exp(H))


def kl_sparsity_penalty(p_hat: torch.Tensor, rho: float, eps: float = 1e-7) -> torch.Tensor:
    """Bernoulli KL divergence KL(rho || p_hat) summed over neurons."""
    p = p_hat.clamp(eps, 1.0 - eps)
    return (rho * (rho / p).log() + (1 - rho) * ((1 - rho) / (1 - p)).log()).sum()


def usage_regularization_cap(z: torch.Tensor, usage_cap: float, min_floor: float = 1e-4):
    """Penalize only neurons whose usage exceeds usage_cap."""
    usage = usage_stats_from_z(z)
    cap = torch.clamp(torch.full_like(usage, usage_cap), min=min_floor)
    penalty = torch.mean(F.relu(usage - cap) ** 2)
    return penalty, usage


def proportional_matryoshka_weights(nested_dims: list[int]) -> list[float]:
    """Sqrt-proportional Matryoshka weights: smaller scales get relatively more
    weight than pure scale-proportional, so small prefixes still learn something
    meaningful."""
    total = sum(math.sqrt(m) for m in nested_dims)
    return [math.sqrt(m) / total for m in nested_dims]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """Weight-tied (Batch)TopK sparse autoencoder with optional Matryoshka training.

    Encoder : h = ReLU(x W^T + b_enc)
    Decoder : x_hat = z W_dec^T + b_dec     (W_dec is W, optionally unit-normed per row)

    sparsity_mode:
      "topk"      — exactly k active features per sample.
      "batchtopk" — a budget of k*batch_size active entries shared across the batch
                    (the default).
      "kl"        — soft ReLU activations during training with a Bernoulli KL sparsity
                    penalty; hard top-(rho*M) gating at inference time.
      "l1"        — soft ReLU activations with an L1 penalty on the code.

    matryoshka_dims: e.g. [64, 128, 256] where 256 == latent_dim. When set, nested
    prefixes of the code are trained to reconstruct the input independently (each
    with its own decoder bias), so early dimensions of the code capture coarse,
    dominant sources of variation and later dimensions capture fine-grained variation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int,
        sparsity_mode: str = "batchtopk",
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
        kl_target: float = 0.05,
        normalize_decoder_only: bool = True,
    ):
        super().__init__()
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"sparsity_mode must be one of {SPARSITY_MODES}, got {sparsity_mode!r}")
        self.sparsity_mode = sparsity_mode
        self.k = k
        self.kl_target = kl_target
        self.latent_dim = latent_dim
        self.normalize_decoder_only = normalize_decoder_only

        self.W = nn.Parameter(torch.empty(latent_dim, input_dim))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        if matryoshka_dims is not None:
            assert latent_dim in matryoshka_dims, "matryoshka_dims must include latent_dim"
            assert all(m <= latent_dim for m in matryoshka_dims)
            nested = [m for m in sorted(matryoshka_dims) if m < latent_dim]
            self.matryoshka_dims = sorted(matryoshka_dims)
            self.matryoshka_weights = (
                list(matryoshka_weights) if matryoshka_weights is not None
                else proportional_matryoshka_weights(nested)
            )
            if matryoshka_weights is not None:
                assert len(matryoshka_weights) == len(nested), \
                    "matryoshka_weights must have one entry per nested (non-full) scale"
            self.b_dec_nested = nn.ParameterList(
                [nn.Parameter(torch.zeros(input_dim)) for _ in nested]
            )
        else:
            self.matryoshka_dims = None
            self.matryoshka_weights = None
            self.b_dec_nested = nn.ParameterList([])

        # Feature standardization stats, populated by `fit`/`train_sae`.
        self.register_buffer("mu", torch.zeros(1, input_dim))
        self.register_buffer("sd", torch.ones(1, input_dim))

    def _decode_scale(self, z_m: torch.Tensor, m: int, b_dec_m: torch.Tensor) -> torch.Tensor:
        W_dec = F.normalize(self.W, p=2, dim=1) if self.normalize_decoder_only else self.W
        return F.linear(z_m, W_dec[:m].t(), b_dec_m)

    def forward(self, x: torch.Tensor, noise_std: float = 0.0,
                return_l1: bool = False, return_h: bool = False):
        pre_act = F.linear(x, self.W, self.b_enc)
        h = F.relu(pre_act)
        if noise_std > 0:
            h = h + noise_std * torch.randn_like(h)

        if self.sparsity_mode == "kl":
            if self.training:
                z = h  # soft during training so gradients flow for the KL penalty
            else:
                k_infer = max(1, round(self.kl_target * h.shape[1]))
                _, top_idx = torch.topk(h, k_infer, dim=1)
                mask = torch.zeros_like(h)
                mask.scatter_(1, top_idx, 1.0)
                z = h * mask
            k_compat = min(self.k, h.shape[1])
            vals, idx = torch.topk(z.detach(), k_compat, dim=1)
        elif self.sparsity_mode == "l1":
            if self.training:
                z = h  # soft during training so gradients flow for the L1 penalty
                vals, idx = torch.topk(h.detach(), self.k, dim=1)
            else:
                z, idx, vals = topk_sparsify(h, self.k)
        else:
            sparsify = topk_sparsify if self.sparsity_mode == "topk" else batchtopk_sparsify
            z, idx, vals = sparsify(h, self.k)

        W_dec = F.normalize(self.W, p=2, dim=1) if self.normalize_decoder_only else self.W
        x_hat = F.linear(z, W_dec.t(), self.b_dec)
        if return_l1:
            l1 = z.abs().sum(dim=1).mean()
            if return_h:
                return x_hat, z, idx, vals, l1, pre_act
            return x_hat, z, idx, vals, l1
        if return_h:
            return x_hat, z, idx, vals, pre_act
        return x_hat, z, idx, vals

    def matryoshka_recons(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Reconstructions for each nested scale (excluding the full scale)."""
        nested = [m for m in self.matryoshka_dims if m < self.latent_dim]
        recons = []
        for i, m in enumerate(nested):
            if self.sparsity_mode == "kl":
                if self.training:
                    z_m = h[:, :m]
                else:
                    k_m = max(1, round(self.kl_target * m))
                    _, top_idx = torch.topk(h[:, :m], k_m, dim=1)
                    mask = torch.zeros_like(h[:, :m])
                    mask.scatter_(1, top_idx, 1.0)
                    z_m = h[:, :m] * mask
            elif self.sparsity_mode == "l1":
                z_m = h[:, :m]
            else:
                k_m = max(1, round(self.k * m / self.latent_dim))
                z_m, _, _ = topk_sparsify(h[:, :m], k_m)
            recons.append(self._decode_scale(z_m, m, self.b_dec_nested[i]))
        return recons

    @torch.no_grad()
    def resample_dead(self, dead_mask: torch.Tensor, bias: float = 0.0) -> int:
        dead_idx = torch.where(dead_mask)[0]
        if dead_idx.numel() == 0:
            return 0
        for j in dead_idx.tolist():
            nn.init.kaiming_uniform_(self.W[j:j + 1], a=np.sqrt(5))
        self.b_enc[dead_idx] = bias
        return int(dead_idx.numel())

    @torch.no_grad()
    def encode(self, X: np.ndarray, device: str | None = None, batch_size: int = 4096) -> np.ndarray:
        """Standardize `X` with the stored train-split stats and return sparse codes."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device).eval()
        X = (np.asarray(X, dtype=np.float32) - self.mu.cpu().numpy()) / self.sd.cpu().numpy()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                             batch_size=batch_size, shuffle=False)
        Z = []
        for (xb,) in loader:
            _, z, _, _ = self.forward(xb.to(device), noise_std=0.0)
            Z.append(z.detach().cpu())
        return torch.cat(Z, dim=0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SAEResult:
    """Everything produced by :func:`train_sae`."""
    model: SparseAutoencoder
    Z: np.ndarray                      # (N, latent_dim) sparse codes for the full input
    history: dict = field(default_factory=dict)
    train_idx: np.ndarray = None
    val_idx: np.ndarray = None


def train_sae(
    X: np.ndarray,
    latent_dim: int | None = None,
    k: int | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    usage_weight: float = 0.5,
    usage_cap_mult: float = 3.0,
    l1_weight: float = 0.0,
    weight_decay: float = 0.0,
    feature_standardize: bool = True,
    normalize_decoder_only: bool = True,
    val_frac: float = 0.1,
    dead_thresh: float = 1e-6,
    resample_every: int = 10,
    resample_frac_max: float = 0.10,
    resample_bias: float = 0.0,
    noise_std_start: float = 0.02,
    noise_std_end: float = 0.0,
    matryoshka_dims: list[int] | None = None,
    matryoshka_weight: float = 1.0,
    matryoshka_weights: list[float] | None = None,
    sparsity_mode: str = "batchtopk",
    kl_target: float = 0.05,
    kl_weight: float = 1.0,
    kl_ema: float = 0.99,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> SAEResult:
    """Train a Matryoshka/(Batch)TopK SAE on an arbitrary embedding matrix `X`.

    `X` can be dense foundation-model embeddings or raw (log-normalized) gene
    expression — the same training routine works for either.

    For D-dim input, `latent_dim` defaults to `4*D` and `k` defaults to
    `round(0.06 * latent_dim)` (~6% active fraction). `matryoshka_dims` defaults to
    `[latent_dim/4, latent_dim/2, latent_dim]` when not given.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    latent_dim = latent_dim or 4 * D
    k = k or max(1, round(0.06 * latent_dim))
    if matryoshka_dims is None:
        matryoshka_dims = sorted({max(1, latent_dim // 4), max(1, latent_dim // 2), latent_dim})

    if verbose:
        print(f"Training SAE: {N} samples, {D} input dims -> {latent_dim} latent dims (k={k}, mode={sparsity_mode})")

    perm = np.random.permutation(N)
    n_val = int(round(val_frac * N))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, X_va = X[tr_idx], X[val_idx]

    if feature_standardize:
        mu = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
        sd = (X_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        X_tr = (X_tr - mu) / sd
        X_va = (X_va - mu) / sd
    else:
        mu = np.zeros((1, D), dtype=np.float32)
        sd = np.ones((1, D), dtype=np.float32)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False, drop_last=False)

    use_matryoshka = matryoshka_dims is not None
    use_kl = sparsity_mode == "kl"

    model = SparseAutoencoder(
        input_dim=D, latent_dim=latent_dim, k=k,
        sparsity_mode=sparsity_mode, matryoshka_dims=matryoshka_dims,
        matryoshka_weights=matryoshka_weights, kl_target=kl_target,
        normalize_decoder_only=normalize_decoder_only,
    ).to(device)
    model.mu = torch.tensor(mu, device=device)
    model.sd = torch.tensor(sd, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()

    target_usage = k / latent_dim
    usage_cap = max(usage_cap_mult * target_usage, 1e-4)

    if use_kl:
        rho_hat_ema = torch.full((latent_dim,), kl_target, device=device)

    history = {key: [] for key in [
        "train_recon", "val_recon", "train_total", "train_usage_loss",
        "train_l1", "train_matryoshka_loss", "train_kl_loss",
        "usage_mean", "usage_max", "usage_gini", "usage_neff",
        "dead_neurons", "resampled",
    ]}

    for ep in range(1, epochs + 1):
        t = (ep - 1) / max(epochs - 1, 1)
        noise_std = noise_std_start * (1 - t) + noise_std_end * t

        model.train()
        sums = dict.fromkeys(["recon", "total", "usage", "l1", "matr", "kl"], 0.0)
        nb = 0
        usage_accum = None

        for (xb,) in tr_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)

            need_h = use_matryoshka or use_kl
            if need_h:
                if l1_weight > 0:
                    x_hat, z, _, _, l1, h_fwd = model(xb, noise_std=noise_std, return_l1=True, return_h=True)
                else:
                    x_hat, z, _, _, h_fwd = model(xb, noise_std=noise_std, return_l1=False, return_h=True)
                    l1 = torch.tensor(0.0, device=device)
            else:
                h_fwd = None
                if l1_weight > 0:
                    x_hat, z, _, _, l1 = model(xb, noise_std=noise_std, return_l1=True)
                else:
                    x_hat, z, _, _ = model(xb, noise_std=noise_std, return_l1=False)
                    l1 = torch.tensor(0.0, device=device)

            recon = recon_loss_fn(x_hat, xb)
            usage_loss, usage_vec = usage_regularization_cap(z, usage_cap)
            loss = recon + usage_weight * usage_loss + l1_weight * l1

            kl_loss_val = torch.tensor(0.0, device=device)
            if use_kl:
                p_hat_batch = torch.sigmoid(h_fwd).mean(dim=0)
                rho_hat_ema = kl_ema * rho_hat_ema.detach() + (1 - kl_ema) * p_hat_batch.detach()
                kl_loss_val = kl_sparsity_penalty(p_hat_batch, kl_target)
                loss = loss + kl_weight * kl_loss_val

            if use_matryoshka:
                h_for_matr = F.relu(h_fwd) if use_kl else h_fwd
                nested_recons = model.matryoshka_recons(h_for_matr)
                scale_losses = torch.stack([recon_loss_fn(r, xb) for r in nested_recons])
                w = torch.tensor(model.matryoshka_weights, dtype=scale_losses.dtype, device=scale_losses.device)
                matr_loss = (w * scale_losses).sum()
                loss = loss + matryoshka_weight * matr_loss
                sums["matr"] += matr_loss.item()

            loss.backward()
            opt.step()

            sums["recon"] += recon.item()
            sums["usage"] += usage_loss.item()
            sums["l1"] += float(l1.item())
            sums["total"] += loss.item()
            sums["kl"] += float(kl_loss_val.item())
            nb += 1
            usage_accum = (usage_vec.detach().float().clone() if usage_accum is None
                           else usage_accum + usage_vec.detach().float())

        usage_epoch = (usage_accum / max(nb, 1)).detach().cpu().numpy()
        dead_count = int((usage_epoch <= dead_thresh).sum())

        model.eval()
        va_recon_sum, vb = 0.0, 0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(device)
                x_hat, _, _, _ = model(xb, noise_std=0.0, return_l1=False)
                va_recon_sum += recon_loss_fn(x_hat, xb).item()
                vb += 1

        resampled = 0
        if resample_every and (ep % resample_every == 0):
            dead_mask = torch.tensor(usage_epoch <= dead_thresh, device=device)
            dead_idx = torch.where(dead_mask)[0]
            if dead_idx.numel() > 0:
                max_rs = int(resample_frac_max * latent_dim)
                if dead_idx.numel() > max_rs:
                    dead_idx = dead_idx[:max_rs]
                    dead_mask = torch.zeros(latent_dim, dtype=torch.bool, device=device)
                    dead_mask[dead_idx] = True
                resampled = model.resample_dead(dead_mask, bias=resample_bias)

        for key, val in [
            ("train_recon", sums["recon"] / max(nb, 1)),
            ("val_recon", va_recon_sum / max(vb, 1)),
            ("train_total", sums["total"] / max(nb, 1)),
            ("train_usage_loss", sums["usage"] / max(nb, 1)),
            ("train_l1", sums["l1"] / max(nb, 1)),
            ("train_matryoshka_loss", sums["matr"] / max(nb, 1)),
            ("train_kl_loss", sums["kl"] / max(nb, 1)),
            ("usage_mean", float(usage_epoch.mean())),
            ("usage_max", float(usage_epoch.max())),
            ("usage_gini", gini_coefficient(usage_epoch)),
            ("usage_neff", effective_num_neurons(usage_epoch)),
            ("dead_neurons", dead_count),
            ("resampled", int(resampled)),
        ]:
            history[key].append(val)

        if verbose and (ep % 10 == 0 or ep == 1 or ep == epochs):
            print(f"Epoch {ep:03d}/{epochs} | train_recon={history['train_recon'][-1]:.6f} "
                  f"val_recon={history['val_recon'][-1]:.6f} | "
                  f"usage_neff={history['usage_neff'][-1]:.1f} dead={dead_count} resampled={resampled}")

    model.eval()
    Z_full = model.encode(X, device=device, batch_size=batch_size)

    return SAEResult(model=model, Z=Z_full, history=history, train_idx=tr_idx, val_idx=val_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing (no plotting/figure code — that stays in downstream analysis code)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: SparseAutoencoder, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": int(model.W.shape[1]),
            "latent_dim": model.latent_dim,
            "k": model.k,
            "sparsity_mode": model.sparsity_mode,
            "matryoshka_dims": model.matryoshka_dims,
            "matryoshka_weights": model.matryoshka_weights,
            "kl_target": model.kl_target,
            "normalize_decoder_only": model.normalize_decoder_only,
        },
    }, path)


def load_checkpoint(path: str, device: str | None = None) -> SparseAutoencoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = SparseAutoencoder(
        input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"], k=cfg["k"],
        sparsity_mode=cfg["sparsity_mode"], matryoshka_dims=cfg["matryoshka_dims"],
        matryoshka_weights=cfg["matryoshka_weights"], kl_target=cfg["kl_target"],
        normalize_decoder_only=cfg["normalize_decoder_only"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
