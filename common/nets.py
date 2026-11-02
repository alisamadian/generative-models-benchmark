"""Neural network architectures: Shared U-Net, Parameter-Matched VAE, and SmallResNet.

Implements Section 7 of the specification:
1. `UNet`: Shared backbone for both DDPM Diffusion and Flow Matching. For a given
   dataset and training seed, `build_model(dataset, "diffusion", seed)` and
   `build_model(dataset, "flow_matching", seed)` produce bit-for-bit identical
   initial weights.
2. `VAE`: Convolutional Variational Autoencoder with a calibrated learned global
   decoder standard deviation (`log_sigma`), parameter-matched within 25% of the
   U-Net on the same dataset.
3. `SmallResNet`: Evaluation classifier trained on the training split to measure
   class balance (predicted-class histogram, TV distance, normalized entropy,
   and mean max softmax confidence).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import DATASET_CONFIGS, UNET_CONFIGS, VAE_CONFIGS
from common.utils import set_seed


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch module.

    Why: Enforces and records the parameter-matching rule between the U-Net
    and VAE on each dataset (Section 7.1, 7.2).
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal scalar timestep embedding (DDPM / Transformer style).

    Why: Diffusion passes integer `t` in {1..1000} and Flow Matching passes
    `t * 1000` with `t` in [0, 1], ensuring both model families operate over the
    identical sinusoidal frequency spectrum (Section 7.1 & Section 16).
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Always compute frequencies in fp32 to avoid fp16 precision loss
        t_fp32 = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t_fp32[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, dropout, and optional time embedding.

    Follows Section 7.1:
    GroupNorm -> SiLU -> 3x3 conv -> (+ projected time emb) ->
    GroupNorm -> SiLU -> Dropout -> 3x3 conv -> (+ skip connection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dropout: float = 0.0,
        time_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError(
                f"GroupNorm num_groups={num_groups} must divide both "
                f"in_channels={in_channels} and out_channels={out_channels}."
            )
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if time_emb_dim is not None:
            self.time_proj: Optional[nn.Module] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
        else:
            self.time_proj = None

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_proj is not None:
            if t_emb is None:
                raise ValueError("ResBlock configured with time_emb_dim requires t_emb.")
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention in the DDPM style (Section 7.1).

    Why fp32 softmax: Section 16 requires keeping attention softmax in fp32 to
    prevent overflow/NaNs during fp16 mixed-precision training on T4 GPUs.
    """

    def __init__(self, channels: int, num_groups: int) -> None:
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"GroupNorm num_groups={num_groups} must divide channels={channels}."
            )
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        # Zero-initialize output projection as in standard DDPM
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for spatial attention: (B, C, H*W)
        q_flat = q.reshape(b, c, h * w).transpose(1, 2).float()  # (B, HW, C)
        k_flat = k.reshape(b, c, h * w).float()  # (B, C, HW)
        v_flat = v.reshape(b, c, h * w).transpose(1, 2).float()  # (B, HW, C)

        scale = 1.0 / math.sqrt(c)
        attn_logits = torch.bmm(q_logits := q_flat * scale, k_flat)  # (B, HW, HW)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        out = torch.bmm(attn_weights, v_flat)  # (B, HW, C)

        out = out.transpose(1, 2).reshape(b, c, h, w).to(x.dtype)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    """Strided 3x3 convolution downsampling by factor 2 (Section 7.1)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling followed by 3x3 convolution (Section 7.1)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x_up)


class UNet(nn.Module):
    """Shared DDPM-style U-Net backbone for Diffusion and Flow Matching (Section 7.1).

    Architecture details:
    - Sinusoidal time embedding + 2-layer MLP with SiLU.
    - Input 3x3 conv to `base_channels`.
    - Down path: `len(channel_multipliers)` resolution levels, `num_res_blocks`
      per level, self-attention at `attention_resolutions`, strided conv down
      between levels, storing skip connections at every block.
    - Middle bottleneck: ResBlock -> AttentionBlock -> ResBlock.
    - Up path: `num_res_blocks + 1` blocks per level consuming concatenated skip
      tensors, nearest-neighbour upsampling + 3x3 conv between levels.
    - Output head: GroupNorm -> SiLU -> zero-initialized 3x3 conv to `in_channels`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        num_groups: int = 32,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        self.dropout = dropout
        self.num_groups = num_groups
        self.image_size = image_size

        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Stem
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_channels: List[int] = [base_channels]
        curr_ch = base_channels
        curr_res = image_size
        num_levels = len(channel_multipliers)

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(
                        curr_ch,
                        out_ch,
                        num_groups=num_groups,
                        dropout=dropout,
                        time_emb_dim=time_emb_dim,
                    )
                )
                curr_ch = out_ch
                if curr_res in attention_resolutions:
                    level_attns.append(AttentionBlock(curr_ch, num_groups=num_groups))
                else:
                    level_attns.append(nn.Identity())
                skip_channels.append(curr_ch)

            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)

            if level != num_levels - 1:
                self.downsamples.append(Downsample(curr_ch))
                skip_channels.append(curr_ch)
                curr_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle bottleneck
        self.mid_block1 = ResBlock(
            curr_ch,
            curr_ch,
            num_groups=num_groups,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )
        self.mid_attn = AttentionBlock(curr_ch, num_groups=num_groups)
        self.mid_block2 = ResBlock(
            curr_ch,
            curr_ch,
            num_groups=num_groups,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(
                    ResBlock(
                        curr_ch + skip_ch,
                        out_ch,
                        num_groups=num_groups,
                        dropout=dropout,
                        time_emb_dim=time_emb_dim,
                    )
                )
                curr_ch = out_ch
                if curr_res in attention_resolutions:
                    level_attns.append(AttentionBlock(curr_ch, num_groups=num_groups))
                else:
                    level_attns.append(nn.Identity())

            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)

            if level != 0:
                self.upsamples.append(Upsample(curr_ch))
                curr_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # Output head (zero-initialised per Section 7.1)
        self.norm_out = nn.GroupNorm(num_groups, curr_ch)
        self.conv_out = nn.Conv2d(curr_ch, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net.

        Args:
            x: Tensor of shape `(B, C, 32, 32)` in `[-1, 1]`.
            t: Scalar timestep per batch item, shape `(B,)`.
               - Diffusion passes integer `t` in `{1..T}`.
               - Flow Matching passes `t_scaled = t * 1000.0` with `t` in `[0, 1]`.
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_emb = self.time_mlp(t).to(x.dtype)

        h = self.conv_in(x)
        skips: List[torch.Tensor] = [h]

        for level_blocks, level_attns, downsample in zip(
            self.down_blocks, self.down_attns, self.downsamples
        ):
            for block, attn in zip(level_blocks, level_attns):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            if not isinstance(downsample, nn.Identity):
                h = downsample(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for level_blocks, level_attns, upsample in zip(
            self.up_blocks, self.up_attns, self.upsamples
        ):
            for block, attn in zip(level_blocks, level_attns):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            if not isinstance(upsample, nn.Identity):
                h = upsample(h)

        return self.conv_out(F.silu(self.norm_out(h)))


class VAE(nn.Module):
    """Convolutional Variational Autoencoder with calibrated decoder variance (Section 7.2).

    Architecture details:
    - Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 using 3x3 stride-2 convolutions and
      GroupNorm+SiLU residual blocks, flattened into two linear heads (`fc_mu`, `fc_logvar`).
    - Decoder: Linear head from `latent_dim` to `4x4` feature map, followed by
      nearest-neighbour upsampling + GroupNorm+SiLU residual blocks back to `32x32`,
      ending with a 3x3 convolution to image channels and `tanh` activation.
    - Calibrated decoder variance: Single learned global scalar `log_sigma`
      (Section 7.2) so reconstruction weight adapts without manual variance tuning.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 3, 3),
        num_res_blocks: int = 2,
        latent_dim: int = 128,
        num_groups: int = 32,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.image_size = image_size

        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4 (3 stride-2 transitions across 4 levels)
        self.enc_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        enc_layers: List[nn.Module] = []
        curr_ch = base_channels
        num_levels = len(channel_multipliers)

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                enc_layers.append(
                    ResBlock(curr_ch, out_ch, num_groups=num_groups, dropout=0.0)
                )
                curr_ch = out_ch
            if level != num_levels - 1:
                # 3x3 stride-2 convolution block with GroupNorm + SiLU
                enc_layers.append(
                    nn.Sequential(
                        nn.GroupNorm(num_groups, curr_ch),
                        nn.SiLU(),
                        nn.Conv2d(curr_ch, curr_ch, kernel_size=3, stride=2, padding=1),
                    )
                )
        self.encoder_net = nn.Sequential(*enc_layers)
        self.enc_norm_out = nn.GroupNorm(num_groups, curr_ch)
        self.bottleneck_channels = curr_ch
        self.bottleneck_spatial = image_size // (2 ** (num_levels - 1))  # 4 for 4 levels
        flat_dim = self.bottleneck_channels * (self.bottleneck_spatial ** 2)

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: latent_dim -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        dec_layers: List[nn.Module] = []
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                dec_layers.append(
                    ResBlock(curr_ch, out_ch, num_groups=num_groups, dropout=0.0)
                )
                curr_ch = out_ch
            if level != 0:
                dec_layers.append(Upsample(curr_ch))
        self.decoder_net = nn.Sequential(*dec_layers)
        self.dec_norm_out = nn.GroupNorm(num_groups, curr_ch)
        self.dec_conv_out = nn.Conv2d(curr_ch, in_channels, kernel_size=3, padding=1)

        # Learned global scalar log(sigma) for calibrated Gaussian decoder (Section 7.2)
        self.log_sigma = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input image `x` in `[-1, 1]` to `(mu, logvar)`."""
        h = self.enc_in(x)
        h = self.encoder_net(h)
        h = F.silu(self.enc_norm_out(h))
        h_flat = h.flatten(start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = torch.clamp(self.fc_logvar(h_flat), min=-20.0, max=10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reparameterization trick `z = mu + exp(0.5 * logvar) * eps`."""
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector `z` to image mean in `[-1, 1]` via `tanh`."""
        h = self.fc_dec(z)
        h = h.view(
            z.shape[0],
            self.bottleneck_channels,
            self.bottleneck_spatial,
            self.bottleneck_spatial,
        )
        h = self.decoder_net(h)
        h = F.silu(self.dec_norm_out(h))
        return torch.tanh(self.dec_conv_out(h))

    def forward(
        self, x: torch.Tensor, eps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE pass returning `(recon_mu, mu, logvar)`."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, eps=eps)
        recon_mu = self.decode(z)
        return recon_mu, mu, logvar


class _BasicResBlock(nn.Module):
    """Standard BatchNorm + ReLU residual block for the evaluation classifier."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallResNet(nn.Module):
    """ResNet-18-style evaluation classifier for 28x28 / 32x32 images (Section 7.7).

    Why: Used exclusively to evaluate generated-sample class balance (predicted
    class histogram, total variation distance from uniform, normalized entropy,
    and mean maximum softmax probability). Targets >=99% on MNIST and >=90% on
    CIFAR-10 when trained on the training split with training-only augmentation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, base_width, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(
            base_width, base_width * 2, num_blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            base_width * 2, base_width * 4, num_blocks=2, stride=2
        )
        self.layer4 = self._make_layer(
            base_width * 4, base_width * 8, num_blocks=2, stride=2
        )
        self.head = nn.Linear(base_width * 8, num_classes)

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [_BasicResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(_BasicResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
        return self.head(h)


def build_model(
    dataset: str,
    model_name: str,
    seed: int,
    hparams: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Build a model with deterministic initialization for `(dataset, model_name, seed)`.

    Why: Section 7.1 & Section 8 require that for a given dataset and seed,
    `diffusion` and `flow_matching` instantiate the exact same `UNet` class with
    identical initial weights.

    Args:
        dataset: `"mnist"` or `"cifar10"`.
        model_name: `"vae"`, `"diffusion"`, `"flow_matching"`, or `"classifier"`.
        seed: Integer seed used to initialize network weights deterministically.
        hparams: Optional hyperparameter dictionary (e.g. `{"latent_dim": 256}` for VAE).
    """
    dataset = dataset.lower()
    model_name = model_name.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'mnist' or 'cifar10'.")

    hparams = hparams or {}
    set_seed(seed)

    if model_name in ("diffusion", "flow_matching"):
        cfg = dict(UNET_CONFIGS[dataset])
        return UNet(
            in_channels=cfg["in_channels"],
            base_channels=cfg["base_channels"],
            channel_multipliers=tuple(cfg["channel_multipliers"]),
            num_res_blocks=cfg["num_res_blocks"],
            attention_resolutions=tuple(cfg["attention_resolutions"]),
            dropout=cfg["dropout"],
            num_groups=cfg["num_groups"],
            image_size=32,
        )

    if model_name == "vae":
        cfg = dict(VAE_CONFIGS[dataset])
        latent_dim = int(hparams.get("latent_dim", cfg["latent_dim"]))
        return VAE(
            in_channels=cfg["in_channels"],
            base_channels=cfg["base_channels"],
            channel_multipliers=tuple(cfg["channel_multipliers"]),
            num_res_blocks=cfg["num_res_blocks"],
            latent_dim=latent_dim,
            num_groups=cfg["num_groups"],
            image_size=32,
        )

    if model_name == "classifier":
        in_ch = DATASET_CONFIGS[dataset]["in_channels"]
        num_classes = DATASET_CONFIGS[dataset]["num_classes"]
        return SmallResNet(in_channels=in_ch, num_classes=num_classes)

    raise ValueError(
        f"Unknown model_name '{model_name}'. Expected 'vae', 'diffusion', "
        f"'flow_matching', or 'classifier'."
    )
