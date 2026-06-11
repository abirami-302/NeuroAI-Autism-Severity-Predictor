# =============================================================
# models/vit.py
# Vision Transformer (ViT) — Optional Feature Extractor.
#
# Provides a lightweight ViT that can replace or complement
# the 3D CNN for MRI feature extraction.
#
# The ViT treats 3D MRI volumes as sequences of 3D patches,
# applies multi-head self-attention, and pools to a feature
# vector of the same size as CNN3D output (CNN_FEATURE_DIM).
#
# Use case: swap vit=True in config to benchmark ViT vs CNN.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (
    IMAGE_SIZE, IMAGE_CHANNELS, CNN_FEATURE_DIM, NUM_CLASSES
)


# =============================================================
# PATCH EMBEDDING
# Splits 3D volume into non-overlapping patches and embeds them.
# =============================================================

class PatchEmbedding3D(nn.Module):
    """
    Partition a 3D MRI volume into equal-sized cubic patches
    and linearly project each patch to an embedding vector.

    Args:
        volume_size (tuple): (D, H, W) of input volume
        patch_size (int):    Cubic patch size (e.g. 16 → 16³ patches)
        in_channels (int):   MRI input channels (1 for grayscale)
        embed_dim (int):     Patch embedding dimension
    """
    def __init__(self,
                 volume_size: tuple = IMAGE_SIZE,
                 patch_size:  int   = 16,
                 in_channels: int   = IMAGE_CHANNELS,
                 embed_dim:   int   = 256):
        super(PatchEmbedding3D, self).__init__()

        D, H, W    = volume_size
        self.patch_size = patch_size

        # Number of patches per axis
        self.n_d = D // patch_size
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.num_patches = self.n_d * self.n_h * self.n_w

        # Each patch has (patch_size³ * channels) raw values
        patch_dim = in_channels * (patch_size ** 3)

        # Linear projection from raw patch → embed_dim
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, C, D, H, W)
        Returns:
            torch.Tensor: Patch tokens (B, num_patches, embed_dim)
        """
        B, C, D, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, n_d, p, n_h, p, n_w, p)
        x = x.reshape(B, C,
                       self.n_d, p,
                       self.n_h, p,
                       self.n_w, p)

        # Permute to (B, n_d, n_h, n_w, C, p, p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)

        # Flatten each patch: (B, num_patches, C*p*p*p)
        x = x.reshape(B, self.num_patches, -1)

        # Project and normalise
        x = self.projection(x)   # (B, num_patches, embed_dim)
        x = self.norm(x)

        return x


# =============================================================
# MULTI-HEAD SELF-ATTENTION BLOCK
# =============================================================

class TransformerBlock(nn.Module):
    """
    Standard ViT Transformer encoder block.
        LayerNorm → Multi-Head Attention → residual
        LayerNorm → Feed-Forward MLP    → residual

    Args:
        embed_dim (int):   Token embedding dimension
        num_heads (int):   Number of attention heads
        mlp_ratio (float): MLP hidden dim = embed_dim * mlp_ratio
        dropout (float):   Dropout probability
    """
    def __init__(self,
                 embed_dim:  int   = 256,
                 num_heads:  int   = 8,
                 mlp_ratio:  float = 2.0,
                 dropout:    float = 0.1):
        super(TransformerBlock, self).__init__()

        mlp_hidden = int(embed_dim * mlp_ratio)

        self.norm1    = nn.LayerNorm(embed_dim)
        self.attn     = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2    = nn.LayerNorm(embed_dim)
        self.ffn      = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Token sequence (B, N, embed_dim)
        Returns:
            torch.Tensor: Updated tokens (B, N, embed_dim)
        """
        # Self-attention with residual
        normed, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + normed

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        return x


# =============================================================
# VISION TRANSFORMER FOR 3D MRI
# =============================================================

class ViT3D(nn.Module):
    """
    Lightweight 3D Vision Transformer for MRI feature extraction.

    Matches CNN3D's output size (CNN_FEATURE_DIM = 128) so it
    can be used as a drop-in replacement in the pipeline.

    Architecture:
        PatchEmbedding3D → [CLS token] → positional encoding
        → N × TransformerBlock
        → CLS token → Linear → feature vector (128)

    Args:
        volume_size (tuple): (D, H, W) — must match IMAGE_SIZE
        patch_size (int):    Cubic patch side length (default 16)
        embed_dim (int):     Transformer embedding dimension
        depth (int):         Number of transformer blocks
        num_heads (int):     Attention heads per block
        feature_dim (int):   Output feature size (= CNN_FEATURE_DIM)
    """
    def __init__(self,
                 volume_size: tuple = IMAGE_SIZE,
                 patch_size:  int   = 16,
                 embed_dim:   int   = 256,
                 depth:       int   = 4,
                 num_heads:   int   = 8,
                 feature_dim: int   = CNN_FEATURE_DIM):
        super(ViT3D, self).__init__()

        self.patch_embed = PatchEmbedding3D(
            volume_size=volume_size,
            patch_size=patch_size,
            in_channels=IMAGE_CHANNELS,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token — used as the global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for all patches + CLS
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim,
                             num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project CLS token → feature_dim (same size as CNN3D output)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True)
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, 1, D, H, W) MRI volume
        Returns:
            torch.Tensor: Feature vector (B, feature_dim)
        """
        B = x.shape[0]

        # 1. Embed patches
        tokens = self.patch_embed(x)                    # (B, N, embed_dim)

        # 2. Prepend CLS token
        cls    = self.cls_token.expand(B, -1, -1)       # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, N+1, embed_dim)

        # 3. Add positional encoding
        tokens = tokens + self.pos_embed

        # 4. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 5. Use CLS token as global feature
        cls_out = tokens[:, 0]                          # (B, embed_dim)

        # 6. Project to feature_dim
        features = self.head(cls_out)                   # (B, feature_dim)

        return features


# =============================================================
# ViT-based End-to-End Classifier (standalone, no GNN)
# =============================================================

class ViT3DClassifier(nn.Module):
    """
    End-to-end ViT classifier: MRI volume → severity label.
    Can replace CNN3DClassifier for ablation studies.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(ViT3DClassifier, self).__init__()
        self.backbone = ViT3D()
        self.head = nn.Sequential(
            nn.Linear(CNN_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing ViT3D...")

    model = ViT3D()
    model.eval()

    # Use a smaller test volume to keep memory reasonable
    x = torch.randn(1, 1, 64, 64, 64)

    with torch.no_grad():
        feats = model(x)

    print(f"  Input shape   : {x.shape}")
    print(f"  Output shape  : {feats.shape}")     # (1, 128)
    print(f"  Num patches   : {model.patch_embed.num_patches}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params  : {total_params:,}")

    print("\n✅ ViT3D test passed!")
