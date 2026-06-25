"""Sparse interface feature processing for PLM-free antibody CDR design.

Handles the 4D complementarity features (shape_comp, elec_comp, hydro_comp, aromatic)
which are ~95% sparse because they only apply to interface residues.

Key components:
- InterfaceFeatureEncoder: Processes sparse features with interface-aware masking
- interface_aware_normalize: Normalizes only over interface residues
- DualPathEncoder: Separate pathways for dense (101D) and sparse (4D) features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InterfaceFeatureEncoder(nn.Module):
    """Encodes sparse interface features with masking.

    Uses interface_mask to:
    1. Apply learned "no-interface" embedding for non-interface residues
    2. Process actual complementarity values only where interface_mask=1
    3. Optionally normalize only over interface residues
    """

    def __init__(
        self,
        hidden_dim: int,
        sparse_dim: int = 4,  # shape_comp, elec_comp, hydro_comp, aromatic
        dropout: float = 0.1,
        use_no_interface_embed: bool = True,
    ):
        """
        Args:
            hidden_dim: Output hidden dimension
            sparse_dim: Number of sparse features (default 4)
            dropout: Dropout probability
            use_no_interface_embed: If True, use learned embedding for non-interface
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparse_dim = sparse_dim
        self.use_no_interface_embed = use_no_interface_embed

        # Project sparse features to hidden
        self.sparse_proj = nn.Sequential(
            nn.Linear(sparse_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned embedding for non-interface positions
        if use_no_interface_embed:
            self.no_interface_embed = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.no_interface_embed, std=0.02)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        sparse_features: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sparse interface features.

        Args:
            sparse_features: (N, 4) complementarity features
            interface_mask: (N,) binary mask (1 = at interface)

        Returns:
            (N, hidden_dim) encoded features
        """
        # Project sparse features
        h_sparse = self.sparse_proj(sparse_features)  # (N, hidden_dim)

        if self.use_no_interface_embed:
            # Replace non-interface positions with learned embedding
            interface_mask_expanded = interface_mask.unsqueeze(-1)  # (N, 1)
            h = h_sparse * interface_mask_expanded + \
                self.no_interface_embed * (1 - interface_mask_expanded)
        else:
            # Zero out non-interface positions
            h = h_sparse * interface_mask.unsqueeze(-1)

        return self.norm(h)


class DualPathEncoder(nn.Module):
    """Encodes dense and sparse features through separate pathways.

    Dense path: 101D features (identity, torsions, structure, surface, etc.)
    Sparse path: 4D complementarity features (interface-only)

    Fusion combines both paths, with sparse path weighted by interface_mask.
    """

    def __init__(
        self,
        hidden_dim: int,
        dense_dim: int = 101,  # 105 - 4 complementarity
        sparse_dim: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'concat',  # 'concat', 'add', 'gate'
    ):
        """
        Args:
            hidden_dim: Output hidden dimension
            dense_dim: Dimension of dense features
            sparse_dim: Dimension of sparse features
            dropout: Dropout probability
            fusion_type: How to combine dense and sparse paths
        """
        super().__init__()
        self.fusion_type = fusion_type

        # Dense pathway
        self.dense_encoder = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Sparse pathway with interface-aware encoding
        self.sparse_encoder = InterfaceFeatureEncoder(
            hidden_dim=hidden_dim // 2 if fusion_type == 'concat' else hidden_dim,
            sparse_dim=sparse_dim,
            dropout=dropout,
        )

        # Fusion layer
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
        elif fusion_type == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.LayerNorm(hidden_dim)
        else:  # 'add'
            self.fusion = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode dense and sparse features through dual pathways.

        Args:
            dense_features: (N, dense_dim) dense per-residue features
            sparse_features: (N, 4) sparse complementarity features
            interface_mask: (N,) binary interface mask

        Returns:
            (N, hidden_dim) fused features
        """
        h_dense = self.dense_encoder(dense_features)
        h_sparse = self.sparse_encoder(sparse_features, interface_mask)

        if self.fusion_type == 'concat':
            h = torch.cat([h_dense, h_sparse], dim=-1)
            h = self.fusion(h)
        elif self.fusion_type == 'gate':
            gate = self.gate(torch.cat([h_dense, h_sparse], dim=-1))
            h = h_dense + gate * h_sparse
            h = self.fusion(h)
        else:  # 'add'
            h = h_dense + h_sparse
            h = self.fusion(h)

        return h


def interface_aware_normalize(
    features: torch.Tensor,
    interface_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize features using statistics from interface residues only.

    Args:
        features: (N, D) or (B, N, D) features to normalize
        interface_mask: (N,) or (B, N) binary interface mask
        eps: Small value for numerical stability

    Returns:
        Normalized features (same shape as input)
    """
    if features.dim() == 2:
        # Single sample: (N, D)
        interface_idx = interface_mask.bool()
        if interface_idx.sum() < 2:
            # Not enough interface residues, use global stats
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + eps
        else:
            interface_feats = features[interface_idx]
            mean = interface_feats.mean(dim=0, keepdim=True)
            std = interface_feats.std(dim=0, keepdim=True) + eps
        return (features - mean) / std

    else:
        # Batched: (B, N, D)
        B, N, D = features.shape
        normalized = torch.zeros_like(features)
        for b in range(B):
            normalized[b] = interface_aware_normalize(
                features[b], interface_mask[b], eps
            )
        return normalized


def split_dense_sparse(
    node_features: torch.Tensor,
    sparse_start: int = 101,
    sparse_end: int = 105,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated features into dense and sparse parts.

    Default layout (105D):
        0-100: Dense features (identity, torsions, structure, surface, etc.)
        101-104: Sparse complementarity features

    Args:
        node_features: (N, 105) or (B, N, 105) concatenated features
        sparse_start: Start index of sparse features
        sparse_end: End index of sparse features

    Returns:
        dense_features: (N, 101) or (B, N, 101)
        sparse_features: (N, 4) or (B, N, 4)
    """
    dense = torch.cat([
        node_features[..., :sparse_start],
        # No trailing dense features in current layout
    ], dim=-1)
    sparse = node_features[..., sparse_start:sparse_end]
    return dense, sparse


class ProteinFeatureEncoderWithSparse(nn.Module):
    """Enhanced feature encoder with separate sparse pathway.

    Replaces the basic ProteinFeatureEncoder for models that want
    interface-aware processing of complementarity features.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int = 108,  # 105 node + 3 segment
        dropout: float = 0.1,
        n_global_tokens: int = 3,
        use_dual_path: bool = True,
        fusion_type: str = 'concat',
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            input_dim: Total input dimension (default 108)
            dropout: Dropout probability
            n_global_tokens: Number of global tokens
            use_dual_path: Whether to use dual-path encoding
            fusion_type: Fusion type for dual-path ('concat', 'add', 'gate')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_global_tokens = n_global_tokens
        self.use_dual_path = use_dual_path

        # Feature dimensions
        # 105 node features: 101 dense + 4 sparse
        # 3 segment type
        self.dense_dim = 101 + 3  # dense node + segment
        self.sparse_dim = 4  # complementarity

        if use_dual_path:
            self.encoder = DualPathEncoder(
                hidden_dim=hidden_dim,
                dense_dim=self.dense_dim,
                sparse_dim=self.sparse_dim,
                dropout=dropout,
                fusion_type=fusion_type,
            )
        else:
            # Fallback to simple projection
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # Global token embeddings
        self.global_embed = nn.Embedding(n_global_tokens, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        segment_type: torch.Tensor,
        interface_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
        global_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode features with interface-aware sparse processing.

        Args:
            node_features: (N, 105) pre-computed features
            segment_type: (N, 3) segment one-hot
            interface_mask: (N,) binary interface mask (required if use_dual_path)
            global_mask: (N_total,) bool mask for global tokens
            global_types: (N_global,) indices for global tokens

        Returns:
            (N_total, hidden_dim) encoded features
        """
        if self.use_dual_path:
            # Split into dense and sparse
            dense_feats = node_features[..., :101]  # First 101D
            sparse_feats = node_features[..., 101:105]  # Last 4D (complementarity)

            # Concatenate segment_type with dense features
            dense_with_seg = torch.cat([dense_feats, segment_type], dim=-1)

            # Default interface_mask to all zeros if not provided
            if interface_mask is None:
                interface_mask = torch.zeros(
                    node_features.shape[0], device=node_features.device
                )

            h = self.encoder(dense_with_seg, sparse_feats, interface_mask)
        else:
            # Simple projection
            x = torch.cat([node_features, segment_type], dim=-1)
            h = self.encoder(x)

        if global_mask is None:
            return h

        # Insert global token embeddings
        n_total = global_mask.shape[0]
        output = torch.zeros(n_total, self.hidden_dim, device=h.device, dtype=h.dtype)
        output[~global_mask] = h

        if global_types is not None and global_types.numel() > 0:
            output[global_mask] = self.global_embed(global_types)

        return output


# ── CDR feature masking and auxiliary reconstruction ──────────────────────

# Feature layout in 105D node_features:
#   0-54:   Identity (55D) - aa_onehot(20) + physicochemical(15) + blosum(20)
#   55-68:  Torsions (14D) - backbone(6) + chi(8)
#   69-75:  Structure (7D) - SS(3) + SASA(1) + burial(1) + packing(1) + contact_order(1)
#   76-87:  Surface (12D) - normal(3) + curvature(2) + shape_index(1) + chemistry(5) + electrostatic(1)
#   88-91:  Electrostatics (4D) - potential(1) + field(3)
#   92-95:  H-bonds (4D) - donors(1) + acceptors(1) + satisfaction(1) + backbone(1)
#   96-100: Dynamics (5D) - fluctuations(3) + collectivity(1) + coupling(1)
#   101-104: Complementarity (4D) - shape(1) + electrostatic(1) + hydrophobic(1) + aromatic(1)

# Auxiliary reconstruction targets: structural + surface + complementarity = 46D
# These features describe what the CDR structure/binding SHOULD look like.
# Identity (55D) is NOT an auxiliary target because predicting AA is already the main task.
AUX_TARGET_SLICES = [
    (55, 69),    # Torsions (14D)
    (69, 76),    # Structure (7D)
    (76, 88),    # Surface (12D)
    (88, 92),    # Electrostatics (4D)
    (92, 96),    # H-bonds (4D)
    (101, 105),  # Complementarity (4D)
]
AUX_TARGET_DIM = sum(e - s for s, e in AUX_TARGET_SLICES)  # 45D


def extract_aux_targets(node_features, cdr_mask_1d):
    """Extract auxiliary reconstruction targets for CDR positions.

    Args:
        node_features: (N, 105) full features BEFORE masking
        cdr_mask_1d: (N,) bool mask for CDR positions

    Returns:
        aux_targets: (N_cdr, 45D) structural+surface features for CDR positions
    """
    parts = [node_features[cdr_mask_1d, s:e] for s, e in AUX_TARGET_SLICES]
    return torch.cat(parts, dim=-1)


def mask_cdr_features(node_features, cdr_mask_1d):
    """Zero out all features for CDR positions.

    At test time, CDR sequence AND structure are unknown, so ALL 105D features
    are unavailable. During training, masking prevents information leakage and
    forces the model to learn antigen→CDR mappings.

    Args:
        node_features: (N, 105) features to mask (modified in-place)
        cdr_mask_1d: (N,) bool mask for CDR positions
    """
    node_features[cdr_mask_1d] = 0.0


class CDRFeaturePredictor(nn.Module):
    """Predicts structural/surface features of CDR from GNN embeddings.

    Auxiliary reconstruction task: given antigen context (via GNN message passing),
    predict what the CDR's structural and binding features SHOULD look like.
    This forces the model to learn antigen→CDR property mappings.

    Target: 45D (torsions + structure + surface + electrostatics + hbonds + complementarity)
    """

    def __init__(self, hidden_dim, aux_dim=AUX_TARGET_DIM, dropout=0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, aux_dim),
        )

    def forward(self, cdr_embeddings):
        """Predict CDR structural features from GNN embeddings.

        Args:
            cdr_embeddings: (N_cdr, hidden_dim) GNN output for CDR positions

        Returns:
            predicted: (N_cdr, aux_dim) predicted structural features
        """
        return self.predictor(cdr_embeddings)

    def aux_loss(self, cdr_embeddings, aux_targets):
        """Compute auxiliary reconstruction loss.

        Args:
            cdr_embeddings: (N_cdr, hidden_dim) GNN output for CDR positions
            aux_targets: (N_cdr, aux_dim) ground truth structural features

        Returns:
            loss: scalar MSE loss
        """
        if cdr_embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=cdr_embeddings.device)
        predicted = self.predictor(cdr_embeddings)
        return F.mse_loss(predicted, aux_targets)
