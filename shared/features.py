"""Shared feature loading module for all 5 GenBio PLM-free models.

Loads rich_features.pt and provides unified feature tensors for models.
All models use the same rich input representation instead of learned embeddings.

Usage:
    from shared.features import RichFeatureLoader

    # During dataset __init__
    loader = RichFeatureLoader('/path/to/rich_features.pt')

    # During __getitem__
    features = loader.get_features(complex_id)
    node_feats = features['node_features']  # (N, 84) - input features
    coords = features['cb_coords']          # (N, 3) - CB coordinates
    ...
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class RichFeatureLoader:
    """Loads and provides rich features for complexes."""

    # Feature dimensions
    # Node features (84D total - no CDR or supervision labels):
    #   - aa_onehot: 20
    #   - aa_physicochemical: 15
    #   - blosum_row: 20
    #   - backbone_torsions: 6
    #   - chi_angles: 8
    #   - secondary_structure: 3
    #   - relative_sasa: 1
    #   - local_packing: 1
    #   - surface_normal: 3
    #   - surface_curvature: 2
    #   - surface_chemistry: 5
    #
    # Plus segment_type (3) added during batching based on chain
    # CDR mask (6) used only for masking, not as input

    NODE_FEATURE_DIM = 84  # Base features without segment type
    FULL_NODE_DIM = 87     # With segment type (3)

    def __init__(self, features_path: str, device: str = 'cpu'):
        """Load rich features from preprocessed file.

        Args:
            features_path: Path to rich_features.pt
            device: Device to store features on ('cpu' recommended for memory)
        """
        self.device = device

        # Load all features
        data = torch.load(features_path, map_location='cpu')

        # Store complex features dict
        if 'features' in data:
            self.complexes = data['features']
            self.normalization_stats = data.get('normalization_stats', {})
        else:
            # Direct dict format (complex_id -> features)
            self.complexes = data
            self.normalization_stats = {}

        self.complex_ids = list(self.complexes.keys())
        print(f"Loaded {len(self.complex_ids)} complexes from {features_path}")

    def get_features(self, complex_id: str) -> Dict[str, torch.Tensor]:
        """Get features for a single complex.

        Returns dict with:
            - node_features: (N, 84) concatenated scalar features
            - segment_type: (N, 3) one-hot segment indicator [H, L, AG]
            - cdr_mask: (N, 6) CDR membership [H1,H2,H3,L1,L2,L3]
            - cb_coords: (N, 3) CB coordinates
            - ca_coords: (N, 3) CA coordinates
            - atom14_coords: (N, 14, 3) all atom coordinates
            - atom14_mask: (N, 14) atom existence mask
            - sc_centroid: (N, 3) side-chain centroid
            - sc_direction: (N, 3) unit vector CA->centroid
            - chi_mask: (N, 4) which chi angles exist
            - epitope_mask: (N,) supervision label
            - paratope_mask: (N,) supervision label
            - sequence: str
            - n_heavy: int
            - n_light: int
            - n_antigen: int
        """
        if complex_id not in self.complexes:
            raise KeyError(f"Complex {complex_id} not found in features")

        feats = self.complexes[complex_id]

        # Build node features tensor (84D)
        node_features = torch.cat([
            feats['aa_onehot'],           # (N, 20)
            feats['aa_physicochemical'],  # (N, 15)
            feats['blosum_row'],          # (N, 20)
            feats['backbone_torsions'],   # (N, 6)
            feats['chi_angles'],          # (N, 8)
            feats['secondary_structure'], # (N, 3)
            feats['relative_sasa'].unsqueeze(-1) if feats['relative_sasa'].dim() == 1 else feats['relative_sasa'],  # (N, 1)
            feats['local_packing'].unsqueeze(-1) if feats['local_packing'].dim() == 1 else feats['local_packing'],  # (N, 1)
            feats['surface_normal'],      # (N, 3)
            feats['surface_curvature'],   # (N, 2)
            feats['surface_chemistry'],   # (N, 5)
        ], dim=-1)

        return {
            'node_features': node_features,
            'segment_type': feats['segment_type'],
            'cdr_mask': feats['cdr_mask'],
            'cb_coords': feats['cb_coords'],
            'ca_coords': feats['ca_coords'],
            'atom14_coords': feats['atom14_coords'],
            'atom14_mask': feats['atom14_mask'],
            'sc_centroid': feats['sc_centroid'],
            'sc_direction': feats['sc_direction'],
            'chi_mask': feats['chi_mask'],
            'epitope_mask': feats['epitope_mask'],
            'paratope_mask': feats['paratope_mask'],
            'sequence': feats['sequence'],
            'n_heavy': feats['n_heavy'],
            'n_light': feats['n_light'],
            'n_antigen': feats['n_antigen'],
        }

    def __contains__(self, complex_id: str) -> bool:
        return complex_id in self.complexes

    def __len__(self) -> int:
        return len(self.complex_ids)


class ProteinFeatureEncoder(nn.Module):
    """Encodes rich features into model hidden dimension.

    This replaces the old nn.Embedding(27, hidden) + ESM-2 approach.
    Uses a simple linear projection of pre-computed features.

    Handles global tokens (BOH, BOL, BOA) which are not in rich features -
    these get learned embeddings while regular residues use rich features.
    """

    def __init__(self, hidden_dim: int, input_dim: int = 87, dropout: float = 0.1,
                 n_global_tokens: int = 3):
        """
        Args:
            hidden_dim: Model hidden dimension
            input_dim: Input feature dimension (default 87 = 84 node + 3 segment)
            dropout: Dropout probability
            n_global_tokens: Number of global tokens (BOH, BOL, BOA)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_global_tokens = n_global_tokens

        # Two-layer projection for rich features
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned embeddings for global tokens (BOH=0, BOL=1, BOA=2)
        self.global_embed = nn.Embedding(n_global_tokens, hidden_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, segment_type: torch.Tensor,
                global_mask: Optional[torch.Tensor] = None,
                global_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode node features to hidden dimension.

        Args:
            node_features: (N, 84) pre-computed features (regular residues only)
            segment_type: (N, 3) segment one-hot (regular residues only)
            global_mask: (N_total,) bool mask indicating global token positions
            global_types: (N_global,) indices for global tokens (0=BOH, 1=BOL, 2=BOA)

        Returns:
            (N_total, hidden_dim) encoded features with global tokens inserted
        """
        # Concatenate node features with segment type
        x = torch.cat([node_features, segment_type], dim=-1)

        # Project to hidden dim
        h = self.proj(x)
        h = self.norm(h)

        # If no global tokens, return as-is
        if global_mask is None:
            return h

        # Insert global token embeddings
        n_total = global_mask.shape[0]
        device = h.device
        output = torch.zeros(n_total, self.hidden_dim, device=device, dtype=h.dtype)

        # Fill regular residue positions
        output[~global_mask] = h

        # Fill global token positions
        if global_types is not None and global_types.numel() > 0:
            output[global_mask] = self.global_embed(global_types)

        return output


class RichFeatureAdapter:
    """Adapts rich features to match model's expected sequence format.

    The models expect S with global tokens (BOH, BOL, BOA) at chain boundaries.
    Rich features don't include these. This adapter:
    1. Takes rich features (N_residues, D)
    2. Takes S with global tokens (N_total,) where N_total = N_residues + 3
    3. Returns features aligned with S, with placeholders for global tokens
    """

    # Global token indices from VOCAB
    BOH_IDX = 24  # VOCAB.symbol_to_idx(VOCAB.BOH)
    BOL_IDX = 25  # VOCAB.symbol_to_idx(VOCAB.BOL)
    BOA_IDX = 26  # VOCAB.symbol_to_idx(VOCAB.BOA)

    @staticmethod
    def get_global_mask(S: torch.Tensor) -> torch.Tensor:
        """Get boolean mask for global token positions in S."""
        return (S == RichFeatureAdapter.BOH_IDX) | \
               (S == RichFeatureAdapter.BOL_IDX) | \
               (S == RichFeatureAdapter.BOA_IDX)

    @staticmethod
    def get_global_types(S: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
        """Get global token types (0=BOH, 1=BOL, 2=BOA) for masked positions."""
        global_S = S[global_mask]
        types = torch.zeros_like(global_S)
        types[global_S == RichFeatureAdapter.BOH_IDX] = 0
        types[global_S == RichFeatureAdapter.BOL_IDX] = 1
        types[global_S == RichFeatureAdapter.BOA_IDX] = 2
        return types

    @staticmethod
    def expand_features(features: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Expand rich features to match S length by inserting zeros at global positions.

        Args:
            features: (N_residues, D) rich features without global tokens
            S: (N_total,) sequence with global tokens

        Returns:
            (N_total, D) features with zeros at global positions
        """
        global_mask = RichFeatureAdapter.get_global_mask(S)
        n_total = S.shape[0]
        d = features.shape[-1]
        device = features.device

        output = torch.zeros(n_total, d, device=device, dtype=features.dtype)
        output[~global_mask] = features

        return output


def compute_edge_features(
    cb_coords: torch.Tensor,
    segment_type: torch.Tensor,
    cdr_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_cutoff: float = 10.0,
    n_rbf: int = 16,
) -> torch.Tensor:
    """Compute edge features for GNN.

    Args:
        cb_coords: (N, 3) CB coordinates
        segment_type: (N, 3) segment one-hot [H, L, AG]
        cdr_mask: (N, 6) CDR membership (any column > 0 = CDR residue)
        edge_index: (2, E) edge indices
        edge_cutoff: Distance cutoff for RBF
        n_rbf: Number of RBF bins

    Returns:
        edge_feats: (E, n_rbf + 10) edge features
            - n_rbf: distance RBF encoding
            - 10: edge type one-hot (H-H, H-L, H-AG, L-L, L-AG, AG-AG,
                  CDR-CDR, CDR-FW, FW-AG, CDR-AG)
    """
    src, dst = edge_index

    # Distance features (RBF)
    dist = (cb_coords[src] - cb_coords[dst]).norm(dim=-1)
    rbf = rbf_encoding(dist, cutoff=edge_cutoff, n_bins=n_rbf)

    # Edge type based on segment membership
    src_seg = segment_type[src].argmax(dim=-1)  # 0=H, 1=L, 2=AG
    dst_seg = segment_type[dst].argmax(dim=-1)

    # CDR membership
    src_is_cdr = cdr_mask[src].any(dim=-1)
    dst_is_cdr = cdr_mask[dst].any(dim=-1)

    # 10 edge types
    edge_types = torch.zeros(len(src), 10, device=cb_coords.device)

    # Chain-chain types (6)
    for i, (s, d) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]):
        mask = ((src_seg == s) & (dst_seg == d)) | ((src_seg == d) & (dst_seg == s))
        edge_types[mask, i] = 1.0

    # CDR-related types (4)
    ab_mask = (src_seg < 2) & (dst_seg < 2)  # Both are antibody
    edge_types[ab_mask & src_is_cdr & dst_is_cdr, 6] = 1.0  # CDR-CDR
    edge_types[ab_mask & (src_is_cdr ^ dst_is_cdr), 7] = 1.0  # CDR-FW

    ag_ab_mask = (src_seg == 2) ^ (dst_seg == 2)  # One is AG
    is_cdr_side = torch.where(src_seg == 2, dst_is_cdr, src_is_cdr)
    edge_types[ag_ab_mask & ~is_cdr_side, 8] = 1.0  # FW-AG
    edge_types[ag_ab_mask & is_cdr_side, 9] = 1.0   # CDR-AG

    return torch.cat([rbf, edge_types], dim=-1)


def rbf_encoding(distances: torch.Tensor, cutoff: float = 10.0, n_bins: int = 16) -> torch.Tensor:
    """RBF distance encoding.

    Args:
        distances: (E,) pairwise distances
        cutoff: Maximum distance for encoding
        n_bins: Number of RBF centers

    Returns:
        (E, n_bins) RBF encoding
    """
    centers = torch.linspace(0, cutoff, n_bins, device=distances.device)
    gamma = 1.0 / (cutoff / n_bins)

    # (E, n_bins)
    rbf = torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)
    return rbf


def mask_cdr_features(
    node_features: torch.Tensor,
    cdr_mask: torch.Tensor,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Mask out CDR features to prevent data leakage.

    At inference time, CDR structure is unknown. This function zeros out
    structure-derived features (torsions, surface) for CDR residues while
    keeping sequence-based features (AA identity, BLOSUM).

    Args:
        node_features: (N, 84) or (B, N, 84) features
        cdr_mask: (N, 6) or (B, N, 6) CDR membership
        mask_value: Value to replace masked features with

    Returns:
        Masked node features with same shape
    """
    # Feature indices that are structure-derived (should be masked for CDRs)
    # aa_onehot: 0-19 (keep - sequence)
    # aa_physicochemical: 20-34 (keep - sequence)
    # blosum_row: 35-54 (keep - sequence)
    # backbone_torsions: 55-60 (mask - structure)
    # chi_angles: 61-67 (mask - structure)
    # secondary_structure: 68-70 (mask - structure)
    # relative_sasa: 71 (mask - structure)
    # local_packing: 72 (mask - structure)
    # surface_normal: 73-75 (mask - structure)
    # surface_curvature: 76-77 (mask - structure)
    # surface_chemistry: 78-82 (mask - structure)

    STRUCTURE_FEATURE_START = 55  # First structure-derived feature

    # Find CDR residues
    is_cdr = cdr_mask.any(dim=-1, keepdim=True)  # (N, 1) or (B, N, 1)

    # Create masked features
    masked = node_features.clone()

    if node_features.dim() == 2:
        # (N, 84)
        masked[is_cdr.squeeze(-1), STRUCTURE_FEATURE_START:] = mask_value
    else:
        # (B, N, 84)
        masked = masked.masked_fill(
            is_cdr.unsqueeze(-1).expand(-1, -1, 84 - STRUCTURE_FEATURE_START),
            mask_value
        )
        # Only mask structure features
        seq_feats = masked[..., :STRUCTURE_FEATURE_START]
        struct_feats = masked[..., STRUCTURE_FEATURE_START:]
        struct_feats = struct_feats.masked_fill(is_cdr.expand(-1, -1, struct_feats.size(-1)), mask_value)
        masked = torch.cat([seq_feats, struct_feats], dim=-1)

    return masked


def get_cdr_sequence_mask(cdr_type: str, cdr_mask: torch.Tensor) -> torch.Tensor:
    """Get mask for a specific CDR type.

    Args:
        cdr_type: One of 'H1', 'H2', 'H3', 'L1', 'L2', 'L3'
        cdr_mask: (N, 6) CDR membership tensor

    Returns:
        (N,) boolean mask for the specified CDR
    """
    CDR_INDEX = {'H1': 0, 'H2': 1, 'H3': 2, 'L1': 3, 'L2': 4, 'L3': 5}
    idx = CDR_INDEX[cdr_type]
    return cdr_mask[:, idx] > 0.5
