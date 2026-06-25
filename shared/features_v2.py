"""Enhanced feature loading module for antibody CDR design models.

Loads v2 features with comprehensive physics-based features:
- Identity: 55D (AA onehot + physicochemical + BLOSUM)
- Torsions: 14D (backbone + chi)
- Structure: 7D (DSSP SS + SASA + burial + packing + contact_order)
- Surface: 12D (normal + curvature + shape_index + chemistry + electrostatic)
- Electrostatics: 4D (potential + field)
- H-bonds: 4D (donors + acceptors + satisfaction + backbone)
- Dynamics: 5D (mode fluctuations + collectivity + coupling)
- Complementarity: 4D (Sc + electrostatic + hydrophobic + aromatic)
TOTAL: 105D per residue (without labels)

Additional masks:
- interface_mask: (N,) binary mask indicating which residues are at the
  Ab-Ag interface (within 8A). Use this to distinguish "no complementarity
  computed" vs "complementarity value is actually zero".

Usage:
    from shared.features_v2 import V2FeatureLoader, ProteinFeatureEncoderV2

    loader = V2FeatureLoader('/path/to/rich_features_v2.pt')
    features = loader.get_features(complex_id)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple


class V2FeatureLoader:
    """Loads and provides enhanced features for complexes."""

    # Feature dimensions breakdown
    FEATURE_DIMS = {
        # Identity (55D)
        'aa_onehot': 20,
        'aa_physicochemical': 15,
        'blosum_row': 20,
        # Torsions (14D)
        'backbone_torsions': 6,
        'chi_angles': 8,
        # Structure (8D)
        'secondary_structure': 3,
        'relative_sasa': 1,
        'burial_fraction': 1,
        'local_packing': 1,
        'contact_order': 1,
        # Surface (16D)
        'surface_normal': 3,
        'surface_curvature': 2,
        'shape_index': 1,
        'surface_chemistry': 5,
        'surface_electrostatic': 1,
        # Electrostatics (4D)
        'electrostatic_potential': 1,
        'electrostatic_field': 3,
        # H-bonds (4D)
        'hbond_donors': 1,
        'hbond_acceptors': 1,
        'hbond_satisfaction': 1,
        'backbone_hbonds': 1,
        # Dynamics (6D)
        'mode_fluctuations': 3,
        'collectivity': 1,
        'mode_coupling': 1,
        # Complementarity (4D)
        'shape_complementarity': 1,
        'electrostatic_complementarity': 1,
        'hydrophobic_complementarity': 1,
        'aromatic_stacking': 1,
    }

    # Total feature dimension (without segment_type and cdr_mask)
    # Actual: 55 + 14 + 7 + 12 + 4 + 4 + 5 + 4 = 105D
    NODE_FEATURE_DIM = 105  # Sum of all above
    FULL_NODE_DIM = 108     # With segment_type (3)

    # Feature groups for selective loading
    FEATURE_GROUPS = {
        'identity': ['aa_onehot', 'aa_physicochemical', 'blosum_row'],
        'torsions': ['backbone_torsions', 'chi_angles'],
        'structure': ['secondary_structure', 'relative_sasa', 'burial_fraction', 'local_packing', 'contact_order'],
        'surface': ['surface_normal', 'surface_curvature', 'shape_index', 'surface_chemistry', 'surface_electrostatic'],
        'electrostatics': ['electrostatic_potential', 'electrostatic_field'],
        'hbonds': ['hbond_donors', 'hbond_acceptors', 'hbond_satisfaction', 'backbone_hbonds'],
        'dynamics': ['mode_fluctuations', 'collectivity', 'mode_coupling'],
        'complementarity': ['shape_complementarity', 'electrostatic_complementarity', 'hydrophobic_complementarity', 'aromatic_stacking'],
    }

    def __init__(self, features_path: str, device: str = 'cpu',
                 feature_groups: Optional[List[str]] = None):
        """Load v2 features from preprocessed file or checkpoint directory.

        Args:
            features_path: Path to rich_features_v2.pt OR a checkpoint file.
                          If a checkpoint file (ckpt_*.pt), will merge all
                          checkpoints from the same directory.
            device: Device to store features on
            feature_groups: Optional list of feature groups to include.
                           If None, includes all groups.
        """
        import os
        import glob

        self.device = device
        self.feature_groups = feature_groups
        self.normalization_stats = {}
        self.version = 1

        # Check if this is a checkpoint file - if so, merge all checkpoints
        basename = os.path.basename(features_path)
        if basename.startswith('ckpt_') and basename.endswith('.pt'):
            # Merge all checkpoints from the directory
            ckpt_dir = os.path.dirname(features_path)
            ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pt')))

            self.complexes = {}
            for ckpt_path in ckpt_files:
                data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                self.complexes.update(data)

            print(f"Merged {len(ckpt_files)} checkpoints: {len(self.complexes)} complexes")
        else:
            # Load single file (final features or single checkpoint)
            data = torch.load(features_path, map_location='cpu', weights_only=False)

            if 'features' in data:
                self.complexes = data['features']
                self.normalization_stats = data.get('normalization_stats', {})
                self.version = data.get('version', 1)
            else:
                self.complexes = data

        self.complex_ids = list(self.complexes.keys())

        # Compute actual feature dim based on selected groups
        if feature_groups is not None:
            selected_features = []
            for group in feature_groups:
                if group in self.FEATURE_GROUPS:
                    selected_features.extend(self.FEATURE_GROUPS[group])
            self.selected_features = selected_features
            self.node_feature_dim = sum(self.FEATURE_DIMS.get(f, 0) for f in selected_features)
        else:
            self.selected_features = None
            self.node_feature_dim = self.NODE_FEATURE_DIM

        print(f"Loaded {len(self.complex_ids)} complexes from {features_path}")
        print(f"Feature dimension: {self.node_feature_dim}D (v{self.version})")

    def get_features(self, complex_id: str,
                     include_complementarity: bool = True,
                     include_dynamics: bool = True) -> Dict[str, torch.Tensor]:
        """Get features for a single complex.

        Args:
            complex_id: Complex identifier
            include_complementarity: Include Sc and complementarity features
            include_dynamics: Include normal mode features

        Returns dict with:
            - node_features: (N, D) concatenated features
            - segment_type: (N, 3) segment one-hot
            - cdr_mask: (N, 6) CDR membership
            - Coordinates and other tensors
        """
        if complex_id not in self.complexes:
            raise KeyError(f"Complex {complex_id} not found")

        feats = self.complexes[complex_id]

        # Build feature tensor by concatenating selected groups
        feature_parts = []

        # Helper to ensure float32
        def to_float(t):
            if isinstance(t, torch.Tensor):
                return t.float()
            return t

        # Identity (55D)
        feature_parts.append(to_float(feats['aa_onehot']))
        feature_parts.append(to_float(feats['aa_physicochemical']))
        feature_parts.append(to_float(feats['blosum_row']))

        # Torsions (14D)
        feature_parts.append(to_float(feats['backbone_torsions']))
        feature_parts.append(to_float(feats['chi_angles']))

        # Structure (7D)
        feature_parts.append(to_float(feats['secondary_structure']))
        feature_parts.append(self._ensure_2d(feats['relative_sasa']))
        feature_parts.append(self._ensure_2d(feats['burial_fraction']))
        feature_parts.append(self._ensure_2d(feats['local_packing']))
        feature_parts.append(self._ensure_2d(feats['contact_order']))

        # Surface (12D)
        feature_parts.append(to_float(feats['surface_normal']))
        feature_parts.append(to_float(feats['surface_curvature']))
        feature_parts.append(self._ensure_2d(feats['shape_index']))
        feature_parts.append(to_float(feats['surface_chemistry']))
        feature_parts.append(self._ensure_2d(feats['surface_electrostatic']))

        # Electrostatics (4D)
        feature_parts.append(self._ensure_2d(feats['electrostatic_potential']))
        feature_parts.append(to_float(feats['electrostatic_field']))

        # H-bonds (4D)
        feature_parts.append(self._ensure_2d(feats['hbond_donors']))
        feature_parts.append(self._ensure_2d(feats['hbond_acceptors']))
        feature_parts.append(self._ensure_2d(feats['hbond_satisfaction']))
        feature_parts.append(self._ensure_2d(feats['backbone_hbonds']))

        # Dynamics (5D) - optional
        if include_dynamics:
            feature_parts.append(to_float(feats['mode_fluctuations']))
            feature_parts.append(self._ensure_2d(feats['collectivity']))
            feature_parts.append(self._ensure_2d(feats['mode_coupling']))

        # Complementarity (4D) - optional
        if include_complementarity:
            feature_parts.append(self._ensure_2d(feats['shape_complementarity']))
            feature_parts.append(self._ensure_2d(feats['electrostatic_complementarity']))
            feature_parts.append(self._ensure_2d(feats['hydrophobic_complementarity']))
            feature_parts.append(self._ensure_2d(feats['aromatic_stacking']))

        node_features = torch.cat(feature_parts, dim=-1)

        return {
            'node_features': node_features,
            'segment_type': feats['segment_type'],
            'cdr_mask': feats['cdr_mask'],
            'cb_coords': feats['cb_coords'],
            'ca_coords': feats['ca_coords'],
            'atom14_coords': feats['atom14_coords'],
            'atom14_mask': feats['atom14_mask'],
            'chi_mask': feats['chi_mask'],
            'epitope_mask': feats['epitope_mask'],
            'paratope_mask': feats['paratope_mask'],
            'sequence': feats['sequence'],
            'n_heavy': feats['n_heavy'],
            'n_light': feats['n_light'],
            'n_antigen': feats['n_antigen'],
            # Additional v2 features for model-specific use
            'shape_complementarity': feats.get('shape_complementarity'),
            'mode_fluctuations': feats.get('mode_fluctuations'),
            'mode_coupling': feats.get('mode_coupling'),
            # Interface mask (1 = at interface, 0 = not at interface)
            # Use to distinguish "no complementarity computed" vs "value is zero"
            'interface_mask': feats.get('interface_mask'),
        }

    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 2D (N, 1) for concatenation and float32."""
        tensor = tensor.float()  # Ensure float32
        if tensor.dim() == 1:
            return tensor.unsqueeze(-1)
        return tensor

    def __contains__(self, complex_id: str) -> bool:
        return complex_id in self.complexes

    def __len__(self) -> int:
        return len(self.complex_ids)


class ProteinFeatureEncoderV2(nn.Module):
    """Encodes v2 enhanced features into model hidden dimension.

    Supports:
    - Full 111D features (all groups)
    - Selective feature groups
    - Global token embeddings for BOH, BOL, BOA
    - Optional feature dropout for regularization
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int = 108,  # 105 node + 3 segment
        dropout: float = 0.1,
        n_global_tokens: int = 3,
        use_layer_norm: bool = True,
        feature_dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            input_dim: Input feature dimension (default 108 = 105 node + 3 segment)
            dropout: Dropout probability
            n_global_tokens: Number of global tokens (BOH, BOL, BOA)
            use_layer_norm: Whether to apply layer normalization
            feature_dropout: Dropout applied to input features (for regularization)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_global_tokens = n_global_tokens

        # Feature dropout (applied to input)
        self.feature_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0 else nn.Identity()

        # Two-layer projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global token embeddings
        self.global_embed = nn.Embedding(n_global_tokens, hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        node_features: torch.Tensor,
        segment_type: torch.Tensor,
        global_mask: Optional[torch.Tensor] = None,
        global_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode features to hidden dimension.

        Args:
            node_features: (N, D) pre-computed features
            segment_type: (N, 3) segment one-hot
            global_mask: (N_total,) bool mask for global tokens
            global_types: (N_global,) indices for global tokens

        Returns:
            (N_total, hidden_dim) encoded features
        """
        # Apply feature dropout
        node_features = self.feature_dropout(node_features)

        # Concatenate with segment type
        x = torch.cat([node_features, segment_type], dim=-1)

        # Project
        h = self.proj(x)
        h = self.norm(h)

        if global_mask is None:
            return h

        # Insert global token embeddings
        n_total = global_mask.shape[0]
        device = h.device
        output = torch.zeros(n_total, self.hidden_dim, device=device, dtype=h.dtype)

        output[~global_mask] = h

        if global_types is not None and global_types.numel() > 0:
            output[global_mask] = self.global_embed(global_types)

        return output


class FeatureGroupEncoder(nn.Module):
    """Encodes specific feature groups with separate projections.

    Useful for models that want to process different feature types
    through different pathways (e.g., surface features through surface encoder,
    dynamics features through dynamics encoder).
    """

    FEATURE_GROUP_DIMS = {
        'identity': 55,
        'torsions': 14,
        'structure': 8,
        'surface': 16,
        'electrostatics': 4,
        'hbonds': 4,
        'dynamics': 6,
        'complementarity': 4,
    }

    def __init__(
        self,
        hidden_dim: int,
        feature_groups: List[str],
        dropout: float = 0.1,
        use_separate_proj: bool = True,
    ):
        """
        Args:
            hidden_dim: Output hidden dimension
            feature_groups: List of feature groups to encode
            dropout: Dropout probability
            use_separate_proj: If True, each group gets its own projection
        """
        super().__init__()

        self.feature_groups = feature_groups
        self.hidden_dim = hidden_dim
        self.use_separate_proj = use_separate_proj

        if use_separate_proj:
            self.projections = nn.ModuleDict()
            for group in feature_groups:
                dim = self.FEATURE_GROUP_DIMS[group]
                self.projections[group] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
        else:
            total_dim = sum(self.FEATURE_GROUP_DIMS[g] for g in feature_groups)
            self.projection = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode feature groups.

        Args:
            feature_dict: Dict mapping group names to feature tensors

        Returns:
            (N, hidden_dim) encoded features
        """
        if self.use_separate_proj:
            encoded = []
            for group in self.feature_groups:
                if group in feature_dict:
                    encoded.append(self.projections[group](feature_dict[group]))
            h = torch.stack(encoded, dim=-1).mean(dim=-1)  # Average pooling
        else:
            parts = [feature_dict[g] for g in self.feature_groups if g in feature_dict]
            x = torch.cat(parts, dim=-1)
            h = self.projection(x)

        return self.norm(h)


class ComplementarityEncoder(nn.Module):
    """Specialized encoder for complementarity features.

    Processes shape complementarity, electrostatic complementarity,
    hydrophobic matching, and aromatic stacking scores.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Input: 4D complementarity features
        self.proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        shape_comp: torch.Tensor,
        elec_comp: torch.Tensor,
        hydro_comp: torch.Tensor,
        aromatic: torch.Tensor,
    ) -> torch.Tensor:
        """Encode complementarity features.

        All inputs are (N,) or (N, 1) tensors.
        Returns (N, hidden_dim) encoded complementarity.
        """
        # Stack into (N, 4)
        if shape_comp.dim() == 1:
            shape_comp = shape_comp.unsqueeze(-1)
        if elec_comp.dim() == 1:
            elec_comp = elec_comp.unsqueeze(-1)
        if hydro_comp.dim() == 1:
            hydro_comp = hydro_comp.unsqueeze(-1)
        if aromatic.dim() == 1:
            aromatic = aromatic.unsqueeze(-1)

        x = torch.cat([shape_comp, elec_comp, hydro_comp, aromatic], dim=-1)
        h = self.proj(x)
        return self.norm(h)


class DynamicsEncoder(nn.Module):
    """Specialized encoder for dynamics features.

    Processes normal mode fluctuations, collectivity, and mode coupling.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Input: 6D dynamics features (3 mode_fluct + 1 collectivity + 1 coupling + 1 reserved)
        self.proj = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mode_fluctuations: torch.Tensor,
        collectivity: torch.Tensor,
        mode_coupling: torch.Tensor,
    ) -> torch.Tensor:
        """Encode dynamics features.

        Args:
            mode_fluctuations: (N, 3) top-3 normal mode fluctuations
            collectivity: (N,) or (N, 1) collectivity score
            mode_coupling: (N,) or (N, 1) coupling to antigen

        Returns:
            (N, hidden_dim) encoded dynamics
        """
        if collectivity.dim() == 1:
            collectivity = collectivity.unsqueeze(-1)
        if mode_coupling.dim() == 1:
            mode_coupling = mode_coupling.unsqueeze(-1)

        # Pad to 6D if needed
        reserved = torch.zeros_like(collectivity)

        x = torch.cat([mode_fluctuations, collectivity, mode_coupling, reserved], dim=-1)
        h = self.proj(x)
        return self.norm(h)


class SurfaceEncoder(nn.Module):
    """Specialized encoder for surface features.

    Processes surface geometry (normal, curvature, shape index)
    and surface chemistry (electrostatics, hydropathy, etc.).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Geometry: 3 normal + 2 curvature + 1 shape_index = 6D
        self.geometry_proj = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
        )

        # Chemistry: 5 chemistry + 1 electrostatic = 6D
        self.chemistry_proj = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
        )

        # Combined
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        surface_normal: torch.Tensor,
        surface_curvature: torch.Tensor,
        shape_index: torch.Tensor,
        surface_chemistry: torch.Tensor,
        surface_electrostatic: torch.Tensor,
    ) -> torch.Tensor:
        """Encode surface features.

        Returns (N, hidden_dim) encoded surface representation.
        """
        if shape_index.dim() == 1:
            shape_index = shape_index.unsqueeze(-1)
        if surface_electrostatic.dim() == 1:
            surface_electrostatic = surface_electrostatic.unsqueeze(-1)

        # Geometry path
        geom = torch.cat([surface_normal, surface_curvature, shape_index], dim=-1)
        h_geom = self.geometry_proj(geom)

        # Chemistry path
        chem = torch.cat([surface_chemistry, surface_electrostatic], dim=-1)
        h_chem = self.chemistry_proj(chem)

        # Combine
        h = torch.cat([h_geom, h_chem], dim=-1)
        h = self.combine(h)
        return self.norm(h)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mask_cdr_structure_features(
    node_features: torch.Tensor,
    cdr_mask: torch.Tensor,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Mask structure-derived features for CDR residues.

    At inference time, CDR structure is unknown. This zeros out
    torsions, surface, dynamics, and complementarity features while
    keeping sequence-based features (AA identity, BLOSUM).

    Feature layout (111D):
        0-54: Identity (keep)
        55-68: Torsions (mask)
        69-76: Structure (mask SS, keep SASA heuristic)
        77-92: Surface (mask)
        93-96: Electrostatics (mask)
        97-100: H-bonds (mask)
        101-106: Dynamics (mask)
        107-110: Complementarity (mask)
    """
    SEQUENCE_FEATURES_END = 55  # First 55D are sequence-based

    is_cdr = cdr_mask.any(dim=-1)  # (N,)

    masked = node_features.clone()

    if node_features.dim() == 2:
        masked[is_cdr, SEQUENCE_FEATURES_END:] = mask_value
    else:
        # Batch dimension
        is_cdr_expanded = is_cdr.unsqueeze(-1).expand(-1, -1, node_features.size(-1) - SEQUENCE_FEATURES_END)
        masked[..., SEQUENCE_FEATURES_END:] = masked[..., SEQUENCE_FEATURES_END:].masked_fill(is_cdr_expanded, mask_value)

    return masked


def get_feature_indices(group_name: str) -> Tuple[int, int]:
    """Get start and end indices for a feature group in the 111D tensor.

    Returns (start, end) indices for slicing.
    """
    FEATURE_RANGES = {
        'identity': (0, 55),
        'torsions': (55, 69),
        'structure': (69, 77),
        'surface': (77, 93),
        'electrostatics': (93, 97),
        'hbonds': (97, 101),
        'dynamics': (101, 107),
        'complementarity': (107, 111),
    }
    return FEATURE_RANGES.get(group_name, (0, 0))


def split_features_by_group(node_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split concatenated features back into named groups.

    Args:
        node_features: (N, 111) concatenated features

    Returns:
        Dict mapping group names to feature tensors
    """
    return {
        'identity': node_features[..., 0:55],
        'torsions': node_features[..., 55:69],
        'structure': node_features[..., 69:77],
        'surface': node_features[..., 77:93],
        'electrostatics': node_features[..., 93:97],
        'hbonds': node_features[..., 97:101],
        'dynamics': node_features[..., 101:107],
        'complementarity': node_features[..., 107:111],
    }


# ============================================================================
# SPLIT UTILITIES
# ============================================================================

def load_split_ids(
    split_name: str,
    splits_dir: str = '/home/exouser/data/chimera/splits',
    available_ids: Optional[List[str]] = None,
    random_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """Load train/val/test IDs for a given split.

    Args:
        split_name: One of 'epitope_group', 'antigen_fold', 'temporal', or 'random'
        splits_dir: Directory containing split JSON files
        available_ids: For 'random' split, list of available complex IDs.
                      If None, must provide for 'random' split.
        random_seed: Random seed for 'random' split reproducibility
        train_ratio: Train set ratio for 'random' split (default 0.8)
        val_ratio: Validation set ratio for 'random' split (default 0.1)

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to lists of complex IDs
    """
    import json
    import random
    import os

    if split_name == 'random':
        if available_ids is None:
            raise ValueError("available_ids required for 'random' split")

        # Shuffle and split
        ids = list(available_ids)
        random.seed(random_seed)
        random.shuffle(ids)

        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return {
            'train': ids[:n_train],
            'val': ids[n_train:n_train + n_val],
            'test': ids[n_train + n_val:],
        }

    # Load from predefined split file
    split_path = os.path.join(splits_dir, f'{split_name}.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, 'r') as f:
        splits = json.load(f)

    result = {
        'train': splits['train'],
        'val': splits['val'],
        'test': splits['test'],
    }

    # Filter to available IDs if provided
    if available_ids is not None:
        available_set = set(available_ids)
        result = {
            k: [cid for cid in v if cid in available_set]
            for k, v in result.items()
        }

    return result


def get_available_complex_ids(
    checkpoints_dir: str = '/home/exouser/data/chimera/processed/checkpoints_v2',
    final_path: str = '/home/exouser/data/chimera/processed/rich_features_v2.pt',
) -> List[str]:
    """Get list of complex IDs available in preprocessed data.

    Checks final file first, falls back to checkpoints.

    Args:
        checkpoints_dir: Directory containing checkpoint .pt files
        final_path: Path to final merged features file

    Returns:
        List of available complex IDs
    """
    import os
    import glob

    # Try final file first
    if os.path.exists(final_path):
        data = torch.load(final_path, map_location='cpu', weights_only=False)
        if 'features' in data:
            return list(data['features'].keys())
        return list(data.keys())

    # Fall back to checkpoints
    all_ids = []
    ckpt_files = sorted(glob.glob(os.path.join(checkpoints_dir, 'ckpt_*.pt')))
    for ckpt_path in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        all_ids.extend(ckpt.keys())

    return all_ids
