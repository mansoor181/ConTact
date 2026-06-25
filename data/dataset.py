"""Feature dataset for CompFirst.

Uses pre-computed 105D features per residue instead of learned embeddings.
"""
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Optional

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..', '..'))
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, _GENBIO_ROOT)

from shared.features_v2 import V2FeatureLoader
from shared.sparse_features import extract_aux_targets, mask_cdr_features
from data.pdb_utils import VOCAB


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset using pre-computed v2 features."""

    def __init__(
        self,
        complex_ids: List[str],
        v2_features_path: str,
        complex_features_dir: str,
        cdr_type: str = '3',
        numbering_scheme: str = 'imgt',
        germline_prior=None,
    ):
        super().__init__()
        self.complex_ids = complex_ids
        self.cdr_type = cdr_type
        self.numbering_scheme = numbering_scheme
        self.complex_features_dir = complex_features_dir
        self.germline_prior = germline_prior

        self.feature_loader = V2FeatureLoader(v2_features_path)

        self.available_ids = [
            cid for cid in complex_ids
            if cid in self.feature_loader
        ]
        print(f"FeatureDataset: {len(self.available_ids)}/{len(complex_ids)} complexes available")

    def __len__(self):
        return len(self.available_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        complex_id = self.available_ids[idx]

        feats = self.feature_loader.get_features(complex_id)

        cf_path = os.path.join(self.complex_features_dir, f'{complex_id}.pt')
        complex_data = torch.load(cf_path, map_location='cpu')

        n_heavy = feats['n_heavy']
        n_light = feats['n_light']
        n_antigen = feats['n_antigen']

        atom14 = feats['atom14_coords']
        ca_coords = feats['ca_coords']

        X_residues = atom14[:, :4, :]

        ca_centroid = ca_coords.mean(dim=0)
        X_residues = X_residues - ca_centroid.unsqueeze(0).unsqueeze(1)

        n_total = X_residues.shape[0] + 3
        X = torch.zeros(n_total, 4, 3, dtype=torch.float32)
        S = torch.zeros(n_total, dtype=torch.long)
        node_features = torch.zeros(n_total, 105, dtype=torch.float32)
        segment_type = torch.zeros(n_total, 3, dtype=torch.float32)
        interface_mask = torch.zeros(n_total, dtype=torch.float32)
        epitope_mask = torch.zeros(n_total, dtype=torch.float32)

        h_start, h_end = 1, 1 + n_heavy
        l_start, l_end = h_end + 1, h_end + 1 + n_light
        a_start, a_end = l_end + 1, l_end + 1 + n_antigen

        X[h_start:h_end] = X_residues[:n_heavy]
        X[l_start:l_end] = X_residues[n_heavy:n_heavy+n_light]
        X[a_start:a_end] = X_residues[n_heavy+n_light:]

        X[0] = X[h_start:h_end].mean(dim=0) if n_heavy > 0 else torch.zeros(4, 3)
        X[h_end] = X[l_start:l_end].mean(dim=0) if n_light > 0 else torch.zeros(4, 3)
        X[l_end] = X[a_start:a_end].mean(dim=0) if n_antigen > 0 else torch.zeros(4, 3)

        node_features[h_start:h_end] = feats['node_features'][:n_heavy]
        node_features[l_start:l_end] = feats['node_features'][n_heavy:n_heavy+n_light]
        node_features[a_start:a_end] = feats['node_features'][n_heavy+n_light:]

        im = feats['interface_mask']
        interface_mask[h_start:h_end] = im[:n_heavy]
        interface_mask[l_start:l_end] = im[n_heavy:n_heavy+n_light]
        interface_mask[a_start:a_end] = im[n_heavy+n_light:]

        em = feats['epitope_mask']
        epitope_mask[h_start:h_end] = em[:n_heavy]
        epitope_mask[l_start:l_end] = em[n_heavy:n_heavy+n_light]
        epitope_mask[a_start:a_end] = em[n_heavy+n_light:]

        segment_type[0:h_end, 0] = 1.0
        segment_type[h_end:l_end, 1] = 1.0
        segment_type[l_end:a_end, 2] = 1.0

        seq = feats['sequence']
        S[0] = VOCAB.symbol_to_idx(VOCAB.BOH)
        S[h_end] = VOCAB.symbol_to_idx(VOCAB.BOL)
        S[l_end] = VOCAB.symbol_to_idx(VOCAB.BOA)

        for i, aa in enumerate(seq[:n_heavy]):
            S[h_start + i] = VOCAB.symbol_to_idx(aa)
        for i, aa in enumerate(seq[n_heavy:n_heavy+n_light]):
            S[l_start + i] = VOCAB.symbol_to_idx(aa)
        for i, aa in enumerate(seq[n_heavy+n_light:]):
            S[a_start + i] = VOCAB.symbol_to_idx(aa)

        cdr_mask = feats['cdr_mask']
        L = ['0'] * n_total

        for i in range(n_heavy):
            for cdr_idx in range(3):
                if cdr_mask[i, cdr_idx] > 0.5:
                    L[h_start + i] = str(cdr_idx + 1)
                    break

        L = ''.join(L)

        global_mask = torch.zeros(n_total, dtype=torch.bool)
        global_mask[0] = True
        global_mask[h_end] = True
        global_mask[l_end] = True

        global_types = torch.tensor([0, 1, 2], dtype=torch.long)

        cdr_idx = int(self.cdr_type) - 1
        heavy_seq = seq[:n_heavy]
        cdr_positions = [i for i in range(n_heavy) if cdr_mask[i, cdr_idx] > 0.5]
        true_cdr_seq = ''.join([heavy_seq[i] for i in cdr_positions]) if cdr_positions else ''

        cdr_start = h_start + cdr_positions[0] if cdr_positions else h_start
        cdr_end = h_start + cdr_positions[-1] + 1 if cdr_positions else h_start

        cdr_mask_1d = torch.zeros(n_total, dtype=torch.bool)
        for i in cdr_positions:
            cdr_mask_1d[h_start + i] = True

        aux_targets = extract_aux_targets(node_features, cdr_mask_1d)
        mask_cdr_features(node_features, cdr_mask_1d)

        if self.germline_prior is not None:
            heavy_numbering = complex_data.get('numbering', {}).get(
                self.numbering_scheme, {}).get('heavy', [])
            germline_prior = self.germline_prior.build_tensor(
                complex_id, n_total, h_start, cdr_positions, heavy_numbering)
        else:
            germline_prior = torch.zeros(n_total, len(VOCAB), dtype=torch.float32)

        return {
            'X': X,
            'S': S,
            'L': L,
            'germline_prior': germline_prior,
            'node_features': node_features,
            'segment_type': segment_type,
            'interface_mask': interface_mask,
            'epitope_mask': epitope_mask,
            'global_mask': global_mask,
            'global_types': global_types,
            'complex_id': complex_id,
            'true_cdr_seq': true_cdr_seq,
            'cdr_range': (cdr_start, cdr_end),
            'cdr_mask_1d': cdr_mask_1d,
            'aux_targets': aux_targets,
            'ca_centroid': ca_centroid,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with variable-length sequences."""
        Xs, Ss, Ls = [], [], []
        node_feats_list, seg_type_list = [], []
        interface_mask_list, epitope_mask_list = [], []
        global_masks, global_types_list = [], []
        cdr_mask_1d_list, aux_targets_list = [], []
        germline_prior_list = []
        complex_ids, true_cdr_seqs, cdr_ranges = [], [], []
        ca_centroids = []
        offsets = [0]

        for data in batch:
            Xs.append(data['X'])
            Ss.append(data['S'])
            Ls.append(data['L'])
            node_feats_list.append(data['node_features'])
            seg_type_list.append(data['segment_type'])
            interface_mask_list.append(data['interface_mask'])
            epitope_mask_list.append(data['epitope_mask'])
            global_masks.append(data['global_mask'])
            global_types_list.append(data['global_types'])
            cdr_mask_1d_list.append(data['cdr_mask_1d'])
            aux_targets_list.append(data['aux_targets'])
            germline_prior_list.append(data['germline_prior'])
            complex_ids.append(data['complex_id'])
            true_cdr_seqs.append(data['true_cdr_seq'])
            cdr_ranges.append(data['cdr_range'])
            ca_centroids.append(data['ca_centroid'])
            offsets.append(offsets[-1] + len(data['S']))

        return {
            'X': torch.cat(Xs, dim=0),
            'S': torch.cat(Ss, dim=0),
            'L': Ls,
            'node_features': torch.cat(node_feats_list, dim=0),
            'segment_type': torch.cat(seg_type_list, dim=0),
            'interface_mask': torch.cat(interface_mask_list, dim=0),
            'epitope_mask': torch.cat(epitope_mask_list, dim=0),
            'global_mask': torch.cat(global_masks, dim=0),
            'global_types': torch.cat(global_types_list, dim=0),
            'cdr_mask_1d': torch.cat(cdr_mask_1d_list, dim=0),
            'aux_targets': torch.cat(aux_targets_list, dim=0),
            'germline_prior': torch.cat(germline_prior_list, dim=0),
            'offsets': torch.tensor(offsets, dtype=torch.long),
            'complex_ids': complex_ids,
            'true_cdr_seqs': true_cdr_seqs,
            'cdr_ranges': cdr_ranges,
            'ca_centroids': ca_centroids,
        }
