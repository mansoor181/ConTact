"""CompFirst: Contact-First CDR Design with Distance-Biased Attention.

Architecture:
    ProteinFeatureEncoder(node_features) -> VirtualNodeEGNN(H_0, X, edges, masks)
    -> distance_biased_cross_attn(CDR_query, AG_key/value, dists) -> attn_out
    -> FingerprintPredictor(cdr_h, attn_out) -> pred_fp
    -> ContrastiveFingerprintLoss(pred_fp, ag_fp, contact_labels) -> fp_loss
    -> ContactPredictor(cdr_h, ag_h, dist_rbf, pred_fp) -> contact_logits
    -> sigmoid(contact_logits) -> soft_contacts
    -> LocalCompInjector(cdr_h, cdr_pos, comp_features, soft_contacts)
    -> seq_head(enriched_cdr_h, masked_attn_out) + germline_lambda * log_prior -> logits
    -> contact_weighted_ce(logits, target, alpha)

Losses (8-tuple): seq, coord, pairing, dock, fp_loss, contact_loss, aux_loss
"""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

import numpy as np

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..', '..'))
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, _GENBIO_ROOT)

from data.pdb_utils import VOCAB
from model.modules import VirtualNodeEGNN
from shared.sparse_features import ProteinFeatureEncoderWithSparse, CDRFeaturePredictor


# ── AA Chemistry features ─────────────────────────────────────────────────────

AA_CHEMISTRY = {
    'A': [1.8, 0.0, 0.0, 0.0, 0.0, 0.1], 'R': [-4.5, 1.0, 3.0, 1.0, 0.0, 0.7],
    'N': [-3.5, 0.0, 1.0, 2.0, 0.0, 0.3], 'D': [-3.5, -1.0, 0.0, 2.0, 0.0, 0.3],
    'C': [2.5, 0.0, 0.0, 0.0, 0.0, 0.2], 'Q': [-3.5, 0.0, 1.0, 2.0, 0.0, 0.4],
    'E': [-3.5, -1.0, 0.0, 2.0, 0.0, 0.4], 'G': [-0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    'H': [-3.2, 0.5, 1.0, 1.0, 1.0, 0.5], 'I': [4.5, 0.0, 0.0, 0.0, 0.0, 0.4],
    'L': [3.8, 0.0, 0.0, 0.0, 0.0, 0.4], 'K': [-3.9, 1.0, 2.0, 0.0, 0.0, 0.5],
    'M': [1.9, 0.0, 0.0, 1.0, 0.0, 0.4], 'F': [2.8, 0.0, 0.0, 0.0, 1.0, 0.6],
    'P': [-1.6, 0.0, 0.0, 0.0, 0.0, 0.2], 'S': [-0.8, 0.0, 1.0, 1.0, 0.0, 0.1],
    'T': [-0.7, 0.0, 1.0, 1.0, 0.0, 0.2], 'W': [-0.9, 0.0, 1.0, 0.0, 1.0, 0.8],
    'Y': [-1.3, 0.0, 1.0, 1.0, 1.0, 0.7], 'V': [4.2, 0.0, 0.0, 0.0, 0.0, 0.3],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

def _get_aa_chemistry_tensor(device):
    feats = []
    for i in range(len(VOCAB)):
        sym = VOCAB.idx_to_symbol(i)
        feats.append(AA_CHEMISTRY.get(sym, AA_CHEMISTRY['X']))
    return torch.tensor(feats, dtype=torch.float32, device=device)


class PosEmbedding(nn.Module):
    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


# ── Distance-Biased Cross-Attention ───────────────────────────────────────────

class DistanceBiasedCrossAttention(nn.Module):
    """Cross-attention with distance bias: nearby residues get higher attention."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.0, dist_scale=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dist_scale = dist_scale

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, query_pos, key_pos, key_padding_mask=None):
        """
        Args:
            query: (L_q, H) query features (CDR)
            key: (L_k, H) key features (antigen)
            value: (L_k, H) value features (antigen)
            query_pos: (L_q, 3) query positions
            key_pos: (L_k, 3) key positions
            key_padding_mask: (L_k,) bool mask, True = ignore
        Returns:
            out: (L_q, H) attended features
        """
        L_q, H = query.shape
        L_k = key.shape[0]

        if L_k == 0:
            return torch.zeros_like(query)

        # Project to Q, K, V
        Q = self.q_proj(query).view(L_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(L_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(L_k, self.num_heads, self.head_dim)

        # Compute attention scores: (L_q, num_heads, L_k)
        attn_scores = torch.einsum('qhd,khd->qhk', Q, K) / (self.head_dim ** 0.5)

        # FIX: Add distance bias - closer residues get higher scores
        # Compute pairwise distances: (L_q, L_k)
        dists = torch.cdist(query_pos, key_pos)
        # Convert to bias: negative distance / scale -> closer = higher score
        # Using Gaussian-like decay: exp(-d^2 / (2 * scale^2))
        dist_bias = -0.5 * (dists / self.dist_scale) ** 2
        # Add to attention scores (broadcast over heads)
        attn_scores = attn_scores + dist_bias.unsqueeze(1)

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attend to values: (L_q, num_heads, head_dim)
        out = torch.einsum('qhk,khd->qhd', attn_weights, V)
        out = out.reshape(L_q, H)
        out = self.out_proj(out)

        return out


# ── Stage 1: Fingerprint Predictor (Increased Capacity) ──────────────────────

class FingerprintPredictor(nn.Module):
    """Predict complementary surface fingerprint per CDR residue.

    v2: Increased capacity with residual connection and LayerNorm.
    """

    def __init__(self, hidden_dim=256, fingerprint_dim=32):
        super().__init__()
        self.fingerprint_dim = fingerprint_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, fingerprint_dim),
        )

    def forward(self, cdr_h, attn_out):
        return self.mlp(torch.cat([cdr_h, attn_out], dim=-1))


# ── Contrastive Fingerprint Loss ──────────────────────────────────────────────

class ContrastiveFingerprintLoss(nn.Module):
    """Learn fingerprints via contrastive learning on binding pairs.

    Idea: CDR positions that contact the same antigen residue should have
    similar fingerprints to that antigen residue's embedding. Non-contacting
    positions should have dissimilar fingerprints.

    This replaces the heuristic chemistry inversion with learned complementarity.
    """

    def __init__(self, hidden_dim, fingerprint_dim=32, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        # Project antigen embeddings to fingerprint space
        self.ag_fingerprint_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, fingerprint_dim),
        )

    def forward(self, cdr_fp, ag_h, ag_pos, cdr_pos, contact_labels, contact_cutoff=8.0):
        """
        Args:
            cdr_fp: (L_cdr, fp_dim) predicted CDR fingerprints
            ag_h: (L_ag, H) antigen node features
            ag_pos: (L_ag, 3) antigen CA positions
            cdr_pos: (L_cdr, 3) CDR CA positions
            contact_labels: (L_cdr,) soft contact predictions (0-1), thresholded at 0.5
            contact_cutoff: distance threshold for contact
        Returns:
            loss: scalar contrastive loss
        """
        if ag_h.shape[0] == 0 or contact_labels.sum() == 0:
            return torch.tensor(0.0, device=cdr_fp.device)

        # Project antigen to fingerprint space
        ag_fp = self.ag_fingerprint_proj(ag_h)  # (L_ag, fp_dim)

        # Normalize fingerprints for cosine similarity
        cdr_fp_norm = F.normalize(cdr_fp, dim=-1)
        ag_fp_norm = F.normalize(ag_fp, dim=-1)

        # Compute all pairwise similarities: (L_cdr, L_ag)
        sim_matrix = torch.mm(cdr_fp_norm, ag_fp_norm.t()) / self.temperature

        # Compute distance matrix for determining positive/negative pairs
        dists = torch.cdist(cdr_pos, ag_pos)

        # Positive pairs: CDR-AG pairs within contact distance
        # For each CDR position, find its nearest antigen (if within cutoff)
        min_dists, nearest_ag = dists.min(dim=1)
        is_contact = contact_labels > 0.5

        loss = torch.tensor(0.0, device=cdr_fp.device)
        n_positives = 0

        for i in range(cdr_fp.shape[0]):
            if is_contact[i] and min_dists[i] < contact_cutoff:
                # This CDR position contacts the antigen
                pos_idx = nearest_ag[i]
                pos_sim = sim_matrix[i, pos_idx]

                # Negative samples: antigen residues far from this CDR position
                neg_mask = dists[i] > contact_cutoff * 1.5
                if neg_mask.sum() > 0:
                    neg_sims = sim_matrix[i, neg_mask]
                    # InfoNCE-style loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                    labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                    loss = loss + F.cross_entropy(logits.unsqueeze(0), labels)
                    n_positives += 1

        if n_positives > 0:
            loss = loss / n_positives

        # Also add margin loss for non-contacts: they should have low similarity
        non_contact_mask = contact_labels < 0.5
        if non_contact_mask.sum() > 0:
            # Max similarity for non-contacts should be below margin
            non_contact_sims = sim_matrix[non_contact_mask].max(dim=1).values
            margin_loss = F.relu(non_contact_sims - self.margin).mean()
            loss = loss + 0.5 * margin_loss

        return loss


# ── Stage 2: Contact Predictor ────────────────────────────────────────────────

class ContactPredictor(nn.Module):
    """Predict which CDR positions contact the epitope, conditioned on fingerprint."""

    def __init__(self, hidden_dim, fingerprint_dim=32, k_neighbors=4):
        super().__init__()
        self.k = k_neighbors
        # cdr_h + ag_agg + dist_rbf + fingerprint
        self.contact_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16 + fingerprint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _rbf(self, d, n_rbf=16, d_max=15.0):
        centers = torch.linspace(0, d_max, n_rbf, device=d.device)
        sigma = d_max / n_rbf
        return torch.exp(-((d.unsqueeze(-1) - centers) / sigma) ** 2)

    def forward(self, cdr_h, cdr_pos, ag_h, ag_pos, fingerprint):
        if ag_h.shape[0] == 0:
            return torch.zeros(cdr_h.shape[0], device=cdr_h.device)

        k = min(self.k, ag_h.shape[0])
        dists = torch.cdist(cdr_pos, ag_pos)
        min_dists, topk_idx = dists.topk(k, dim=-1, largest=False)

        neighbor_h = ag_h[topk_idx]
        weights = F.softmax(-min_dists, dim=-1)
        ag_agg = (weights.unsqueeze(-1) * neighbor_h).sum(dim=1)

        dist_rbf = self._rbf(min_dists[:, 0])
        contact_input = torch.cat([cdr_h, ag_agg, dist_rbf, fingerprint], dim=-1)
        return self.contact_mlp(contact_input).squeeze(-1)


# ── Stage 3: Local Complementarity Injector ───────────────────────────────────

class LocalCompInjector(nn.Module):
    """Per-position K-NN aggregation of antigen features weighted by contact confidence."""

    def __init__(self, hidden_dim, k=4):
        super().__init__()
        self.k = k
        self.attn = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, cdr_h, cdr_pos, ag_h, ag_pos, contact_confidence):
        if ag_h.shape[0] == 0:
            return cdr_h

        k = min(self.k, ag_h.shape[0])
        dists = torch.cdist(cdr_pos, ag_pos)
        _, topk_idx = dists.topk(k, dim=-1, largest=False)
        neighbor_h = ag_h[topk_idx]
        attn_weights = F.softmax(self.attn(neighbor_h), dim=1)
        local_comp = (attn_weights * neighbor_h).sum(dim=1)
        gate = contact_confidence.unsqueeze(-1)
        return cdr_h + gate * self.proj(local_comp)


# ── Contact-Weighted CE ───────────────────────────────────────────────────────

def contact_weighted_ce(logits, targets, contact_confidence, alpha=3.0, label_smoothing=0.1):
    """Contact-weighted cross-entropy with label smoothing."""
    weights = 1.0 + alpha * contact_confidence
    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    return (weights * ce).sum() / weights.sum()


# ── ProteinFeature (unchanged from v1) ────────────────────────────────────────

class ProteinFeature(nn.Module):
    def __init__(self, interface_only):
        super().__init__()
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 3, 1, 2
        self.node_pos_embedding = PosEmbedding(16)
        self.edge_pos_embedding = PosEmbedding(16)
        self.interface_only = interface_only

    def _construct_segment_ids(self, S):
        glbl_node_mask = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        segment_idx = torch.zeros_like(S)
        segment_idx[glbl_node_mask] = 1.0
        segment_mask = torch.cumsum(segment_idx, dim=0)
        return segment_ids, segment_mask, torch.nonzero(segment_idx)[:, 0]

    def _radial_edges(self, X, src_dst, cutoff):
        dist = X[:, 1][src_dst]
        dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1)
        src_dst = src_dst[dist <= cutoff]
        return src_dst.transpose(0, 1)

    def _knn_edges(self, X, offsets, segment_ids, is_global, top_k=5, eps=1e-6):
        for batch in range(len(offsets)):
            if batch != len(offsets) - 1:
                X_batch = X[offsets[batch]:offsets[batch+1], 1, :]
            else:
                X_batch = X[offsets[batch]:, 1, :]
            dX = torch.unsqueeze(X_batch, 0) - torch.unsqueeze(X_batch, 1)
            D = torch.sqrt(torch.sum(dX**2, 2) + eps)
            _, E_idx = torch.topk(D, top_k, dim=-1, largest=False)
            if batch == 0:
                row = torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1)
                col = E_idx.view(-1)
            else:
                row = torch.cat([row, torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1) + offsets[batch]], dim=0)
                col = torch.cat([col, E_idx.view(-1) + offsets[batch]], dim=0)
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_edges_knn = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_edges_knn = torch.stack([row[select_edges], col[select_edges]])
        return ctx_edges_knn, inter_edges_knn

    def get_node_pos(self, X, segment_mask, segment_idx):
        pos = torch.arange(X.shape[0], device=X.device) - segment_idx[segment_mask-1]
        pos_node_feats = self.node_pos_embedding(pos.view(1, X.shape[0], 1))[0, :, 0, :]
        return pos_node_feats

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def get_node_dist(self, X, eps=1e-6):
        d_NC = torch.sqrt(torch.sum((X[:, 0, :] - X[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X[:, 2, :] - X[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X[:, 3, :] - X[:, 1, :])**2, dim=1) + eps)
        return torch.cat((self._rbf(d_NC), self._rbf(d_CC), self._rbf(d_OC)), 1)

    def get_node_angle(self, X, segment_idx, segment_ids, eps=1e-6):
        X = X[:, :3,:].reshape(1, 3*X.shape[0], 3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2, u_1, u_0 = U[:,:-2,:], U[:,1:-1,:], U[:,2:,:]
        n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        cosD = (u_2*u_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        angle_node_feats = torch.cat((Dihedral_Angle_features, Angle_features), 2)[0]
        for i in segment_idx:
            if i == 0:
                angle_node_feats[i:i+2] = 0
            else:
                angle_node_feats[i-1:i+2] = 0
        if self.interface_only == 0:
            angle_node_feats[segment_ids == self.ag_seg_id] = 0
        return angle_node_feats

    def get_node_direct(self, Xs, segment_idx, segment_ids):
        X = Xs[:, 1,:].reshape(1, Xs.shape[0], 3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.linalg.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O = O.view(list(O.shape[:2]) + [3,3])
        for i in segment_idx:
            if i == 0:
                O[:, i:i+2, :, :] = 0
            else:
                O[:, i-2:i+2, :, :] = 0
        if self.interface_only == 0:
            O[:, segment_ids == self.ag_seg_id, :, :] = 0
        d_NC = (Xs[:, 0, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O, d_NC).squeeze(-1), dim=-1)
        d_CC = (Xs[:, 2, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O, d_CC).squeeze(-1), dim=-1)
        d_OC = (Xs[:, 3, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O, d_OC).squeeze(-1), dim=-1)
        return torch.cat((d_NC, d_CC, d_OC), 2)[0], O

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, - Rxx + Ryy - Rzz, - Rxx - Ryy + Rzz], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2), _R(0,2) - _R(2,0), _R(1,0) - _R(0,1)], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        return F.normalize(Q, dim=-1)

    def get_edge_pos(self, edge_index):
        pos = (edge_index[0:1, :] - edge_index[1:2, :]).float().unsqueeze(-1)
        return self.edge_pos_embedding(pos)[0, :, 0, :]

    def get_edge_dist(self, X, edge_index, eps=1e-6):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        d_NC = torch.sqrt(torch.sum((X_row[:, 0, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CAC = torch.sqrt(torch.sum((X_row[:, 1, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X_row[:, 2, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X_row[:, 3, :] - X_col[:, 1, :])**2, dim=1) + eps)
        return torch.cat((self._rbf(d_NC), self._rbf(d_CAC), self._rbf(d_CC), self._rbf(d_OC)), 1)

    def get_edge_angle(self, O, edge_index):
        O_row, O_col = O[:, edge_index[0, :], :, :].unsqueeze(2), O[:, edge_index[1, :], :, :].unsqueeze(2)
        R = torch.matmul(O_row.transpose(-1,-2), O_col)
        return self._quaternions(R)[0, :, 0, :]

    def get_edge_direct(self, X, O, edge_index):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        _, O_col = O[:, edge_index[0, :], :, :], O[:, edge_index[1, :], :, :]
        d_NC = (X_row[:, 0, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O_col, d_NC).squeeze(-1), dim=-1)
        d_CAC = (X_row[:, 1, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CAC = F.normalize(torch.matmul(O_col, d_CAC).squeeze(-1), dim=-1)
        d_CC = (X_row[:, 2, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O_col, d_CC).squeeze(-1), dim=-1)
        d_OC = (X_row[:, 3, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O_col, d_OC).squeeze(-1), dim=-1)
        return torch.cat((d_NC, d_CAC, d_CC, d_OC), 2)[0]

    def edge_masking(self, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, edge_type):
        if edge_type == 1 or edge_type == 2 or edge_type == 6 or edge_type == 7:
            pos_edge_feats *= 0
        return pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id):
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]
        segment_ids, segment_mask, segment_idx = self._construct_segment_ids(S)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0
        row, col = torch.nonzero(same_bid).T
        col = col + offsets[batch_id[row]]
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        ctx_edges_rball = self._radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)
        ctx_edges_knn, inter_edges_knn = self._knn_edges(X, offsets, segment_ids, is_global, top_k=8)
        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges)
        ctx_edges_seq_d1 = torch.stack([row[select_edges_seq], col[select_edges_seq]])
        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges)
        ctx_edges_seq_d2 = torch.stack([row[select_edges_seq], col[select_edges_seq]])
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges_rball = self._radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_global, col_global)
        global_global = torch.stack([row[select_edges], col[select_edges]])
        pos_node_feats = self.get_node_pos(X, segment_mask, segment_idx)
        dis_node_feats = self.get_node_dist(X)
        angle_node_feats = self.get_node_angle(X, segment_idx.tolist(), segment_ids)
        direct_node_feats, O = self.get_node_direct(X, segment_idx.tolist(), segment_ids)
        node_feats = torch.cat((pos_node_feats, dis_node_feats, angle_node_feats, direct_node_feats), 1)
        edges_list = [ctx_edges_rball, global_normal, global_global, ctx_edges_seq_d1, ctx_edges_knn, ctx_edges_seq_d2, inter_edges_rball, inter_edges_knn]
        edge_class_type = torch.eye(len(edges_list), dtype=torch.float, device=X.device)
        edge_feats_list = []
        for i in range(len(edges_list)):
            type_edge_feats = edge_class_type[torch.ones(edges_list[i].shape[1]).long() * i]
            pos_edge_feats = self.get_edge_pos(edges_list[i])
            dis_edge_feats = self.get_edge_dist(X, edges_list[i])
            angle_edge_feats = self.get_edge_angle(O, edges_list[i])
            direct_edge_feats = self.get_edge_direct(X, O, edges_list[i])
            pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats = self.edge_masking(pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, i)
            edge_feats = torch.cat((type_edge_feats, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats), 1)
            edge_feats_list.append(edge_feats)
        return edges_list, edge_feats_list, node_feats, segment_idx, segment_ids

    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id = torch.cumsum(batch_id, dim=0)
        return self.construct_edges(X, S, batch_id)


# ── CompFirst v2 DockDesigner ─────────────────────────────────────────────────

class DockDesigner(nn.Module):
    """CompFirst v2: Contact-First CDR Design with All Fixes.

    Improvements over v1:
      1. Learned fingerprint targets (contrastive loss, not heuristic)
      2. Soft sigmoid contacts (no Gumbel-Softmax)
      3. Distance-biased cross-attention
      4. Loss normalization with running statistics
      5. Larger fingerprint_dim (32, not 6)
    """

    def __init__(self, embed_size, hidden_size, n_channel, n_layers, dropout, cdr_type, args):
        super().__init__()
        self.cdr_type = cdr_type
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = getattr(args, 'gamma', 0.5)
        self.dock_cutoff = getattr(args, 'dock_cutoff', 8.0)
        self.num_attn_heads = getattr(args, 'num_attn_heads', 4)
        self.n_virtual_nodes = getattr(args, 'n_virtual_nodes', 3)
        self.hidden_size = hidden_size

        # CompFirst-specific (v2: larger fingerprint_dim)
        self.fingerprint_dim = getattr(args, 'fingerprint_dim', 32)
        self.fp_k = getattr(args, 'fp_k', 8)
        self.contact_weight_alpha = getattr(args, 'contact_weight_alpha', 3.0)
        self.zeta = getattr(args, 'zeta', 0.1)
        self.contact_loss_weight = getattr(args, 'contact_loss_weight', 1.0)

        node_feats_mode = args.node_feats_mode
        edge_feats_mode = args.edge_feats_mode
        self.interface_only = args.interface_only

        node_feats_dim = int(node_feats_mode[0]) * 16 + int(node_feats_mode[1]) * 48 + int(node_feats_mode[2]) * 12 + int(node_feats_mode[3]) * 9
        edge_feats_dim = int(edge_feats_mode[0]) * 16 + int(edge_feats_mode[1]) * 64 + int(edge_feats_mode[2]) * 4 + int(edge_feats_mode[3]) * 12 + 8

        self.num_aa_type = len(VOCAB)
        self.mask_token_id = VOCAB.get_unk_idx()
        self.projection_head_cdr = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))
        self.projection_head_ant = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))

        # Rich feature encoder (105D pre-computed features + 3D segment type)
        self.protein_feature_encoder = ProteinFeatureEncoderWithSparse(
            hidden_dim=embed_size, input_dim=108, dropout=dropout,
            use_dual_path=True, fusion_type='concat')

        # GNN
        self.gnn = VirtualNodeEGNN(
            embed_size, hidden_size, self.num_aa_type, n_channel,
            n_layers=n_layers, dropout=dropout,
            node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim,
            n_virtual_nodes=self.n_virtual_nodes)

        # FIX #3: Distance-biased cross-attention (replaces standard MHA)
        dist_scale = getattr(args, 'dist_attn_scale', 4.0)
        self.ag_cross_attn = DistanceBiasedCrossAttention(
            hidden_size, num_heads=self.num_attn_heads, dropout=dropout,
            dist_scale=dist_scale)

        # Stage 1: Fingerprint prediction (v2: larger capacity)
        self.fingerprint_predictor = FingerprintPredictor(hidden_size, self.fingerprint_dim)

        # FIX #1: Contrastive fingerprint loss (replaces heuristic inversion)
        self.contrastive_fp_loss = ContrastiveFingerprintLoss(
            hidden_size, self.fingerprint_dim, temperature=0.1, margin=0.3)

        # Stage 2: Contact prediction
        self.contact_predictor = ContactPredictor(
            hidden_size, self.fingerprint_dim, k_neighbors=4)

        # FIX #2: Remove Gumbel-Softmax - use soft sigmoid directly
        # (No ContactLatentSampler needed)

        # Stage 3: Local comp injection
        self.local_comp_injector = LocalCompInjector(hidden_size, k=self.fp_k)

        # Seq head
        self.seq_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.num_aa_type),
        )

        # L1: germline (J,pos) log-prior fusion into the seq head (config-gated; preserves v1 when off).
        # logits = seq_head(...) + germline_lambda * log p_germline(J, pos). lambda is learnable.
        self.use_germline_prior = getattr(args, 'use_germline_prior', False)
        if self.use_germline_prior:
            lam_init = float(getattr(args, 'germline_prior_lambda_init', 1.0))
            self.germline_lambda = nn.Parameter(torch.tensor(lam_init))

        self.protein_feature = ProteinFeature(args.interface_only)

        # Auxiliary CDR feature reconstruction
        self.eta = getattr(args, 'eta', 0.3)
        self.cdr_feature_predictor = CDRFeaturePredictor(hidden_size, dropout=dropout)

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    seq_loss_fn = seq_loss

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')

    def init_mask(self, X, S, cdr_range):
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask

    def _contrastive_pairing_loss(self, aa_embd, S, cdr_range, segment_idx):
        aa_embd_cdr = self.projection_head_cdr(aa_embd)
        aa_embd_ant = self.projection_head_ant(aa_embd)
        cdr_logits = []
        for start, end in cdr_range:
            cdr_logits.append(torch.mean(aa_embd_cdr[start:end+1], dim=0, keepdim=True))
        cdr_logits = torch.cat(cdr_logits, dim=0)
        ant_logits = []
        segment_list = segment_idx.tolist()
        for i, index in enumerate(segment_list):
            if S[index] == VOCAB.symbol_to_idx(VOCAB.BOA):
                if index == segment_list[-1]:
                    ant_logits.append(torch.mean(aa_embd_ant[index:], dim=0, keepdim=True))
                else:
                    ant_logits.append(torch.mean(aa_embd_ant[index:segment_list[i+1]], dim=0, keepdim=True))
        if len(ant_logits) == 0 or len(ant_logits) != len(cdr_logits):
            return torch.tensor(0.0, device=aa_embd.device)
        ant_logits = torch.cat(ant_logits, dim=0)
        norm1, norm2 = cdr_logits.norm(dim=1), ant_logits.norm(dim=1)
        mat_norm = torch.einsum('i,j->ij', norm1, norm2)
        mat_sim = torch.exp(torch.einsum('ik,jk,ij->ij', cdr_logits, ant_logits, 1/mat_norm.clamp(min=1e-8)) / 1.0)
        b = cdr_logits.size(0)
        diag = mat_sim[range(b), range(b)]
        p_loss = -torch.log(diag / (mat_sim.sum(dim=1) - diag).clamp(min=1e-8)).mean()
        return p_loss

    def _dock_loss(self, Z, true_X, cdr_range, S, segment_ids):
        device = Z.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]
        if ag_ca.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        losses = []
        for start, end in cdr_range:
            true_cdr_ca = true_X[start:end + 1, 1, :]
            pred_cdr_ca = Z[start:end + 1, 1, :]
            true_dists = torch.cdist(ag_ca, true_cdr_ca)
            epi_mask = true_dists.min(dim=1).values < self.dock_cutoff
            if not epi_mask.any():
                continue
            epi_ca = ag_ca[epi_mask]
            pred_dists = torch.cdist(pred_cdr_ca, epi_ca)
            min_dists = pred_dists.min(dim=1).values
            losses.append(min_dists.mean())
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _build_masks(self, S, segment_ids, cdr_range, X=None):
        device = S.device
        n_nodes = S.shape[0]
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        cdr_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        for start, end in cdr_range:
            cdr_mask[start:end + 1] = True
        if X is not None and ag_mask.any() and cdr_mask.any():
            ag_ca = X[ag_mask, 1, :]
            cdr_ca = X[cdr_mask, 1, :]
            dists = torch.cdist(ag_ca, cdr_ca).min(dim=1).values
            near_mask = dists < self.dock_cutoff
            epitope_mask = torch.zeros_like(ag_mask)
            epitope_mask[ag_mask] = near_mask
        else:
            epitope_mask = ag_mask
        return epitope_mask, cdr_mask

    def _get_contact_labels(self, true_X, cdr_range, S, segment_ids):
        """Ground truth contact labels for CDR positions."""
        device = true_X.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]
        all_contacts = []
        for start, end in cdr_range:
            cdr_ca = true_X[start:end + 1, 1, :]
            if ag_ca.shape[0] == 0:
                all_contacts.append(torch.zeros(end - start + 1, device=device))
            else:
                dists = torch.cdist(cdr_ca, ag_ca)
                contacts = (dists.min(dim=-1).values < self.dock_cutoff).float()
                all_contacts.append(contacts)
        return torch.cat(all_contacts, dim=0)

    def _three_stage_logits_v2(self, aa_embd, S, segment_ids, offsets, cdr_range, Z, true_X,
                                epitope_mask=None, germline_prior=None):
        """Three-stage cascade v2: with distance-biased attention and soft contacts."""
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)
        ag_seg_id = self.protein_feature.ag_seg_id

        all_logits = []
        all_fps = []
        all_contact_logits = []
        all_ag_h = []
        all_ag_pos = []
        all_cdr_pos = []

        for i, (start, end) in enumerate(cdr_range):
            cdr_h = aa_embd[start:end + 1]
            cdr_pos = Z[start:end + 1, 1, :]

            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if i + 1 < len(offsets) else len(aa_embd)
            c_seg = segment_ids[c_start:c_end]
            c_global = is_global[c_start:c_end]

            if epitope_mask is not None:
                ag_idx = epitope_mask[c_start:c_end] & ~c_global
            else:
                ag_idx = (c_seg == ag_seg_id) & ~c_global
            ag_h = aa_embd[c_start:c_end][ag_idx]
            ag_pos = true_X[c_start:c_end][ag_idx, 1, :]

            # Store for contrastive loss
            all_ag_h.append(ag_h)
            all_ag_pos.append(ag_pos)
            all_cdr_pos.append(cdr_pos)

            # FIX #3: Distance-biased cross-attention (actually uses distance!)
            if ag_h.shape[0] > 0:
                attn_out = self.ag_cross_attn(
                    cdr_h, ag_h, ag_h, cdr_pos, ag_pos)
            else:
                attn_out = torch.zeros_like(cdr_h)

            # Stage 1: Fingerprint prediction
            pred_fp = self.fingerprint_predictor(cdr_h, attn_out)
            all_fps.append(pred_fp)

            # Stage 2: Contact prediction
            contact_logits = self.contact_predictor(
                cdr_h, cdr_pos, ag_h, ag_pos, pred_fp)
            all_contact_logits.append(contact_logits)

            # FIX #2: Use soft sigmoid directly (no Gumbel-Softmax)
            contact_soft = torch.sigmoid(contact_logits)

            # Stage 3: Local comp injection (gated by soft contact)
            enriched_h = self.local_comp_injector(
                cdr_h, cdr_pos, ag_h, ag_pos, contact_soft)

            # Contact-gated attention (soft, not hard)
            min_weight = 0.15
            soft_mask = min_weight + (1.0 - min_weight) * contact_soft
            masked_attn = attn_out * soft_mask.unsqueeze(-1)

            # Seq head
            combined = torch.cat([enriched_h, masked_attn], dim=-1)
            logits = self.seq_head(combined)
            # L1: additive germline log-prior (J,pos) at the binding/framework positions
            if self.use_germline_prior and germline_prior is not None:
                logits = logits + self.germline_lambda * germline_prior[start:end + 1]
            all_logits.append(logits)

        return (torch.cat(all_logits, dim=0),
                torch.cat(all_fps, dim=0),
                torch.cat(all_contact_logits, dim=0),
                all_ag_h, all_ag_pos, all_cdr_pos)

    def forward(self, X, S, L, offsets, node_features=None, segment_type=None,
                interface_mask=None, global_mask=None, global_types=None,
                cdr_mask_1d=None, aux_targets=None, germline_prior=None):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        # Encode input (105D pre-computed features)
        residue_mask = ~global_mask
        residue_feats = node_features[residue_mask]
        residue_seg = segment_type[residue_mask]
        residue_interface = interface_mask[residue_mask] if interface_mask is not None else None
        H_0 = self.protein_feature_encoder(
            residue_feats, residue_seg, residue_interface, global_mask, global_types)

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)

        epitope_mask, cdr_mask = self._build_masks(S, segment_ids, cdr_range, X=true_X)
        H, Z, aa_embd = self.gnn(
            H_0, X, edges_list, edge_feats_list, node_feats, segment_ids,
            self.interface_only, epitope_mask, cdr_mask)

        # Three-stage cascade v2
        cdr_logits, pred_fps, contact_logits, all_ag_h, all_ag_pos, all_cdr_pos = \
            self._three_stage_logits_v2(
                aa_embd, S, segment_ids, offsets, cdr_range, Z, true_X,
                epitope_mask=epitope_mask, germline_prior=germline_prior)

        # Contact labels (ground truth) - only for contact predictor supervision
        contact_labels = self._get_contact_labels(true_X, cdr_range, S, segment_ids)

        # FIX #2: Use soft contacts for weighting (sigmoid, not sampled)
        # FIX #5: Use PREDICTED contacts for loss weighting (no data leakage)
        contact_soft = torch.sigmoid(contact_logits)

        # Sequence loss with contact weighting - use PREDICTED contacts, not true labels
        seq_loss = contact_weighted_ce(
            cdr_logits, true_S[mask], contact_soft.detach(),
            alpha=self.contact_weight_alpha)

        # Coordinate loss
        coord_loss = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt

        # Pairing loss
        if len(cdr_range) > 1 and (S == VOCAB.symbol_to_idx(VOCAB.BOA)).sum() == len(cdr_range):
            pairing_loss = self._contrastive_pairing_loss(aa_embd, S, cdr_range, segment_idx)
        else:
            pairing_loss = torch.tensor(0.0, device=X.device)

        # Dock loss
        dock_loss = self._dock_loss(Z, true_X, cdr_range, S, segment_ids)

        # FIX #1: Contrastive fingerprint loss (learned, not heuristic)
        # FIX #5: Use PREDICTED contacts for FP loss (no data leakage)
        fp_loss = torch.tensor(0.0, device=X.device)
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            cdr_fp = pred_fps[offset:offset + length]
            cdr_pos = all_cdr_pos[i]
            ag_h = all_ag_h[i]
            ag_pos = all_ag_pos[i]
            # Use predicted contacts (soft), not true labels
            cdr_contacts_pred = contact_soft[offset:offset + length].detach()

            fp_loss = fp_loss + self.contrastive_fp_loss(
                cdr_fp, ag_h, ag_pos, cdr_pos, cdr_contacts_pred, self.dock_cutoff)
            offset += length
        fp_loss = fp_loss / len(cdr_range) if len(cdr_range) > 0 else fp_loss

        # Contact prediction loss with focal weighting
        bce = F.binary_cross_entropy_with_logits(contact_logits, contact_labels, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** 2
        contact_loss = (focal_weight * bce).mean()

        # Auxiliary loss
        if cdr_mask_1d is not None and aux_targets is not None and cdr_mask_1d.any():
            cdr_embeddings = aa_embd[cdr_mask_1d]
            aux_loss = self.cdr_feature_predictor.aux_loss(cdr_embeddings, aux_targets)
        else:
            aux_loss = torch.tensor(0.0, device=X.device)

        # Weighted loss combination
        loss = (seq_loss
                + self.alpha * coord_loss
                + self.beta * pairing_loss
                + self.gamma * dock_loss
                + self.zeta * fp_loss
                + self.contact_loss_weight * contact_loss
                + self.eta * aux_loss)

        return loss, seq_loss, coord_loss, pairing_loss, dock_loss, fp_loss, contact_loss, aux_loss

    def generate(self, X, S, L, offsets, node_features=None,
                 segment_type=None, interface_mask=None, global_mask=None, global_types=None,
                 germline_prior=None):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()

        residue_mask = ~global_mask
        residue_feats = node_features[residue_mask]
        residue_seg = segment_type[residue_mask]
        residue_interface = interface_mask[residue_mask] if interface_mask is not None else None
        H_0 = self.protein_feature_encoder(
            residue_feats, residue_seg, residue_interface, global_mask, global_types)

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)

        epitope_mask, cdr_mask = self._build_masks(S, segment_ids, cdr_range, X=true_X)
        H, Z, aa_embd = self.gnn(
            H_0, X, edges_list, edge_feats_list, node_feats, segment_ids,
            self.interface_only, epitope_mask, cdr_mask)

        X = X.clone()
        X[mask] = Z[mask]

        cdr_logits, _, _, _, _, _ = self._three_stage_logits_v2(
            aa_embd, S, segment_ids, offsets, cdr_range, Z, true_X,
            epitope_mask=epitope_mask, germline_prior=germline_prior)
        logits = cdr_logits.masked_fill(smask, float('-inf'))

        S[mask] = torch.argmax(logits, dim=-1)
        snll_all = self.seq_loss_fn(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range, logits

    def infer(self, batch, device):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)

        node_features = batch.get('node_features')
        segment_type = batch.get('segment_type')
        interface_mask = batch.get('interface_mask')
        global_mask = batch.get('global_mask')
        global_types = batch.get('global_types')
        germline_prior = batch.get('germline_prior')

        if node_features is not None:
            node_features = node_features.to(device)
            segment_type = segment_type.to(device)
            interface_mask = interface_mask.to(device) if interface_mask is not None else None
            global_mask = global_mask.to(device)
            global_types = global_types.to(device)
        if germline_prior is not None:
            germline_prior = germline_prior.to(device)

        snll_all, pred_S, pred_X, true_X, cdr_range, logits = self.generate(
            X, S, L, offsets,
            node_features=node_features, segment_type=segment_type,
            interface_mask=interface_mask,
            global_mask=global_mask, global_types=global_types,
            germline_prior=germline_prior)

        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.detach().cpu().numpy(), true_X.detach().cpu().numpy()

        seq, x, true_x, logits_list = [], [], [], []
        logit_offset = 0
        for start, end in cdr_range:
            end = end + 1
            length = end - start
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
            logits_list.append(logits[logit_offset:logit_offset + length].detach().cpu())
            logit_offset += length

        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length

        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()

        return ppl, seq, x, true_x, logits_list
