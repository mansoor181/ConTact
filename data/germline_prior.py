"""Germline (J-gene, IMGT-position) sequence prior for L1 fusion.

The H3 distribution factorizes the way the immune system generates it: the J-gene (assignable by
ANARCI HMM from the GIVEN heavy framework -- bioinformatics, no PLM) fixes the C-terminal anchor, and
each IMGT position has a germline-conditioned AA distribution. ConTact cannot learn these VDJ
statistics from 2337 examples, so we supply them as an additive log-prior at the sequence head.

Tables are built from TRAIN ONLY (no leakage). Backoff: (J, pos) categorical when it has >=3 counts,
else marginal(pos). Smoothing: +0.5. Log-probs live in the model's VOCAB index space (size 27).
"""
import os
import numpy as np
import torch
from collections import defaultdict, Counter

from data.pdb_utils import VOCAB

_AA = 'ACDEFGHIKLMNPQRSTVWY'
_MIN_JCOUNT = 3
_SMOOTH = 0.5
_LOG_FLOOR = -15.0  # non-AA / unobserved columns: exp(-15)~3e-7 mass, finite (no CE blow-up)


class GermlinePrior:
    def __init__(self, train_ids, complex_features_dir, germline_assign,
                 numbering_scheme='imgt', cdr_type='3'):
        self.nv = len(VOCAB)
        self.aa_idx = {a: VOCAB.symbol_to_idx(a) for a in _AA}
        self.assign = germline_assign  # cid -> (V_allele, J_allele)
        self.cdr_idx = int(cdr_type) - 1
        self.scheme = numbering_scheme
        pj = defaultdict(Counter)
        pm = defaultdict(Counter)
        for cid in train_ids:
            p = os.path.join(complex_features_dir, cid + '.pt')
            if not os.path.exists(p):
                continue
            d = torch.load(p, map_location='cpu', weights_only=False)
            cm = d['cdr_masks'][self.scheme]['heavy']
            idx = [i for i, x in enumerate(cm) if x == self.cdr_idx]
            if len(idx) < 4:
                continue
            num = d['numbering'][self.scheme]['heavy']
            seq = d['heavy_sequence'][:len(num)]
            if any(seq[i] not in self.aa_idx for i in idx):
                continue
            J = self._jgene(cid)
            for i in idx:
                key = (num[i][0], num[i][1])
                pm[key][seq[i]] += 1
                pj[(J, key)][seq[i]] += 1
        self.pj_log = {k: self._tolog(c) for k, c in pj.items() if sum(c.values()) >= _MIN_JCOUNT}
        self.pm_log = {k: self._tolog(c) for k, c in pm.items()}
        self._zero = np.zeros(self.nv, dtype=np.float32)

    def _jgene(self, cid):
        j = self.assign.get(cid, ('NA', 'NA'))[1]
        return j.split('*')[0] if j != 'NA' else 'NA'

    def _tolog(self, counter):
        v = np.full(self.nv, _LOG_FLOOR, dtype=np.float32)
        tot = sum(counter.values()) + _SMOOTH * len(_AA)
        for a in _AA:
            v[self.aa_idx[a]] = np.log((counter.get(a, 0) + _SMOOTH) / tot)
        return v

    def logvec(self, J, key):
        v = self.pj_log.get((J, key))
        if v is None:
            v = self.pm_log.get(key, self._zero)
        return v

    def build_tensor(self, cid, n_total, h_start, cdr_positions, heavy_numbering):
        """Full-length (n_total, nv) prior: zero everywhere except this complex's CDR positions."""
        gp = torch.zeros(n_total, self.nv, dtype=torch.float32)
        J = self._jgene(cid)
        for p in cdr_positions:
            if p < len(heavy_numbering):
                key = (heavy_numbering[p][0], heavy_numbering[p][1])
                gp[h_start + p] = torch.from_numpy(self.logvec(J, key))
        return gp
