"""CompFirst Trainer (Hydra-based).

Contact-First CDR Design with Distance-Biased Attention.

Architecture:
  1. FingerprintPredictor: predict complementary surface fingerprints
  2. ContactPredictor: predict CDR-AG contact map
  3. LocalCompInjector: K-NN aggregation gated by contact confidence
  4. Contact-weighted CE at sequence head

Usage:
    python trainer.py
    python trainer.py training.gpu=1 training.max_epoch=50
"""

import glob
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = _SCRIPT_DIR
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..', '..'))
_CHIMERA_ROOT = os.path.normpath(os.path.join(_GENBIO_ROOT, '..'))

sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, _GENBIO_ROOT)
sys.path.insert(0, _CHIMERA_ROOT)
sys.path.insert(0, os.path.join(_CHIMERA_ROOT, "baselines"))

from model.core import DockDesigner
from data.dataset import FeatureDataset
from data.germline_prior import GermlinePrior
from data.pdb_utils import VOCAB

from shared.features_v2 import get_available_complex_ids, load_split_ids
from chimera_utils import (
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, to_device, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)
from benchmark.evaluation.metrics import aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score, count_liabilities

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities", "top3_aar", "top5_aar"]
LOSS_KEYS = ["loss", "seq", "coord", "pair", "dock", "fp_loss", "contact_loss", "aux"]


def build_model_args(cfg: DictConfig) -> SimpleNamespace:
    """Convert Hydra config to SimpleNamespace for model compatibility."""
    m = cfg.model
    t = cfg.training
    return SimpleNamespace(
        embed_size=m.embed_size,
        hidden_size=m.hidden_size,
        n_layers=m.n_layers,
        dropout=m.dropout,
        num_attn_heads=m.num_attn_heads,
        n_virtual_nodes=m.n_virtual_nodes,
        fingerprint_dim=m.fingerprint_dim,
        fp_k=m.fp_k,
        contact_weight_alpha=m.contact_weight_alpha,
        dist_attn_scale=m.dist_attn_scale,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        zeta=m.zeta,
        contact_loss_weight=m.contact_loss_weight,
        eta=m.eta,
        dock_cutoff=m.dock_cutoff,
        node_feats_mode=str(m.node_feats_mode),
        edge_feats_mode=str(m.edge_feats_mode),
        interface_only=m.interface_only,
        mode=str(m.mode),
        use_germline_prior=m.use_germline_prior,
        germline_prior_lambda_init=m.germline_prior_lambda_init,
        anneal_base=t.anneal_base,
    )


def find_features():
    """Find the latest v2 features file."""
    final = '/home/exouser/data/chimera/processed/rich_features_v2.pt'
    if os.path.exists(final):
        return final
    ckpts = sorted(glob.glob('/home/exouser/data/chimera/processed/checkpoints_v2/ckpt_*.pt'))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError("No v2 features found")


def _agg(m):
    return {k: float(np.mean(v)) for k, v in m.items()}


def train_epoch(model, loader, opt, device, grad_clip):
    model.train()
    ms = {k: [] for k in LOSS_KEYS}
    for b in loader:
        b = to_device(b, device)
        extra_kwargs = {"cdr_mask_1d": b["cdr_mask_1d"], "aux_targets": b["aux_targets"]}
        if "germline_prior" in b:
            extra_kwargs["germline_prior"] = b["germline_prior"]
        vals = model(b["X"], b["S"], b["L"], b["offsets"],
                     b["node_features"], b["segment_type"],
                     b["interface_mask"], b["global_mask"], b["global_types"],
                     **extra_kwargs)
        opt.zero_grad()
        vals[0].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        for k, v in zip(LOSS_KEYS, vals):
            ms[k].append(v.item() if torch.is_tensor(v) else v)
    return _agg(ms)


def valid_epoch(model, loader, device):
    model.eval()
    ms = {k: [] for k in LOSS_KEYS}
    with torch.no_grad():
        for b in loader:
            b = to_device(b, device)
            extra_kwargs = {}
            if "germline_prior" in b:
                extra_kwargs["germline_prior"] = b["germline_prior"]
            vals = model(b["X"], b["S"], b["L"], b["offsets"],
                         b["node_features"], b["segment_type"],
                         b["interface_mask"], b["global_mask"], b["global_types"],
                         **extra_kwargs)
            for k, v in zip(LOSS_KEYS, vals):
                ms[k].append(v.item() if torch.is_tensor(v) else v)
    return _agg(ms)


def compute_topk_metrics(logits, true_seq, k_values=(3, 5)):
    """Compute Top-K AAR from logits."""
    results = {}
    L = len(true_seq)
    if logits.shape[0] != L:
        return {f'top{k}_aar': 0.0 for k in k_values}
    true_indices = [VOCAB.symbol_to_idx(aa) for aa in true_seq]
    for k in k_values:
        topk_preds = torch.topk(logits, k=k, dim=-1).indices
        in_topk = torch.tensor([true_indices[i] in topk_preds[i].tolist() for i in range(L)])
        results[f'top{k}_aar'] = in_topk.float().mean().item()
    return results


def run_inference(model, loader, device, cdr_type):
    model.eval()
    preds, metrics = [], {k: [] for k in METRIC_KEYS}
    cdr_label = f"H{cdr_type}"
    with torch.no_grad():
        for b in loader:
            ppls, seqs, xs, true_xs, logits_list = model.infer(b, device)
            ca_centroids = b.get('ca_centroids', [None] * len(b["L"]))
            for i in range(len(b["L"])):
                cid, true_seq = b['complex_ids'][i], b['true_cdr_seqs'][i]
                pred_ca = (xs[i][:, 1, :] if xs[i].ndim == 3 else xs[i])
                true_ca = (true_xs[i][:, 1, :] if true_xs[i].ndim == 3 else true_xs[i])
                if isinstance(pred_ca, torch.Tensor):
                    pred_ca = pred_ca.cpu().numpy()
                if isinstance(true_ca, torch.Tensor):
                    true_ca = true_ca.cpu().numpy()
                centroid = ca_centroids[i]
                if centroid is not None:
                    centroid_np = centroid.cpu().numpy() if isinstance(centroid, torch.Tensor) else centroid
                    pred_ca = pred_ca + centroid_np
                    true_ca = true_ca + centroid_np
                aar = chimera_aar(seqs[i], true_seq) if true_seq else 0.0
                try:
                    rmsd, tm = kabsch_rmsd(pred_ca, true_ca), chimera_tm_score(pred_ca, true_ca)
                except:
                    rmsd, tm = float('inf'), 0.0
                liab = count_liabilities(seqs[i])
                topk_metrics = compute_topk_metrics(logits_list[i], true_seq) if true_seq else {'top3_aar': 0.0, 'top5_aar': 0.0}
                for k, v in zip(METRIC_KEYS[:5], [ppls[i], aar, rmsd, tm, liab]):
                    metrics[k].append(v)
                for k in ['top3_aar', 'top5_aar']:
                    metrics[k].append(topk_metrics[k])
                preds.append({
                    "complex_id": cid, "cdr_type": cdr_label,
                    "pred_sequence": seqs[i], "true_sequence": true_seq,
                    "pred_coords": pred_ca, "true_coords": true_ca,
                    "ppl": ppls[i], "aar": aar, "rmsd": rmsd, "tm_score": tm, "n_liabilities": liab,
                    "top3_aar": topk_metrics['top3_aar'], "top5_aar": topk_metrics['top5_aar']
                })
    return preds, {k: float(np.mean(v)) for k, v in metrics.items() if v}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    args = build_model_args(cfg)
    args.cdr_type = str(cfg.dataset.cdr_type)
    args.split = cfg.dataset.split
    args.data_root = cfg.data_root
    args.output_dir = cfg.results_dir
    args.numbering_scheme = cfg.numbering_scheme
    args.features_path = cfg.features_path

    args.seed = cfg.training.seed
    args.gpu = cfg.training.gpu
    args.lr = cfg.training.lr
    args.weight_decay = cfg.training.weight_decay
    args.batch_size = cfg.training.batch_size
    args.max_epoch = cfg.training.max_epoch
    args.grad_clip = cfg.training.grad_clip
    args.num_workers = cfg.training.num_workers
    args.run_name = cfg.training.run_name
    args.patience = cfg.callbacks.early_stopping.patience

    args.use_wandb = cfg.wandb.enabled
    args.wandb_project = cfg.wandb.project

    args.test_only = getattr(cfg.training, 'test_only', False)
    args.checkpoint = getattr(cfg.training, 'checkpoint', None)

    run_pipeline(args)


def run_pipeline(c, wandb_run=None):
    seed_everything(c.seed)
    device = torch.device(f"cuda:{c.gpu}" if torch.cuda.is_available() else "cpu")
    cdr_label = f"H{c.cdr_type}"
    run_name = c.run_name or f"compfirst_cdr{c.cdr_type}_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    features_path = find_features()
    cf_dir = '/home/exouser/data/chimera/processed/complex_features'
    available = get_available_complex_ids()
    splits = load_split_ids(c.split, available_ids=available if c.split == 'random' else None, random_seed=c.seed)

    germline_prior = None
    if c.use_germline_prior:
        assign = pickle.load(open('/home/exouser/data/chimera/processed/germline_assignment.pkl', 'rb'))
        germline_prior = GermlinePrior(splits['train'], cf_dir, assign, c.numbering_scheme, c.cdr_type)
        print(f"Germline prior: {len(germline_prior.pj_log)} (J,pos) cells, {len(germline_prior.pm_log)} marginal(pos) cells")

    train_set = FeatureDataset(splits['train'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, germline_prior)
    valid_set = FeatureDataset(splits['val'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, germline_prior)
    test_set = FeatureDataset(splits['test'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, germline_prior)
    print(f"Data: {len(train_set)}/{len(valid_set)}/{len(test_set)} train/val/test")

    model = DockDesigner(c.embed_size, c.hidden_size, 4, n_layers=c.n_layers, dropout=c.dropout, cdr_type=c.cdr_type, args=c).to(device)
    print(f"CompFirst: {sum(p.numel() for p in model.parameters()):,} params")

    if c.use_wandb and not getattr(c, 'test_only', False):
        wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=True)

    if getattr(c, 'test_only', False):
        ckpt_path = c.checkpoint or os.path.join(save_dir, "checkpoints", "best.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    else:
        train_loader = DataLoader(train_set, c.batch_size, shuffle=True, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
        valid_loader = DataLoader(valid_set, c.batch_size, shuffle=False, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
        opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: c.anneal_base ** e)
        es, ckpt = EarlyStopping(patience=c.patience, mode="min"), ModelCheckpoint(save_dir, mode="min")

        for ep in range(c.max_epoch):
            t0 = time.time()
            tm = train_epoch(model, train_loader, opt, device, c.grad_clip)
            vm = valid_epoch(model, valid_loader, device)
            sched.step()
            _, vm2 = run_inference(model, valid_loader, device, c.cdr_type)
            cur_lr = opt.param_groups[0]["lr"]
            best = ckpt.save(model, opt, sched, ep, vm["loss"])
            print(f"Ep {ep:3d} | loss={tm['loss']:.4f}/{vm['loss']:.4f} "
                  f"seq={tm['seq']:.3f}/{vm['seq']:.3f} coord={tm['coord']:.3f}/{vm['coord']:.3f} "
                  f"dock={tm['dock']:.3f}/{vm['dock']:.3f} fp={tm['fp_loss']:.3f}/{vm['fp_loss']:.3f} "
                  f"contact={tm['contact_loss']:.3f}/{vm['contact_loss']:.3f} "
                  f"aar={vm2.get('aar',0):.3f} rmsd={vm2.get('rmsd',0):.2f} "
                  f"lr={cur_lr:.6f} {'*' if best else ''} [{time.time()-t0:.0f}s]")
            if wandb_run:
                log_dict = {"epoch": ep, "lr": cur_lr}
                log_dict.update({f"train_{k}": tm[k] for k in LOSS_KEYS})
                log_dict.update({f"val_{k}": vm[k] for k in LOSS_KEYS})
                log_dict.update({f"val_{k}": v for k, v in vm2.items()})
                wandb_run.log(log_dict)
            if es(vm["loss"]):
                break
        ckpt.load_best(model, device)

    test_loader = DataLoader(test_set, c.batch_size, shuffle=False, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
    preds, summary = run_inference(model, test_loader, device, c.cdr_type)
    pred_dir = os.path.join(save_dir, "predictions", cdr_label)
    if os.path.exists(pred_dir):
        for old_f in Path(pred_dir).glob("*.pt"):
            old_f.unlink()
    save_predictions(preds, pred_dir)
    if c.split != 'random':
        _, summary, _ = run_full_evaluation(pred_dir, c.split, c.data_root, cdr_type_hint=cdr_label, numbering_scheme=c.numbering_scheme)

    def fmt(v):
        return f"{v['mean']:.3f}" if isinstance(v, dict) else f"{v:.3f}"
    print(f"\nTest ({cdr_label}): " + " ".join(f"{k}={fmt(v)}" for k, v in summary.items()))

    if wandb_run:
        for k in FULL_METRIC_KEYS:
            v = summary.get(k, {})
            if isinstance(v, dict):
                v = v.get("mean", 0)
            wandb_run.log({f"test_{cdr_label}_{k}": v})
        wandb_run.log({"test_n": len(preds)})
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    main()
