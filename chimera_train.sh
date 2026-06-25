#!/bin/bash
# CompFirst training script (Hydra-based)
# Contact-First CDR Design with Distance-Biased Attention
# Usage: bash chimera_train.sh --gpu 0 --epochs 100

# ==============================================================================
# TRAINING CONFIG - Optimized hyperparameters from sweep
# ==============================================================================

# Training
gpu=0
max_epoch=100
batch_size=8
lr=6.31e-4
weight_decay=0
anneal_base=0.944
grad_clip=0.5
num_workers=4
seed=42

# Dataset
split="epitope_group"   # epitope_group | antigen_fold | temporal | random
cdr_type="3"            # 1 (H1) | 2 (H2) | 3 (H3)

# Model architecture
embed_size=32
hidden_size=256
n_layers=4
dropout=0.1
num_attn_heads=2
n_virtual_nodes=3

# CompFirst-specific
fingerprint_dim=32
fp_k=4
contact_weight_alpha=4.466
dist_attn_scale=4.0

# Loss weights (from sweep)
alpha=0.598         # coordinate loss
beta=0.103          # contrastive pairing loss
gamma=0.233         # docking proximity loss
zeta=0.020          # fingerprint loss
contact_loss_weight=1.763  # contact prediction loss
eta=0.2             # auxiliary CDR feature reconstruction
dock_cutoff=10.0

# Feature modes
node_feats_mode="1111"
edge_feats_mode="1111"
interface_only=1
mode="111"

# Germline prior (L1)
use_germline_prior=true   # set true to enable germline (J,pos) log-prior fusion
germline_prior_lambda_init=1.0

# Callbacks
early_stopping_patience=10
checkpoint_mode="min"

# WandB
wandb_enabled=true
wandb_project="chimera"

# Run name
run_name="compfirst-germline-epitopegroup-v0"

# ==============================================================================
# CLI OVERRIDES - These override the above config
# ==============================================================================

usage() {
    echo "Usage: $0 [--gpu <id>] [--epochs <n>]"
    echo ""
    echo "All other parameters are configured in the script itself."
    echo "Edit the CONFIG section at the top of this file."
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Parse CLI args (optional overrides)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) gpu="$2"; shift 2 ;;
        --epochs) max_epoch="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

# ==============================================================================
# RUN TRAINING
# ==============================================================================

mkdir -p logs

timestamp=$(date +%Y%m%d-%H%M%S)
log_file="logs/${timestamp}.log"

echo "=========================================="
echo "CompFirst Training (Hydra)"
echo "=========================================="
echo "GPU: $gpu | Epochs: $max_epoch | Batch: $batch_size | LR: $lr"
echo "Split: $split | CDR: H$cdr_type"
echo "Log: $log_file"
echo "=========================================="

nohup python -u trainer.py \
    training.gpu=$gpu \
    training.max_epoch=$max_epoch \
    training.batch_size=$batch_size \
    training.lr=$lr \
    training.weight_decay=$weight_decay \
    training.anneal_base=$anneal_base \
    training.grad_clip=$grad_clip \
    training.num_workers=$num_workers \
    training.seed=$seed \
    training.run_name="$run_name" \
    dataset.split=$split \
    dataset.cdr_type=\"$cdr_type\" \
    model.embed_size=$embed_size \
    model.hidden_size=$hidden_size \
    model.n_layers=$n_layers \
    model.dropout=$dropout \
    model.num_attn_heads=$num_attn_heads \
    model.n_virtual_nodes=$n_virtual_nodes \
    model.fingerprint_dim=$fingerprint_dim \
    model.fp_k=$fp_k \
    model.contact_weight_alpha=$contact_weight_alpha \
    model.dist_attn_scale=$dist_attn_scale \
    model.alpha=$alpha \
    model.beta=$beta \
    model.gamma=$gamma \
    model.zeta=$zeta \
    model.contact_loss_weight=$contact_loss_weight \
    model.eta=$eta \
    model.dock_cutoff=$dock_cutoff \
    model.node_feats_mode=\"$node_feats_mode\" \
    model.edge_feats_mode=\"$edge_feats_mode\" \
    model.interface_only=$interface_only \
    model.mode=\"$mode\" \
    model.use_germline_prior=$use_germline_prior \
    model.germline_prior_lambda_init=$germline_prior_lambda_init \
    callbacks.early_stopping.patience=$early_stopping_patience \
    callbacks.checkpoint.mode=$checkpoint_mode \
    wandb.enabled=$wandb_enabled \
    wandb.project=$wandb_project \
    > "$log_file" 2>&1 &

pid=$!
echo "Started (PID: $pid)"
echo "Monitor: tail -f $log_file"
echo "Kill: kill $pid"
