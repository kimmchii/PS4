#!/usr/bin/env bash
# ============================================================
# TSE-ASR end-to-end training launcher (single GPU & multi-GPU DDP)
#
# Usage:
#   # Fresh training from a config name (configs/config_<name>.yaml)
#   bash run_train.sh --model bsrnn_ecapa_vox1 --gpus 0,1,4,5
#   bash run_train.sh --model tfmap_context_100 --gpus 0
#
#   # Load config + weights from an existing exp dir into a new experiment (finetune)
#   bash run_train.sh --model exp/20240101_120000_bsrnn_ecapa_vox1 --gpus 0,1
#
#   # Continue training in the same exp dir (resume)
#   bash run_train.sh --model exp/20240101_120000_bsrnn_ecapa_vox1 --resume --gpus 0,1,4,5
#
#   # Config name + resume: continue in the most recent matching exp dir
#   bash run_train.sh --model bsrnn_ecapa_vox1 --resume --gpus 0,1,4,5
#
# Arguments:
#   --model   <name|path>  config name (any configs/config_<name>.yaml, e.g. bsrnn_ecapa_vox1)
#                          or the path of an existing experiment dir under exp/
#   --resume               flag: continue training in place in the --model dir
#   --gpus    <ids>        comma-separated GPU ids, e.g. 0,1,4,5
# ============================================================

set -euo pipefail

# ── Script directory ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Known model names and directories ─────────────────────────────────────
PRETRAINED_ROOT="/home/yuque3/nwy/real-t/REAL-TSE-Challenge/pretrained"
EXP_ROOT="$SCRIPT_DIR/exp"
KNOWN_MODELS=("bsrnn_ecapa_vox1" "tfmap_context_100")

# ── Argument parsing (manual, getopt-style) ───────────────────────────────
MODEL=""
RESUME=0
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash run_train.sh --model <name|path> [--resume] [--gpus 0,1,4,5]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "[ERROR] --model is required (config name or exp directory path)"
    echo "Usage: bash run_train.sh --model <name|path> [--resume] [--gpus 0,1,4,5]"
    exit 1
fi

# ── Helper: newest .pt checkpoint under models/ ───────────────────────────
find_latest_ckpt() {
    local models_dir="$1"
    # Newest .pt by modification time
    ls -t "$models_dir"/*.pt 2>/dev/null | head -1 || true
}

# ── Is MODEL a config name or a path? ─────────────────────────────────────
IS_NAME=0
for n in "${KNOWN_MODELS[@]}"; do
    if [[ "$MODEL" == "$n" ]]; then
        IS_NAME=1
        break
    fi
done
# Any other name also works if configs/config_<name>.yaml exists (e.g. bsrnn_ecapa_vox1_custom)
if [[ "$IS_NAME" -eq 0 && "$MODEL" != */* && -f "$SCRIPT_DIR/configs/config_${MODEL}.yaml" ]]; then
    IS_NAME=1
fi

CONFIG_FILE=""
RESUME_CKPT=""
INIT_CKPT=""
EXP_DIR_OVERRIDE=""

if [[ "$IS_NAME" -eq 1 ]]; then
    # ── Config-name mode ────────────────────────────────────────────────
    CONFIG_FILE="$SCRIPT_DIR/configs/config_${MODEL}.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] Config file not found: $CONFIG_FILE"
        echo "Available configs:"
        ls "$SCRIPT_DIR/configs/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml || echo "  (none)"
        exit 1
    fi

    if [[ "$RESUME" -eq 1 ]]; then
        # Find the most recent exp dir under exp/ ending in _<MODEL>
        LATEST_EXP=$(ls -dt "$EXP_ROOT"/*_"${MODEL}" 2>/dev/null | head -1 || true)
        if [[ -z "$LATEST_EXP" ]]; then
            echo "[ERROR] --resume: no experiment dir matching *_${MODEL} found under $EXP_ROOT"
            exit 1
        fi
        LATEST_CKPT=$(find_latest_ckpt "$LATEST_EXP/models")
        if [[ -z "$LATEST_CKPT" ]]; then
            echo "[ERROR] No .pt checkpoint found in $LATEST_EXP/models"
            exit 1
        fi
        EXP_DIR_OVERRIDE="$LATEST_EXP"
        RESUME_CKPT="$LATEST_CKPT"
        echo "[INFO] Resume mode: continuing experiment dir $LATEST_EXP"
        echo "[INFO] Loading checkpoint: $LATEST_CKPT"
    fi

else
    # ── Path mode ───────────────────────────────────────────────────────
    # Relative paths are resolved against SCRIPT_DIR
    if [[ ! -d "$MODEL" ]]; then
        ABS_MODEL="$SCRIPT_DIR/$MODEL"
        if [[ -d "$ABS_MODEL" ]]; then
            MODEL="$ABS_MODEL"
        else
            echo "[ERROR] Experiment dir not found: $MODEL"
            echo "Not a config name either; available: $(ls "$SCRIPT_DIR/configs/"config_*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^config_//; s/\.yaml$//' | tr '\n' ' ')"
            exit 1
        fi
    fi
    MODEL="$(cd "$MODEL" && pwd)"  # to an absolute path

    CONFIG_FILE="$MODEL/config.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] config.yaml not found in the experiment dir: $CONFIG_FILE"
        exit 1
    fi

    LATEST_CKPT=$(find_latest_ckpt "$MODEL/models")
    if [[ -z "$LATEST_CKPT" ]]; then
        echo "[ERROR] No .pt checkpoint found in $MODEL/models"
        exit 1
    fi

    if [[ "$RESUME" -eq 1 ]]; then
        # Continue in place: pass exp_dir so no new timestamped dir is created
        RESUME_CKPT="$LATEST_CKPT"
        EXP_DIR_OVERRIDE="$MODEL"
        echo "[INFO] Resume mode: continuing in place in $MODEL"
    else
        # finetune: load model weights only; optimizer/scheduler/epoch start fresh
        INIT_CKPT="$LATEST_CKPT"
        echo "[INFO] Finetune mode: initializing weights from $LATEST_CKPT into a new experiment dir"
    fi
    echo "[INFO] Loading checkpoint: $LATEST_CKPT"
fi

# ── GPU setup ─────────────────────────────────────────────────────────────
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    # Number of GPUs (count of comma-separated ids)
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
else
    # Auto-detect
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 1))
    else
        NUM_GPUS=1
    fi
fi

# ── Python ────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] Python not found: $PYTHON"
    exit 1
fi

# ── Environment: add wesep_ps4 (wesep + wespeaker) to PYTHONPATH ──────────
WESEP_PATH="$SCRIPT_DIR/wesep_ps4/wesep_real_tse"
WESPEAKER_PATH="$SCRIPT_DIR/wesep_ps4/wespeaker"
export PYTHONPATH="$WESEP_PATH:$WESPEAKER_PATH:${PYTHONPATH:-}"

# ── Print summary ─────────────────────────────────────────────────────────
echo "============================================================"
echo "  TSE-ASR end-to-end training"
echo "  Config    : $CONFIG_FILE"
echo "  GPU ids   : ${GPUS:-auto}"
echo "  GPU count : $NUM_GPUS"
[[ -n "$EXP_DIR_OVERRIDE" ]] && echo "  Exp dir   : $EXP_DIR_OVERRIDE"
[[ -n "$RESUME_CKPT"      ]] && echo "  Resume    : $RESUME_CKPT"
[[ -n "$INIT_CKPT"        ]] && echo "  Init      : $INIT_CKPT"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Build train.py arguments ──────────────────────────────────────────────
TRAIN_ARGS=(--config "$CONFIG_FILE")
[[ -n "$EXP_DIR_OVERRIDE" ]] && TRAIN_ARGS+=(--exp_dir "$EXP_DIR_OVERRIDE")
[[ -n "$RESUME_CKPT"      ]] && TRAIN_ARGS+=(--resume  "$RESUME_CKPT")
[[ -n "$INIT_CKPT"        ]] && TRAIN_ARGS+=(--pretrained_tse "$INIT_CKPT")

# ── Launch training ───────────────────────────────────────────────────────
if [[ "$NUM_GPUS" -le 1 ]]; then
    echo "[INFO] Single-GPU mode: $PYTHON train.py ${TRAIN_ARGS[*]}"
    exec "$PYTHON" train.py "${TRAIN_ARGS[@]}"
else
    TORCHRUN="${TORCHRUN:-torchrun}"
    if ! command -v "$TORCHRUN" &>/dev/null; then
        echo "[WARN] torchrun not found, falling back to python -m torch.distributed.run"
        TORCHRUN="$PYTHON -m torch.distributed.run"
    fi

    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    echo "[INFO] Multi-GPU DDP mode: ${TORCHRUN} --nproc_per_node=${NUM_GPUS} train.py ${TRAIN_ARGS[*]}"
    exec $TORCHRUN \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        train.py "${TRAIN_ARGS[@]}"
fi
