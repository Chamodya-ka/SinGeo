#!/usr/bin/env bash
# roundP41 replicate on hpc-cvgl01 (A40-48Q).
#
# Recipe: server B roundP41-p39-tau1gt2_20261111-165807 (CVUSA, 82.55 Avg), its 23
# SINGEO_* overrides value for value. Only the paths differ. SEED=None as in P41, so
# this is an independent sample, not a bit-identical rerun.
#
#   bash run_p41_cvgl01.sh check    preflight only: code, python env, data, GPU, config
#   bash run_p41_cvgl01.sh launch   preflight, then start detached in screen
#   bash run_p41_cvgl01.sh tail     last lines of the run's log.txt
#   bash run_p41_cvgl01.sh attach   attach to the screen session (detach: Ctrl-a d)
#   bash run_p41_cvgl01.sh stop     kill the screen session
#
# Paths can be overridden from the environment, e.g.
#   PY=/path/to/venv/bin/python DATA=/path/to/CVPR_subset bash run_p41_cvgl01.sh check
set -euo pipefail

REPO="${REPO:-/home/71/25021871/temp/SinGeo-h100}"
PY="${PY:-/home/71/25021871/Workspace/phd/bin/python}"
DATA="${DATA:-/data/cvusa/CVPR_subset}"
OUT="${OUT:-/data/singeo/checkpoint}"
GPS_DICT="${GPS_DICT:-$REPO/data/CVUSA/CVPR_subset/gps_dict.pkl}"
LOGS="${LOGS:-/data/singeo/run_logs}"
GPU="${GPU:-0}"                                  # physical GPU index (CUDA_VISIBLE_DEVICES)
NAME="${NAME:-roundP41-p39-tau1gt2-cvgl01}"
MODEL="convnext_base.fb_in22k_ft_in1k_384"

# --- roundP41, exactly as logged in _20261111-165807/log.txt ---------------------
P41=(
    SINGEO_SEED=None
    SINGEO_FOV_PAD=true SINGEO_FOV_BORDER_PX=0
    SINGEO_FOV_SAMPLING=gaussian
    SINGEO_FOV_GAUSSIAN_STD=40.0
    SINGEO_FOV_GAUSSIAN_RAMP_FRAC=0.3
    SINGEO_FOV_CURRICULUM_END=90.0
    SINGEO_FOV_GAUSSIAN_CLIP_LO=70.0
    SINGEO_RNC_WEIGHT=0.25 SINGEO_RNC_TAU=0.5
    SINGEO_RNC_POSITIVES_ONLY=true SINGEO_RNC_HARDEST_NEGATIVE=true
    "SINGEO_RNC_GROUP_WEIGHTS=(1.0, 0.0, 1.0, 0.0)"
    SINGEO_RNC_EXCLUDE_TIES=true
    SINGEO_RNC_TAU_RANK1=1.0
    SINGEO_AERIAL_ROTATION=continuous SINGEO_AERIAL_CIRCULAR_MASK=false
    SINGEO_OVERLAP_GATE_MEASURE=circle
    SINGEO_INFONCE_GATE_SCALES_TERMS=true
    "SINGEO_INFONCE_TERM_WEIGHTS=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)"
    SINGEO_DSS_FOV_FROM_MEASURED=true
    "SINGEO_EVAL_FOV_EXTRA=(360.0, 270, 180.0, 70.0, 0.0)"
    SINGEO_NUM_WORKERS=4
)
PATHS=(
    "SINGEO_DATA_FOLDER=$DATA"
    "SINGEO_MODEL_PATH=$OUT"
    "SINGEO_GPS_DICT_PATH=$GPS_DICT"
)
NOTE='P41 replicate on hpc-cvgl01 (A40-48Q, server B ran it on an H100 NVL). Same 23 overrides as server B roundP41-p39-tau1gt2
_20261111-165807 (82.55 Avg: 95.35 / 92.05 / 75.72 / 67.09); only data, output and gps
dict paths differ. Seed None as in P41, so an independent sample, not a bit-identical rerun.
Code: branch padding-issue-overcome-h100 @ d7157da, synced from server B. Tests run-to-run spread of the P41 recipe on a second machine.'

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; WARNED=1; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILED=1; }

preflight() {
    FAILED=0; WARNED=0
    echo "== preflight on $(hostname) =="

    # code
    if [[ -f "$REPO/train_singeo_cvusa.py" ]]; then ok "trainer: $REPO/train_singeo_cvusa.py"
    else bad "no train_singeo_cvusa.py in REPO=$REPO"; return 1; fi
    if git -C "$REPO" rev-parse --short HEAD >/dev/null 2>&1; then
        ok "git: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
    fi

    # python env
    if [[ ! -x "$PY" ]]; then bad "python not found: PY=$PY  (set PY=/path/to/venv/bin/python)"; return 1; fi
    if "$PY" - <<'EOF'
import sys
import torch, timm, albumentations, transformers, pandas, cv2
print("  ok    python %s  torch %s (cuda %s)  timm %s  albumentations %s  transformers %s"
      % (sys.version.split()[0], torch.__version__, torch.version.cuda, timm.__version__,
         albumentations.__version__, transformers.__version__))
if not torch.cuda.is_available():
    print("  \033[31mFAIL\033[0m  torch sees no CUDA device"); sys.exit(1)
if albumentations.__version__ != "1.3.1":
    print("  \033[31mFAIL\033[0m  albumentations must be 1.3.1 (2.x crashes LimitedFoV and ignores "
          "CoarseDropout/ImageCompression args): pip install albumentations==1.3.1"); sys.exit(1)
EOF
    then :; else bad "python environment (see above)"; fi

    # data
    for f in splits/train-19zl.csv splits/val-19zl.csv; do
        [[ -f "$DATA/$f" ]] || bad "missing $DATA/$f  (set DATA=/path/to/CVPR_subset)"
    done
    if [[ -f "$DATA/splits/train-19zl.csv" && -f "$DATA/splits/val-19zl.csv" ]]; then
        local ntr nva
        ntr=$(wc -l < "$DATA/splits/train-19zl.csv"); nva=$(wc -l < "$DATA/splits/val-19zl.csv")
        if [[ "$ntr" -eq 35532 && "$nva" -eq 8884 ]]; then ok "data: $DATA (train $ntr, val $nva)"
        else warn "data: train $ntr / val $nva rows, server B has 35532 / 8884"; fi
        local first; first=$(head -1 "$DATA/splits/train-19zl.csv" | cut -d, -f1)
        [[ -f "$DATA/$first" ]] && ok "images present (first in split: $first)" \
                                || bad "image listed in split not found: $DATA/$first"
    fi
    if [[ -f "$GPS_DICT" ]]; then ok "gps dict: $GPS_DICT"
    else bad "missing gps dict $GPS_DICT (copy data/CVUSA/CVPR_subset/gps_dict.pkl from server B, or set GPS_DICT=)"; fi

    # output
    if mkdir -p "$OUT" "$LOGS" 2>/dev/null && [[ -w "$OUT" ]]; then
        ok "output: $OUT  ($(df -h "$OUT" | awk 'NR==2{print $4}') free; a run writes ~20 checkpoints of ~350 MB)"
    else bad "cannot write OUT=$OUT"; fi

    # screen, GPU
    command -v screen >/dev/null && ok "screen: $(screen -v | head -1)" || bad "screen not installed"
    if command -v nvidia-smi >/dev/null; then
        local line; line=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || true)
        if [[ -z "$line" ]]; then bad "no GPU with index $GPU"
        else
            local used total; used=$(cut -d, -f3 <<<"$line" | tr -d ' '); total=$(cut -d, -f4 <<<"$line" | tr -d ' ')
            if (( total - used >= 25000 )); then ok "GPU $line MiB (needs ~23 GB)"
            else warn "GPU $GPU has only $(( (total-used)/1024 )) GiB free; P41 needs ~23 GB"; fi
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null | sed 's/^/        running on GPU: /'
        fi
    fi

    # config: this copy's defaults + P41 overrides vs. P41's effective config
    if ( cd "$REPO" && env "${P41[@]}" "${PATHS[@]}" "$PY" - <<'EOF'
import ast, importlib.util, os, sys
sys.path.insert(0, os.getcwd())
REF = ast.literal_eval("""{'aerial_arc_end': 180.0,
 'aerial_arc_start': 360.0,
 'aerial_circular_mask': False,
 'aerial_mask_mode': 'off',
 'aerial_ramp_frac': 1.0,
 'aerial_rot_max': 180.0,
 'aerial_rotation': 'continuous',
 'batch_size': 16,
 'batch_size_eval': 16,
 'checkpoint_start': None,
 'clip_grad': 100.0,
 'conv_padding_mode': 'zeros',
 'crop_dropout_strength': 0.5,
 'cudnn_benchmark': True,
 'cudnn_deterministic': False,
 'custom_sampling': True,
 'dataset': 'cvusa',
 'decay_exclue_bias': False,
 'device': 'cuda:0',
 'dss_fov_from_measured': True,
 'enable_aerial_crop': True,
 'epochs': 80,
 'eval_every_n_epoch': 4,
 'eval_fov_extra': (360.0, 270, 180.0, 70.0, 0.0),
 'fov': 90,
 'fov_border_px': 0,
 'fov_border_q1': False,
 'fov_curriculum_end': 90.0,
 'fov_full_pano_epochs': 0,
 'fov_full_pano_prob': 0.0,
 'fov_gap_segments': 4,
 'fov_gaussian_clip_lo': 70.0,
 'fov_gaussian_ramp_frac': 0.3,
 'fov_gaussian_std': 40.0,
 'fov_pad': True,
 'fov_pad_batch_max': False,
 'fov_pad_random_start': True,
 'fov_ramp_delay_epochs': 0,
 'fov_ramp_frac': 0.2,
 'fov_sampling': 'gaussian',
 'fov_schedule_lambda': 0.0,
 'gps_sample': True,
 'gpu_ids': (0,),
 'grad_checkpointing': False,
 'ground_roll_q1': False,
 'ground_roll_q1_ramp_frac': 0.0,
 'img_size': 384,
 'infonce_gate_mass_target': None,
 'infonce_gate_normalise_mass': False,
 'infonce_gate_scales_terms': True,
 'infonce_hold_loss1_share': False,
 'infonce_term_weights': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
 'infonce_weight': 1.0,
 'input_collage_every': 10,
 'input_collage_samples': 4,
 'label_smoothing': 0.1,
 'lr': 0.0001,
 'lr_end': 0.0001,
 'mask_fill': 'mean',
 'mixed_precision': True,
 'model': 'convnext_base.fb_in22k_ft_in1k_384',
 'negative_tiering': 'embed',
 'neighbour_range': 128,
 'neighbour_select': 64,
 'normalize_features': True,
 'num_workers': 4,
 'overlap_gate_measure': 'circle',
 'overlap_gated_infonce': True,
 'poly_power': 1.5,
 'prob_flip': 0.5,
 'prob_rotate': 0.75,
 'random_fov': False,
 'resume_dss_fov': None,
 'rnc_exclude_ties': True,
 'rnc_geo_max_km': None,
 'rnc_geo_source': 'coords',
 'rnc_group_weights': (1.0, 0.0, 1.0, 0.0),
 'rnc_hardest_negative': True,
 'rnc_loss_ratio': None,
 'rnc_loss_ratio_momentum': 0.99,
 'rnc_negative_margin': 0.01,
 'rnc_positive_overlap': 'circle',
 'rnc_positive_scale': 0.5,
 'rnc_positives_only': True,
 'rnc_similarity': 'cosine',
 'rnc_tau': 0.5,
 'rnc_tau_rank1': 1.0,
 'rnc_weight': 0.25,
 'scheduler': 'cosine',
 'seed': None,
 'sim_sample': True,
 'start_epoch': 1,
 'use_rnc': True,
 'verbose': True,
 'warmup_epochs': 1,
 'zero_shot': False}""")
NEUTRAL = {"rnc_loss_form": "rnc", "rince_variant": "out"}     # fields added after P41, P41's path
SKIP = {"data_folder", "model_path", "gps_dict_path", "sat_embedding_csv", "gps_coords_csv"}
spec = importlib.util.spec_from_file_location("trainer", "train_singeo_cvusa.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.Configuration()
try:
    m.apply_env_overrides(c)
except ValueError as e:
    print("  \033[31mFAIL\033[0m  this trainer rejects a P41 override (code too old?): %s" % e); sys.exit(1)
cur = {k: getattr(c, k) for k in dir(c)
       if not k.startswith("_") and not callable(getattr(c, k)) and k not in SKIP}
diff  = sorted(k for k in REF if k in cur and cur[k] != REF[k])
miss  = sorted(k for k in REF if k not in cur)
extra = sorted(k for k in cur if k not in REF)
bad = False
for k in diff:
    print("  \033[31mFAIL\033[0m  config %s = %r here, P41 had %r" % (k, cur[k], REF[k])); bad = True
for k in miss:
    print("  \033[31mFAIL\033[0m  config field %s missing here (P41 had %r)" % (k, REF[k])); bad = True
for k in extra:
    if k in NEUTRAL and cur[k] == NEUTRAL[k]:
        continue
    print("  \033[31mFAIL\033[0m  config field %s = %r is newer than P41; check it keeps the P41 path" % (k, cur[k])); bad = True
if bad:
    sys.exit(1)
print("  ok    effective config matches P41 on all %d fields (paths excluded)" % len(REF))
EOF
    ); then :; else bad "effective config differs from P41 (see above)"; fi

    # pretrained weights (timm / Hugging Face): downloads once if not cached
    if CUDA_VISIBLE_DEVICES="" "$PY" -c "import timm; timm.create_model('$MODEL', pretrained=True, num_classes=0)" >/dev/null 2>&1
    then ok "pretrained $MODEL loads"
    else bad "cannot load pretrained $MODEL (no internet and no HF cache?): run  $PY -c \"import timm; timm.create_model('$MODEL', pretrained=True)\"  to see why"; fi

    if (( FAILED )); then echo "== preflight FAILED: fix the items above =="; return 1; fi
    (( WARNED )) && echo "== preflight passed with warnings ==" || echo "== preflight passed =="
}

launch() {
    if screen -list | grep -q "\.${NAME}[[:space:]]"; then
        echo "!! screen session '$NAME' already running ('bash $0 stop' first)"; exit 1
    fi
    preflight || exit 1
    mkdir -p "$LOGS"
    local console="$LOGS/${NAME}.console" runner="$LOGS/${NAME}.cmd.sh"
    {
        echo '#!/usr/bin/env bash'
        echo "# written by $0 on $(date '+%F %T') -- the exact command this run used"
        echo 'set -o pipefail'
        printf 'cd %q || exit 1\n' "$REPO"
        printf 'export CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1\n' "$GPU"
        printf 'env'
        printf ' %q' "${P41[@]}" "${PATHS[@]}" "SINGEO_RUN_NAME=$NAME" "SINGEO_RUN_NOTE=$NOTE" "$PY"
        printf ' train_singeo_cvusa.py 2>&1 | tee -a %q\n' "$console"
    } > "$runner"
    screen -dmS "$NAME" bash "$runner"
    sleep 2
    if screen -list | grep -q "\.${NAME}[[:space:]]"; then
        echo "started '$NAME' on GPU $GPU"
        echo "  console : $console"
        echo "  command : $runner"
        echo "  log.txt : $OUT/$MODEL/${NAME}_<stamp>/log.txt   ('bash $0 tail')"
    else
        echo "!! session exited immediately; last console lines:"; tail -n 30 "$console" 2>/dev/null; exit 1
    fi
}

run_dir() { ls -dt "$OUT/$MODEL/${NAME}_"* 2>/dev/null | head -1; }

case "${1:-}" in
    check)  preflight ;;
    launch) launch ;;
    tail)   d=$(run_dir); [[ -n "$d" ]] || { echo "no run dir yet under $OUT/$MODEL"; exit 1; }
            echo "== $d/log.txt =="; tail -n "${2:-40}" "$d/log.txt" ;;
    attach) screen -r "$NAME" ;;
    stop)   screen -S "$NAME" -X quit && echo "stopped '$NAME'" ;;
    *)      sed -n '2,16p' "$0"; exit 1 ;;
esac
