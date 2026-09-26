"""
RetiLink ML configuration (mBRSET only).

Outputs mirror OculoMoE's config: data_dir + exp_name + exp_no, i.e.
  /data/users3/nshaik3/Projects/Oculomics/RetiLink/<RETILINK_EXP_NO>/
    splits.json                     frozen patient-grouped split (copied from OculoMoE P1)
    label_audit.json                label codings / counts / missingness
    ckpt/image_seed<S>.pth          fine-tuned multi-task image model (LoRA + unfrozen blocks + heads)
    ckpt/image_seed<S>_history.json per-epoch metrics
    ckpt/systemic.joblib            patient-level systemic association models
    features/                       per-image embeddings + logits for every image (per seed)
    calibration.json                temperatures + frozen validation thresholds
    metrics/                        test metrics (json/csv), model_card.json
"""
import os

SEED = 42

MBRSET_ROOT = "/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0/"
MBRSET_IMAGES = os.path.join(MBRSET_ROOT, "images/")
MBRSET_LABELS = os.path.join(MBRSET_ROOT, "labels_mbrset.csv")
P1_SPLIT = "/data/users3/nshaik3/Projects/Oculomics/OculoMoE/cache/mbrset/splits_protocol_P1.json"

DATA_DIR = "/data/users3/nshaik3/"
EXP_NAME = "Projects/Oculomics/RetiLink/"
SMOKE = os.environ.get("RETILINK_SMOKE", "0") == "1"
EXP_NO = "smoke" if SMOKE else os.environ.get("RETILINK_EXP_NO", "1")
OUTPUT_DIR = os.path.join(DATA_DIR, EXP_NAME, EXP_NO)
CKPT_DIR = os.path.join(OUTPUT_DIR, "ckpt")
FEAT_DIR = os.path.join(OUTPUT_DIR, "features")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# ---------------------------------------------------------------- image model
BACKBONE = "facebook/dinov2-large"
IMAGE_SIZE = 224 if SMOKE else 392
LORA_RANK = 16
UNFREEZE_LAST = 4
EPOCHS = 2 if SMOKE else 15
PATIENCE = 4
BATCH_SIZE = 8 if SMOKE else 16
LR_LORA = 1e-4
LR_BLOCKS = 1e-5
WEIGHT_DECAY = 0.05
SMOKE_PATIENTS = 60
SEEDS = [42, 43]

# (name, type, n_out, loss weight). Systemic heads are auxiliary: they shape the
# embedding used by the patient-level systemic models, not shown directly.
SYSTEMIC_TARGETS = ["systemic_hypertension", "nephropathy", "vascular_disease", "neuropathy",
                    "acute_myocardial_infarction", "diabetic_foot", "obesity"]
HEADS = [
    ("quality_poor", "binary", 1, 1.0),   # final_quality == no  -> image unusable
    ("dr_referable", "binary", 1, 1.0),   # final_icdr >= 2      -> P0 screening endpoint
    ("icdr", "ordinal", 4, 1.0),          # final_icdr 0..4 (CORAL)
    ("edema", "binary", 1, 1.0),          # final_edema == yes   -> research signal
] + [(t, "binary", 1, 0.3) for t in SYSTEMIC_TARGETS]
RETINAL_HEADS = ["quality_poor", "dr_referable", "edema"]

# --------------------------------------------------------------- systemic model
METADATA_FEATURES = ["age", "sex", "dm_time", "insulin", "oraltreatment_dm"]
SYSTEMIC_ENABLE_MIN_AUROC = 0.65     # heads below this on validation show "Not evaluated"
SYSTEMIC_ENABLE_MIN_LOWER = 0.55     # ... and the validation patient-bootstrap 95% lower bound must reach this

# ------------------------------------------------------------- operating points
TARGET_DR_SENSITIVITY = 0.90
TARGET_QUALITY_RECALL = 0.95


def ensure_dirs():
    for d in (OUTPUT_DIR, CKPT_DIR, FEAT_DIR, METRICS_DIR):
        os.makedirs(d, exist_ok=True)
