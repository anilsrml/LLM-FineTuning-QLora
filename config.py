# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "vngrs-ai/Kumru-2B"
OUTPUT_DIR     = "./kumru-medical-lora"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT   = True

# ─── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0
RANDOM_STATE = 42

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ─── Eğitim ───────────────────────────────────────────────────────────────────
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 8      # efektif batch = 8
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 1
WARMUP_STEPS     = 50
MAX_STEPS        = -1     # -1 → tüm epoch boyunca eğit
SAVE_STEPS       = 200
LOGGING_STEPS    = 25

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1"

SYSTEM_PROMPT = (
    "Sen bir tıbbi asistansın. Türk tıbbi sınavlarına (TUS) yönelik "
    "kapsamlı ve doğru tıbbi bilgi sağlıyorsun."
)
