import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from config import (
    MODEL_NAME, OUTPUT_DIR, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, RANDOM_STATE,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, NUM_EPOCHS,
    WARMUP_STEPS, MAX_STEPS, SAVE_STEPS, LOGGING_STEPS,
)
from dataset import load_and_prepare


def load_model():
    """Kumru-2B modelini Unsloth üzerinden 4-bit QLoRA ile yükler."""
    print("Model yükleniyor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # otomatik tespit (bf16/fp16)
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer


def add_lora(model):
    """Modele LoRA adaptörleri ekler."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",   # VRAM optimizasyonu
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, train_dataset, eval_dataset) -> SFTTrainer:
    """TrainingArguments ve SFTTrainer nesnesini oluşturur."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",         # VRAM tasarrufu için 8-bit optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",           # wandb/tensorboard devre dışı
        seed=RANDOM_STATE,
        dataloader_num_workers=0,   # Windows uyumluluğu
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )


def save_model(model, tokenizer):
    """LoRA adapter ve tokenizer'ı diske kaydeder."""
    print(f"\nLoRA adapter kaydediliyor -> {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Kayıt tamamlandı.")

    # Tam birleştirilmiş model kaydetmek için (opsiyonel):
    # model.save_pretrained_merged(
    #     OUTPUT_DIR + "-merged",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )

    # GGUF formatında kaydetmek için (llama.cpp / Ollama uyumlu):
    # model.save_pretrained_gguf(
    #     OUTPUT_DIR + "-gguf",
    #     tokenizer,
    #     quantization_method="q4_k_m",
    # )


def main():
    train_dataset, eval_dataset = load_and_prepare()

    model, tokenizer = load_model()
    model = add_lora(model)

    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset)

    print("\nEğitim başlıyor...")
    stats = trainer.train()
    print(f"Eğitim tamamlandı. Toplam süre: {stats.metrics['train_runtime']:.1f} saniye")

    save_model(model, tokenizer)
    print(f"\nFine-tune tamamlandı! Model dizini: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
