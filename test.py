from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# ─── Model Yükle ───────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./kumru-medical-lora",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # hız optimizasyonu

SYSTEM_PROMPT = (
    "Sen bir tıbbi asistansın. Türk tıbbi sınavlarına (TUS) yönelik "
    "kapsamlı ve doğru tıbbi bilgi sağlıyorsun."
)

# ─── Soru Sor ──────────────────────────────────────────
def sor(soru):
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{soru} [/INST]"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    print(f"\n{'─'*50}")
    print(f"SORU: {soru}")
    print(f"{'─'*50}")
    print("CEVAP: ", end="", flush=True)

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.1,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(f"{'─'*50}\n")

# ─── Test Soruları ─────────────────────────────────────
sor("Hipertansiyonun ilk basamak tedavisinde hangi ilaç kullanılır ?")
    