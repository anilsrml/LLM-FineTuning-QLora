from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# ─── Model Yükle ───────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="vngrs-ai/Kumru-2B",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = (
    "Sen bir tıbbi asistansın. Türk tıbbi sınavlarına (TUS) yönelik "
    "kapsamlı ve doğru tıbbi bilgi sağlıyorsun."
)

# ─── Soru Sor ──────────────────────────────────────────
def sor(soru):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": soru},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
            temperature=0.7,
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.1,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(f"{'─'*50}\n")

# ─── Test Soruları ─────────────────────────────────────
sor("Tip 2 diyabetes mellitusun birinci basamak tedavisinde tercih edilen ilaç nedir?")
