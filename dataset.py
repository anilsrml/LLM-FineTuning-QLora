from datasets import load_dataset, DatasetDict
from config import DATASET_NAME, SYSTEM_PROMPT


def format_example(example: dict) -> dict:
    """
    instruction / output kolonlarını Kumru (Mistral) chat template formatına dönüştürür.

    Çıktı formatı:
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {instruction} [/INST] {output} </s>
    """
    text = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{example['instruction']} [/INST] "
        f"{example['output']} </s>"
    )
    return {"text": text}


def load_and_prepare() -> tuple:
    """
    Veri setini HuggingFace'ten indirir, train ve validation bölümlerini
    chat template formatına dönüştürür.

    Returns:
        train_dataset, eval_dataset
    """
    print(f"Veri seti yükleniyor: {DATASET_NAME}")
    raw: DatasetDict = load_dataset(DATASET_NAME)

    train_dataset = raw["train"].map(
        format_example,
        remove_columns=raw["train"].column_names,
        num_proc=4,
        desc="Eğitim verisi formatlanıyor",
    )

    eval_dataset = raw["validation"].map(
        format_example,
        remove_columns=raw["validation"].column_names,
        num_proc=4,
        desc="Doğrulama verisi formatlanıyor",
    )

    print(f"Eğitim örnekleri   : {len(train_dataset)}")
    print(f"Doğrulama örnekleri: {len(eval_dataset)}")
    print("\nÖrnek formatlı metin:")
    print(train_dataset[0]["text"][:400])

    return train_dataset, eval_dataset
