# LLM-FineTuning-QLora

Bu proje, vngrs-ai tarafından geliştirilen Kumru-2B büyük dil modelini QLoRA tekniği kullanarak tıbbi alanda (özellikle TUS sınavlarına yönelik) ince ayar yapmak için tasarlanmıştır. Proje, Hugging Face `unsloth` kütüphanesini kullanarak hızlı ve verimli bir fine-tuning süreci sunar.

## Özellikler

- **Kumru-2B Modeli**: vngrs-ai tarafından geliştirilen Kumru-2B modelini temel alır.
- **QLoRA Fine-tuning**: 4-bit nicemleme (quantization) ile QLoRA tekniğini kullanarak bellek verimli fine-tuning.
- **Tıbbi Alan Odaklı**: Türk Tıbbi Yeterlilik Sınavı (TUS) için özel olarak hazırlanmış veri seti (`turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1`) ile eğitim.
- **Unsloth Entegrasyonu**: Eğitim sürecini hızlandırmak için `unsloth` kütüphanesinin optimizasyonlarından faydalanılır.
- **Yapılandırılabilir Parametreler**: `config.py` dosyası aracılığıyla model, LoRA ve eğitim parametreleri kolayca yapılandırılabilir.

## Dosya Yapısı

- `config.py`: Model, LoRA ve eğitim parametrelerini içerir.
- `dataset.py`: Hugging Face veri setini yükler ve Kumru (Mistral) chat template formatına dönüştürür.
- `train.py`: Model yükleme, LoRA adaptörü ekleme, eğitici oluşturma ve eğitim sürecini yönetir.
- `main.py`: `train.py` dosyasındaki `main` fonksiyonunu çalıştırarak eğitim sürecini başlatır.
- `test_base.py`: Orijinal Kumru-2B modelini kullanarak test soruları sormak için bir betik.
- `test.py`: İnce ayarlanmış Kumru-2B modelini kullanarak test soruları sormak için bir betik.
- `requirements.txt`: Projenin bağımlılıklarını listeler.
- `.gitignore`: Versiyon kontrolünden hariç tutulacak dosyaları ve dizinleri belirtir (örneğin, eğitim çıktıları ve önbellek dosyaları).

## Kurulum

1.  **Depoyu Klonlayın:**

    ```bash
    git clone https://github.com/anilsrml/LLM-FineTuning-QLora.git
    cd LLM-FineTuning-QLora
    ```

2.  **Sanal Ortam Oluşturun ve Bağımlılıkları Yükleyin:**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # source .venv/bin/activate # Linux/macOS
    pip install -r requirements.txt
    ```

3.  **CUDA Sürümüne Göre Unsloth Kurulumu:**
    `requirements.txt` dosyasında `unsloth[cu124]` belirtilmiştir. Eğer farklı bir CUDA sürümü kullanıyorsanız, `requirements.txt` dosyasını düzenlemeniz gerekebilir. Unsloth dokümantasyonuna bakın: [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

## Kullanım

### Eğitimi Başlatma

Eğitim sürecini başlatmak için `main.py` dosyasını çalıştırın:

```bash
python main.py
```

Bu komut, `config.py` dosyasında belirtilen parametrelere göre modeli yükleyecek, LoRA adaptörlerini ekleyecek ve `turkerberkdonmez/TUSGPT-TR-Medical-Dataset-v1` veri seti üzerinde fine-tuning yapacaktır. Eğitilmiş model `kumru-medical-lora` https://huggingface.co/anilsrml/Kumru-2B-TUS-Medical  dizinine kaydedilecektir.

### İnce Ayarlanmış Modeli Test Etme

İnce ayarlanmış modeli test etmek için `test.py` dosyasını çalıştırın:

```bash
python test.py
```

Bu betik, kaydedilen LoRA adaptörünü yükleyecek ve tıbbi sorulara yanıtlar üretecektir.

### Orijinal Modeli Test Etme

İnce ayarlı olmayan orijinal Kumru-2B modelini test etmek için `test_base.py` dosyasını çalıştırın:

```bash
python test_base.py
```

## Yapılandırma

`config.py` dosyasını düzenleyerek eğitim parametrelerini, model ayarlarını ve LoRA adaptör özelliklerini değiştirebilirsiniz.

```python
# config.py içeriği
# ...
```

## Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Lütfen bir pull request açmadan önce değişikliklerinizi detaylı bir şekilde açıklayın.

---
