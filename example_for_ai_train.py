from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import torch

# CUDA desteği kontrolü (GPU kullanımı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will be trained on: {device}")

# Veri kümesini yükle
ds = load_dataset("itopcu/hate-speech-target")

# Eğitim verisinden doğrulama verisi ayır
train_dataset = ds['train']
train_dataset = train_dataset.shuffle(seed=42)

# Veriyi %90 eğitim, %10 doğrulama olarak ayır
split_dataset = train_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# DistilBERT tokenizer'ı yükle
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize işlemi
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Dataset üzerinde tokenizasyon yap
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Label'ı sayısal değere dönüştürme: 'non' gibi değerleri geç
def convert_labels(x):
    try:
        x['label'] = int(x['label'])  # Sayısal dönüşüm
    except ValueError:  # Eğer 'label' sayısal değilse, uygun bir değeri atama (örn. 0 veya 1)
        x['label'] = 0  # Burada 'non' için 0 olarak ayarlıyoruz
    return x

# Label'ları temizle
train_dataset = train_dataset.map(convert_labels)
eval_dataset = eval_dataset.map(convert_labels)

# DistilBERT modelini yükle (ikili sınıflandırma için)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

# Eğitim için parametreleri ayarlayın
training_args = TrainingArguments(
    output_dir='./results',            # Sonuçları kaydedeceğimiz dizin
    evaluation_strategy="epoch",       # Değerlendirmeyi her epoch'ta yap
    save_strategy="epoch",             # Modeli her epoch sonunda kaydet
    learning_rate=1e-4,                # Öğrenme oranını 5e-5 olarak ayarladık
    per_device_train_batch_size=16,    # Batch boyutunu artırdık (T4 GPU ile bu değer iyi çalışır)
    per_device_eval_batch_size=16,     # Değerlendirme batch boyutunu artırdık
    num_train_epochs=6,                # Epoch sayısını artırdık (6 epoch)
    weight_decay=0.1,                 # Ağırlık düşüşü (modelin overfitting yapmaması için)
    gradient_accumulation_steps=4,     # Küçük batch'lerde işlem yaparak toplamda daha büyük batch etkisi yaratabiliriz
    logging_dir='./logs',              # Log dosyaları
    logging_steps=100,                 # Loglamayı her 100 adımda bir yapıyoruz
    save_steps=500,                    # Modeli her 500 adımda bir kaydediyoruz
    load_best_model_at_end=True,       # Eğitimi en iyi model ile bitir
    report_to="tensorboard",           # TensorBoard ile eğitim sürecini izlemek için
    fp16=True,                         # Karışık hassasiyetli eğitim (mixed precision training) etkinleştirildi
    dataloader_num_workers=4,          # Veri yükleme hızını artırmak için işçi sayısını artırdık
    warmup_steps=1000,                 # İlk başta düşük öğrenme oranı, sonra artırma
    lr_scheduler_type="linear",        # Öğrenme oranı azalmayı takip etsin
    run_name="hate_speech_classification", # Eğitimi isimlendirme
)

# Early stopping (erken durdurma) ekleme
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  # Sabır değerini 3 olarak belirledik

# Trainer nesnesi oluştur
trainer = Trainer(
    model=model,                         # Eğitilecek model
    args=training_args,                  # Eğitim argümanları
    train_dataset=train_dataset,         # Eğitim veri kümesi
    eval_dataset=eval_dataset,           # Doğrulama veri kümesi
    tokenizer=tokenizer,                 # Tokenizer
    callbacks=[early_stopping_callback]  # Erken durdurma callback'i ekle
)

# TensorBoard ile eğitim sürecini izlemek için
# TensorBoard'ı başlatmak için aşağıdaki komutları çalıştırın:
# %load_ext tensorboard
# %tensorboard --logdir ./logs

# Modeli eğit
trainer.train()

# Modeli kaydet
model.save_pretrained("/content/hate_speech_model")
tokenizer.save_pretrained("/content/hate_speech_model")
