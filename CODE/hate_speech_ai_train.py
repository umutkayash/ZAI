import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

# CUDA desteği kontrolü (GPU kullanımı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will be trained on: {device}")

# Veri kümesini yükleyin
ds = load_dataset("dileepa/Hate_Speech_Dataset")

# Eğitim ve test verilerini ayırın
train_dataset = ds['train']
eval_dataset = ds['test']

# Veri kümesinin yapısını kontrol edin (hangi sütunlar mevcut?)
print("Train Dataset Columns:", train_dataset.column_names)
print("Eval Dataset Columns:", eval_dataset.column_names)

# DistilBERT tokenizer'ı yükleyin
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenizasyon fonksiyonunu oluşturun (doğru sütun adıyla)
def tokenize_function(examples):
    # Veri türü kontrolü yaparak, yalnızca string olanları tokenize et
    if isinstance(examples['Content'], list):
        # Eğer Content sütunu bir liste ise, her bir öğeyi kontrol edip tokenize et
        return tokenizer(examples['Content'], padding="max_length", truncation=True, max_length=128)
    else:
        # Content sütunu tek bir string ise doğrudan tokenize et
        return tokenizer([examples['Content']], padding="max_length", truncation=True, max_length=128)

# Eksik veriyi kontrol et (boş metinleri filtrele)
train_dataset = train_dataset.filter(lambda x: isinstance(x['Content'], str) and len(x['Content']) > 0)
eval_dataset = eval_dataset.filter(lambda x: isinstance(x['Content'], str) and len(x['Content']) > 0)

# Tokenize işlemini gerçekleştirin
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Label'ları sayısal değere dönüştürme
def convert_labels(x):
    try:
        x['label'] = int(x['Label'])  # Sayısal dönüşüm (Label sütunu)
    except ValueError:  # Eğer 'Label' sayısal değilse, uygun bir değeri atama (örn. 0 veya 1)
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
    learning_rate=2e-6,                # Öğrenme oranı (optimize edilmiş)
    per_device_train_batch_size=32,    # Batch boyutunu optimize ettik (T4 GPU ile bu değer iyi çalışır)
    per_device_eval_batch_size=32,     # Değerlendirme batch boyutunu artırdık
    num_train_epochs=9,                # Epoch sayısını artırdık (5 epoch)
    weight_decay=0.01,                 # Ağırlık düşüşü (modelin overfitting yapmaması için)
    gradient_accumulation_steps=8,     # Küçük batch'lerde işlem yaparak toplamda daha büyük batch etkisi yaratabiliriz
    logging_dir='./logs',              # Log dosyaları
    logging_steps=50,                  # Loglamayı her 50 adımda bir yapıyoruz
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
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)  # Sabır değerini 2 olarak belirledik

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
model.save_pretrained("./hate_speech_model")
tokenizer.save_pretrained("./hate_speech_model")
