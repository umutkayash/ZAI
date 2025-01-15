import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# CUDA desteği kontrolü (GPU kullanımı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will be used on: {device}")

# Model ve tokenizer'ı yükle
model_path = "./hate_speech_model"  # Eğitilen modelin kaydedildiği yol
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Metin tahmin fonksiyonu
def predict(text):
    # Metni tokenle
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    # Model ile tahmin yap
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Softmax ile olasılıkları hesapla
    probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]  # CPU'ya al ve numpy dizisine dönüştür
    return probabilities

# Kullanıcıdan girdi alma
user_input = input("Lütfen bir metin girin: ")

# Tahmin yap
probabilities = predict(user_input)

# Sonuçları yazdır
label_0_prob = probabilities[0] * 100  # sınıf 0 için olasılık
label_1_prob = probabilities[1] * 100  # sınıf 1 için olasılık

print(f"\nTahmin Sonuçları:")
print(f"Nefret Söylemi (Label 1) Olasılığı: {label_1_prob:.2f}%")
print(f"Nefret Söylemi Olmayan (Label 0) Olasılığı: {label_0_prob:.2f}%")

# En yüksek olasılık ile tahmin edilen sınıfı belirt
if label_1_prob > label_0_prob:
    print("\nBu metin nefret söylemi içeriyor!")
else:
    print("\nBu metin nefret söylemi içermiyor.")
