# Görev: OpenAI'ın (veya ücretsiz Hugging Face'in) embedding modelini 
# kullanarak 5 farklı cümlenin vektörünü çıkar. Kod ile iki cümlenin 
# birbirine ne kadar benzediğini hesaplat.

# Bu görevi Python kullanarak oldukça pratik bir şekilde halledebiliriz. 
# Hugging Face'in sentence-transformers kütüphanesi bu iş için tamamen ücretsiz, 
# yerel ortamında (lokal) çalışan ve oldukça güçlü bir standarttır.

# pip install sentence-transformers scikit-learn




from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Modeli yükle (Hugging Face'in ücretsiz ve popüler bir İngilizce/Çok dilli modeli)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 5 farklı cümleyi tanımla
sentences = [
    "Yapay zeka modelleri doğal dil işlemeyi çok kolaylaştırdı.",
    "Makine öğrenmesi algoritmaları metinleri vektörlere dönüştürür.",
    "Bugün hava çok güzel, dışarıda yürüyüş yapalım.",
    "Güneşli havalarda parkta vakit geçirmeyi seviyorum.",
    "Mikroservis mimarileri ile backend geliştirmek oldukça keyifli."
]

# 3. Cümlelerin embedding'lerini (vektörlerini) çıkar
embeddings = model.encode(sentences)

print(f"Toplam {len(embeddings)} cümlenin vektörü çıkarıldı.")
print(f"Her bir vektörün boyutu: {len(embeddings[0])}\n")

# 4. İki cümle arasındaki benzerliği hesapla (Kosinüs Benzerliği)
# Sklearn kütüphanesi 2 boyutlu (2D) array beklediği için reshape(1, -1) uyguluyoruz
vector1 = embeddings[0].reshape(1, -1)
vector2 = embeddings[1].reshape(1, -1)

# Benzerlik skoru hesaplama
similarity_score = cosine_similarity(vector1, vector2)[0][0]

print(f"Cümle 1: '{sentences[0]}'")
print(f"Cümle 2: '{sentences[1]}'")
print(f"-> Anlamsal Benzerlik Skoru: {similarity_score:.4f}")