# Metin Gömmeleri (Text Embeddings) Öğrenme Raporu: Teoriden Uygulamaya

## 1. Metin Gömmelerinin (Text Embeddings) Temelleri

Metin gömmeleri (text embeddings), yapay zekanın insan dilini anlamlandırmak için kullandığı en temel ve büyüleyici araçlardan biridir. Özünde bu teknoloji, bir metin dizesini (kelime, cümle veya paragraf) bilgisayarların işleyebileceği sayısal bir vektöre dönüştürür.

Google'ın `textembedding-gecko@001` modelini ele aldığımızda, bu modelin her girişi 768 boyutlu bir vektöre dönüştürdüğünü görürüz. Buradaki "768 boyut", metnin anlamını temsil eden 768 farklı özelliğin (feature) matematiksel olarak ifade edilmesidir. Bu yüksek boyutluluk, modelin dilin içindeki ince nüansları, bağlamı ve karmaşık anlam ilişkilerini yakalamasını sağlar. Bu sayılar rastgele diziler değil, dilin geometrik bir haritasıdır.

### Vektör Yapısı (Örnek Gösterim)

"Hayat" (Life) gibi bir kavramın 768 boyutlu dünyasındaki ilk 10 elemanına göz atalım:

| Boyut | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Değer** | 0.002 | -0.015 | 0.042 | -0.008 | 0.011 | -0.023 | 0.031 | 0.005 | -0.012 | 0.019 |

> **Öğrenci İçin "Neden Önemli?":**  
> Yapay zekanın "yeni elektrik" olduğu bu çağda, metin gömmeleri bu elektriği anlamlı bir güce dönüştüren transformatörler gibidir. Sadece kelimelerin harflerini değil, onların taşıdığı anlamı matematiksel bir düzleme taşıyarak makinelerin dünyayı bizim gibi anlamasını sağlar.

---

## 2. Semantik Benzerlik ve Cosine Similarity Mekanizması

İki metnin birbirine ne kadar benzediğini ölçmek için en yaygın kullanılan yöntem **Kosinüs Benzerliği (Cosine Similarity)** algoritmasıdır. Matematiksel olarak bu işlem, iki vektörün normalize edilip nokta çarpımının (dot product) alınmasıdır. Vertex AI üzerindeki `TextEmbeddingModel` bu hesaplamaları arka planda otomatik olarak yürütür.

### Karşılaştırmalı Analiz: Anlam Benzerliği Skorları

| Metin 1 | Metin 2 | Benzerlik Skoru |
| :--- | :--- | :--- |
| "What is the meaning of life?" | "How does one spend their time well on Earth?" | **0.655** |
| "What is the meaning of life?" | "Would you like a salad?" | **0.540** |
| "How does one spend their time well on Earth?" | "Would you like a salad?" | **0.520** |

* **Kritik Analiz:** İlk iki cümle arasında tek bir ortak kelime bile olmamasına rağmen model, aralarındaki varoluşsal bağı yakalayarak en yüksek skoru (0.655) üretmiştir. Buna karşın, bir salata teklifi ile hayatın anlamı arasındaki semantik bağ çok daha zayıf kalmıştır (0.540).
* **Kritik Çıkarım:** 768 boyutlu bir uzayda hiçbir vektör birbirine tam olarak dik (90 derece) değildir. Skorlar 0 ile 1 arasında ama dar bir aralıkta toplanır. Bu uzayda mutlak değerlerden ziyade **göreceli değerler** (hangisinin diğerinden daha yüksek olduğu) kritik önem taşır.

---

## 3. Kelime Gömmeleri (Word Embeddings) ve Cümle Gömmeleri Karşılaştırması

Geleneksel NLP yöntemlerinde "the", "in", "for" gibi durma kelimeleri (stop words) genellikle atılır ve kalan kelimelerin ortalaması alınarak bir cümle vektörü oluşturulurdu. Ancak bu "torba kelime" (bag of words) mantığı dilin bağlamsal yapısını kaybeder.

### Kelime Sırasının Önemi (Vaka Analizi)

* **Örnek A:** "the kids play in the park" *(çocuklar parkta oynuyor)*
* **Örnek B:** "the play was for kids in the park" *(tiyatro oyunu parktaki çocuklar içindi)*

Her iki cümle de `kids`, `play`, `park` anahtar kelimelerini içerir. Eski usul ortalama alma yöntemiyle bu iki cümle aynı vektörü üretirdi. Oysa `Gecko-001` gibi modern Transformer tabanlı modeller, kelime sırasını ve bağlamı analiz ederek bu iki farklı anlamı birbirinden net şekilde ayırt eder.

> **Öğrenme İpucu:**  
> Kelime bazlı ortalama alma (Word2Vec dönemi), bağlamı kaybeden ilkel bir yaklaşımdır. Modern modeller tüm cümleyi bir bütün olarak ele alarak kelimeler arasındaki ilişkiyi korur.

---

## 4. Vertex AI ile Python Üzerinde Uygulama Adımları

### Ortam Hazırlığı ve Kimlik Doğrulama

Kütüphaneyi kurarken `pip install google-cloud-aiplatform` komutu kullanılır, ancak Python içinde `vertexai` modülü üzerinden çağrılır:

    import vertexai
    from vertexai.language_models import TextEmbeddingModel

    # Proje ilklendirme
    vertexai.init(project="PROJECT_ID", location="us-central1")

### Model Yükleme ve Gömme Elde Etme

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(["What is the meaning of life?"])

    # İlk 10 vektör değerini görelim
    vector_values = embeddings[0].values
    print(vector_values[:10])

### Uygulama Akış Şeması

1. **Kütüphane Erişimi:** `google-cloud-aiplatform` paketinin kurulması.
2. **Kimlik Doğrulama:** `vertexai.init` ile proje ve konum bilgilerinin tanımlanması.
3. **Model Çağırma:** `from_pretrained` ile Gecko-001 modelinin yüklenmesi.
4. **Vektörizasyon:** `get_embeddings` ile metnin sayısal temsile dökülmesi.
5. **Analiz:** Çıktıların benzerlik, kümeleme veya sınıflandırma görevlerinde işlenmesi.

---

## 5. Kurs Müfredatı, Uygulama Alanları ve Gelecek Vizyonu

Metin gömmeleri endüstride geniş bir kullanım alanına sahiptir:

* **Kümeleme (Clustering):** Milyonlarca dokümanı manuel müdahale olmadan konularına göre gruplandırmak.
* **Sınıflandırma (Classification):** Müşteri geri bildirimlerini anında ilgili kategorilere atamak.
* **Aykırı Değer Tespiti (Outlier Detection):** Veri setindeki alakasız veya bozuk verileri temizlemek.
* **Semantik Arama ve Soru-Cevap (Semantic Search & Q&A):** Kullanıcının birebir kelimelerine değil, niyetine dayalı bilgi getirme sistemleri.

**Özet ve "So What?" Analizi:**  
Metin gömmeleri, sadece sayı dizileri değil; makinelerin insan dilini anlam düzeyinde kavramasını sağlayan bir köprüdür. Bu teknoloji sayesinde bilgisayarlar, kelimelerin sözlük anlamlarının ötesine geçerek niyeti algılayabilir hale gelmiştir.

> **Final Notu:** Bir sonraki aşama, bu vektörleri görselleştirerek verideki gizli desenleri keşfetmek ve ardından **RAG (Retrieval-Augmented Generation)** mimarilerine odaklanmaktır.
